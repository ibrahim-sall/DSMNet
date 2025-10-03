import numpy as np
import laspy
import sys
import rasterio
from rasterio.crs import CRS
import logging
import time
from datetime import datetime


def setup_logger(log_level=logging.INFO, log_file=None):
    """Setup logger with proper formatting and handlers."""
    logger = logging.getLogger('DSMNetEvaluator')
    logger.setLevel(log_level)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

logger = setup_logger()

try:
    import cupy as cp
    GPU_AVAILABLE = True
    logger.info("GPU acceleration available with CuPy.")
except ImportError:
    cp = np
    GPU_AVAILABLE = False
    logger.warning("CuPy not found. Using CPU (NumPy). Install 'cupy-cudaXX' for GPU acceleration.")


def load_las_points(las_file):
    """Load LAS point cloud and return points and metadata."""
    logger.info(f"Loading LAS file: {las_file}")
    start_time = time.time()
    
    las = laspy.read(las_file)
    points = np.vstack((las.x, las.y, las.z)).T
    
    # Attempt to get CRS info
    crs = None
    try:
        if las.header.global_encoding.wkt:
            crs = CRS.from_wkt(las.header.wkt)
            logger.info(f"Found WKT CRS in LAS file: {crs.to_string()}")
        else:
            # Fallback for older LAS formats if needed
            logger.warning("No WKT CRS found in LAS header.")
    except Exception as e:
        logger.warning(f"Could not read CRS from LAS file: {e}")

    logger.info(f"Loaded {len(points)} LAS points in {time.time() - start_time:.2f} seconds")
    
    return points, crs

def load_dsm_raster(dsm_file):
    """Load DSM raster and return the dataset object and its CRS."""
    logger.info(f"Loading DSM raster file: {dsm_file}")
    start_time = time.time()
    
    with rasterio.open(dsm_file) as src:
        logger.info(f"Loaded DSM in {time.time() - start_time:.2f} seconds. CRS: {src.crs}, Shape: {src.shape}")
        if src.crs is None:
            logger.warning("DSM file does not have a CRS defined.")
        # The file will be closed after this block, so we can't return src.
        # We will reopen it in the main block. This function is for validation.
    return True


def compute_rmse(las_points, dsm_file, las_crs=None, use_gpu=True):
    """
    Computes the Root Mean Square Error (RMSE) between a LAS point cloud and a DSM raster.
    Can use GPU for acceleration if CuPy is installed.

    Args:
        las_points (np.ndarray): Nx3 array of LAS points (x, y, z).
        dsm_file (str): Path to the DSM raster file.
        las_crs (rasterio.crs.CRS, optional): The CRS of the LAS points.
        use_gpu (bool): Whether to attempt to use the GPU.
    """
    start_time = time.time()
    xp = cp if (use_gpu and GPU_AVAILABLE) else np
    
    if use_gpu and GPU_AVAILABLE:
        logger.info("Attempting GPU-accelerated RMSE computation.")
    else:
        logger.info("Using CPU-based RMSE computation.")

    with rasterio.open(dsm_file) as dsm_raster:
        # --- CRS Check ---
        if dsm_raster.crs and las_crs:
            if not dsm_raster.crs.equals(las_crs):
                logger.warning(f"CRS mismatch! LAS: '{las_crs.to_string()}' vs DSM: '{dsm_raster.crs.to_string()}'.")
                logger.warning("Proceeding with comparison, but results may be invalid if projections differ.")
            else:
                logger.info("LAS and DSM CRS match.")
        else:
            logger.warning("Could not verify CRS match between LAS and DSM. Assuming they are aligned.")

        # --- Bounding Box Check ---
        las_bounds = (las_points[:, 0].min(), las_points[:, 1].min(), las_points[:, 0].max(), las_points[:, 1].max())
        dsm_bounds = dsm_raster.bounds
        logger.info(f"LAS Bounds:  ({las_bounds[0]:.2f}, {las_bounds[1]:.2f}, {las_bounds[2]:.2f}, {las_bounds[3]:.2f})")
        logger.info(f"DSM Bounds: ({dsm_bounds.left:.2f}, {dsm_bounds.bottom:.2f}, {dsm_bounds.right:.2f}, {dsm_bounds.top:.2f})")

        if use_gpu and GPU_AVAILABLE:
            logger.info("Loading data to GPU...")
            dsm_data = dsm_raster.read(1)
            dsm_gpu = xp.asarray(dsm_data)
            las_gpu = xp.asarray(las_points)
            
            transform = dsm_raster.transform
            inv_transform = ~transform

            # Convert world coordinates to pixel coordinates on GPU
            cols_gpu, rows_gpu = inv_transform * (las_gpu[:, 0], las_gpu[:, 1])

            # Round to nearest integer to get pixel indices
            cols_gpu = xp.floor(cols_gpu).astype(xp.int32)
            rows_gpu = xp.floor(rows_gpu).astype(xp.int32)

            # --- Filter points outside raster bounds ---
            height, width = dsm_gpu.shape
            valid_mask_gpu = (rows_gpu >= 0) & (rows_gpu < height) & (cols_gpu >= 0) & (cols_gpu < width)
            
            # --- Filter based on nodata value ---
            nodata_value = dsm_raster.nodata
            if nodata_value is not None:
                dsm_values_at_las_locs_gpu = dsm_gpu[rows_gpu[valid_mask_gpu], cols_gpu[valid_mask_gpu]]
                nodata_mask_gpu = dsm_values_at_las_locs_gpu != nodata_value
                # Combine masks
                valid_mask_gpu[valid_mask_gpu] = nodata_mask_gpu
            
            num_valid_points = int(xp.sum(valid_mask_gpu))
            if num_valid_points == 0:
                raise ValueError("No LAS points fall within the valid data area of the DSM raster.")
            
            logger.info(f"Found {num_valid_points} / {len(las_points)} LAS points within the DSM's valid area (GPU).")

            las_z_valid = las_gpu[valid_mask_gpu, 2]
            valid_rows_gpu = rows_gpu[valid_mask_gpu]
            valid_cols_gpu = cols_gpu[valid_mask_gpu]
            dsm_z_valid = dsm_gpu[valid_rows_gpu, valid_cols_gpu]

            # Move results back to CPU for logging if needed, or keep on GPU for calcs
            las_z_valid_cpu = las_z_valid.get()
            dsm_z_valid_cpu = dsm_z_valid.get()
            las_z_min, las_z_max = las_z_valid_cpu.min(), las_z_valid_cpu.max()
            dsm_z_min, dsm_z_max = dsm_z_valid_cpu.min(), dsm_z_valid_cpu.max()

        else: # CPU fallback
            logger.info("Querying DSM elevations at LAS point locations (CPU)...")
            las_xy = las_points[:, :2]
            dsm_values_at_las_locs = np.array([val[0] for val in dsm_raster.sample(las_xy)])
            
            nodata_value = dsm_raster.nodata
            if nodata_value is None:
                logger.warning("DSM has no nodata value defined. Assuming all queried points are valid.")
                valid_mask = np.ones(len(dsm_values_at_las_locs), dtype=bool)
            else:
                valid_mask = dsm_values_at_las_locs != nodata_value
                
            num_valid_points = np.sum(valid_mask)
            if num_valid_points == 0:
                raise ValueError("No LAS points fall within the valid data area of the DSM raster.")
                
            logger.info(f"Found {num_valid_points} / {len(las_points)} LAS points within the DSM's valid area (CPU).")

            las_z_valid = las_points[valid_mask, 2]
            dsm_z_valid = dsm_values_at_las_locs[valid_mask]
            las_z_min, las_z_max = las_z_valid.min(), las_z_valid.max()
            dsm_z_min, dsm_z_max = dsm_z_valid.min(), dsm_z_valid.max()

        # --- Analyze height ranges and compute error ---
        logger.info(f"LAS Z-range (matched points): [{las_z_min:.3f}, {las_z_max:.3f}]")
        logger.info(f"DSM Z-range (matched points): [{dsm_z_min:.3f}, {dsm_z_max:.3f}]")

        if dsm_z_max - dsm_z_min < 2.0 and las_z_max - las_z_min > 10.0:
            logger.warning("DSM appears to have a very small dynamic range (possibly normalized 0-1),")
            logger.warning("while LAS heights have a large range. This will likely result in a large error.")
            logger.warning("Ensure the DSM contains absolute elevation values, not normalized ones.")

        height_differences = las_z_valid - dsm_z_valid
        rmse = xp.sqrt(xp.mean(height_differences ** 2))
        mae = xp.mean(xp.abs(height_differences))
        mean_error = xp.mean(height_differences)
        
        # If on GPU, move results to CPU for printing
        if use_gpu and GPU_AVAILABLE:
            rmse, mae, mean_error = rmse.get(), mae.get(), mean_error.get()

        logger.info(f"Comparison finished in {time.time() - start_time:.2f} seconds.")
        logger.info(f"ME (LAS - DSM): {mean_error:.4f} (Bias)")
        logger.info(f"MAE: {mae:.4f}")
    
    return rmse, mean_error

def denormalize_dsm(normalized_dsm_path, output_path, las_points):
    """
    Denormalizes a DSM from [0, 1] range to absolute meter values using a LAS file's elevation range.

    Args:
        normalized_dsm_path (str): Path to the normalized input raster.
        output_path (str): Path to save the denormalized raster.
        las_points (np.ndarray): The loaded LAS point cloud (Nx3).
    """
    logger.info(f"Denormalizing {normalized_dsm_path} using LiDAR elevation range.")
    
    # Get min and max from LAS data
    las_min_z = las_points[:, 2].min()
    las_max_z = las_points[:, 2].max()
    height_range = las_max_z - las_min_z
    
    logger.info(f"LiDAR elevation range: [{las_min_z:.3f}, {las_max_z:.3f}]. Using range: {height_range:.3f}m.")

    with rasterio.open(normalized_dsm_path) as src:
        normalized_data = src.read(1)
        profile = src.profile
        
        # Denormalization formula
        denormalized_data = normalized_data * height_range + las_min_z
        
        # Update profile for writing
        profile.update(
            dtype=denormalized_data.dtype,
            compress='lzw'
        )
        
        logger.info(f"Saving denormalized DSM to: {output_path}")
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(denormalized_data, 1)
            
    logger.info("Denormalization complete.")
    return output_path


def correct_and_save_dsm(input_dsm_path, output_dsm_path, elevation_offset):
    """
    Adds an elevation offset to a DSM/NDSM raster and saves the result.

    Args:
        input_dsm_path (str): Path to the input raster.
        output_dsm_path (str): Path to save the corrected raster.
        elevation_offset (float): The elevation value to add to each pixel.
    """
    logger.info(f"Applying elevation offset of {elevation_offset:.4f} to {input_dsm_path}")
    
    with rasterio.open(input_dsm_path) as src:
        dsm_data = src.read(1)
        profile = src.profile
        
        # Check if the data is integer type, if so, the offset might require a float type
        if np.issubdtype(dsm_data.dtype, np.integer):
            logger.warning("Input DSM has integer data type. Converting to float32 for correction.")
            profile['dtype'] = 'float32'
            dsm_data = dsm_data.astype('float32')

        # Add the offset
        corrected_dsm_data = dsm_data + elevation_offset
        
        # Update profile for writing
        profile.update(
            dtype=corrected_dsm_data.dtype,
            compress='lzw' # A good default compression
        )
        
        logger.info(f"Saving corrected DSM to: {output_dsm_path}")
        with rasterio.open(output_dsm_path, 'w', **profile) as dst:
            dst.write(corrected_dsm_data, 1)
    
    logger.info("Corrected DSM saved successfully.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python error.py <dsm_raster.tif> <lidar_cloud.las>")
        sys.exit(1)
    
    dsm_file = sys.argv[1]
    las_file = sys.argv[2]

    try:
        # 1. Load LiDAR data to get points and CRS
        las_points, las_crs = load_las_points(las_file)
        
        # 2. Define path for the denormalized DSM and create it
        denormalized_dsm_file = dsm_file.replace('.tif', '_denormalized.tif')
        denormalize_dsm(dsm_file, denormalized_dsm_file, las_points)

        # 3. Compute RMSE using the new denormalized DSM
        # Pass use_gpu=True to the function. It will automatically fall back to CPU if needed.
        rmse, bias = compute_rmse(las_points, denormalized_dsm_file, las_crs, use_gpu=True)
        
        print("\n--- FINAL RESULTS ---")
        print(f"RMSE between denormalized DSM and LAS: {rmse:.4f}")
        print(f"Calculated Bias (Offset): {bias:.4f}")

        # 4. Create and save the final corrected DSM by applying the bias to the denormalized version
        output_dsm_file = denormalized_dsm_file.replace('.tif', '_corrected.tif')
        if output_dsm_file == denormalized_dsm_file: # Avoid overwriting
            output_dsm_file = f"{denormalized_dsm_file}_corrected"
            
        correct_and_save_dsm(denormalized_dsm_file, output_dsm_file, bias)

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        sys.exit(1)