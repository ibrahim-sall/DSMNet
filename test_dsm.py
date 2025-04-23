from config import *
from nets import *
from utils import *
from metrics import *  # Import the new metrics module

import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from tifffile import *
import logging  # Add the logging module

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.utils import *
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.applications.densenet import DenseNet121


# Call the test function based on the mode, either train, validation, or test
def test_dsm(mtl, dae, mode, save_test=False, verbose=False):
    # Set up logging configuration
    if verbose:
        logging.basicConfig(
            level=logging.INFO,  # Set the logging level
            format='%(asctime)s - %(levelname)s - %(message)s',  # Specify the format
            handlers=[
                logging.FileHandler(f"{dataset_name}_{'mtl' if not correction else 'dae'}_test_dsm_output.log", mode='w'),  # Log to file (w: overwrite mode; a: append mode)
                logging.StreamHandler()  # Also log to console
            ]
        )
        logger = logging.getLogger()
    else:
        logger = logging.getLogger('dummy')
        logger.addHandler(logging.NullHandler())
    
    # Collect the required file addresses for testing the entire end-to-end model
    test_rgb, test_sar, test_dsm, test_sem, test_count = collect_tilenames(mode)

    NUM_VAL_IMAGES = len(test_rgb)
    if verbose: 
        logger.info(f"\nNumber of {mode} samples: {NUM_VAL_IMAGES}\n")

    if mtl is None:
        # Define the backbone of MTL serving as the encoder section
        backboneNet = DenseNet121(
            weights='imagenet', 
            include_top=False, 
            input_tensor=Input(shape=(cropSize, cropSize, 3))
        )
        # Define the MTL model with necessary parameters and loading the saved weights thereafter
        mtl = MTL(
            backboneNet, 
            sem_flag=sem_flag, 
            norm_flag=norm_flag, 
            edge_flag=edge_flag
        )
        mtl.load_weights(predCheckPointPath)

    # Load the DAE model with saved weights if it is the case
    if correction and dae is None:
        dae = Autoencoder()
        dae.load_weights(corrCheckPointPath)

    # Initialize the test error metrics
    total_delta1 = total_delta2 = total_delta3 = 0.0
    total_mse = total_mae = total_rmse = 0.0
    confusion_matrix = np.zeros((sem_k, sem_k))  # Initialize confusion matrix
    total_time = 0

    # Get the number of test items
    tilesLen = len(test_rgb)
    # Estimate nDSM for every input test tile
    for tile in range(tilesLen):
        filename = '.'.join(test_rgb[tile].split('/')[-1].split('.')[:-1])
        if verbose: 
            logger.info(f"\nCurrent {mode} tile #{tile + 1}/{tilesLen}: {filename}")

        # Initialize the RGB pack of regular patches for DSM estimation procedure
        rgb_data = []
        # Get the respective coordinates for every individual input tile.
        coordinates = []

        # Load the RGB image and the respective ground truth DSM data
        rgb_tile, dsm_tile, sem_tile = load_test_tiles(test_rgb, test_sar, test_dsm, test_sem, tile)

        # Collect regular overlapping patches out of the input RGB image
        for x1, x2, y1, y2 in sliding_window(rgb_tile, step=int(cropSize / 6), window_size=(cropSize, cropSize)):
            coordinates.append([y1, y2, x1, x2])
            rgb_data.append(rgb_tile[y1:y2, x1:x2, :])

        # Initialize the DSM and SEM prediction tensors
        gaussian = np.zeros([rgb_tile.shape[0], rgb_tile.shape[1]])
        dsm_pred = np.zeros([rgb_tile.shape[0], rgb_tile.shape[1]])
        sem_pred = np.zeros([rgb_tile.shape[0], rgb_tile.shape[1], sem_k])

        # Compute DSM estimation
        start = time.time()
        for crop in range(len(rgb_data)):
            cropRGB = rgb_data[crop]
            y1, y2, x1, x2 = coordinates[crop]
            prob_matrix = gaussian_kernel(cropRGB.shape[0], cropRGB.shape[1])
            dsm_output, sem_output, norm_output, edge_output = mtl.call(cropRGB[np.newaxis, ...], mtl_head_mode, training=False)

            if correction:
                correctionList = []
                if sem_flag:
                    correctionList.append(sem_output)
                if norm_flag:
                    correctionList.append(norm_output)
                if edge_flag:
                    correctionList.append(edge_output)
                correctionList = [dsm_output] + correctionList + [cropRGB[np.newaxis, ...]]
                correctionInput = tf.concat(correctionList, axis=-1)

                noise = dae.call(correctionInput, training=False)
                dsm_output = dsm_output - noise
            
            dsm_output = dsm_output.numpy().squeeze()
            sem_output = sem_output.numpy().squeeze()

            dsm_pred[y1:y2, x1:x2] += np.multiply(dsm_output, prob_matrix)
            sem_pred[y1:y2, x1:x2, :] += np.multiply(sem_output, prob_matrix[:, :, np.newaxis])
            gaussian[y1:y2, x1:x2] += prob_matrix

        end = time.time()

        # Finalize Gaussian smoothing
        dsm_pred = np.divide(dsm_pred, gaussian)
        sem_pred = np.divide(sem_pred, gaussian[:, :, np.newaxis])

        # Calculate SEM metrics - accumulate confusion matrix
        sem_pred = convert_sem_onehot_to_annotation(sem_pred)
        confusion_matrix += update_confusion_matrix(sem_pred, sem_tile)

        # Fuse DSM prediction with semantic segmentation mask for binary classification tasks
        # This restricts height values to only appear for the class of interest (e.g., buildings)
        if binary_classification_flag:
            # Create binary mask for the class of interest from predictions
            pred_mask = (sem_pred == 1).astype(np.float32)
            
            # Create binary mask from ground truth segmentation
            gt_mask = (sem_tile == 1).astype(np.float32)
            
            # Store original values for logging purposes
            dsm_tile_original = dsm_tile.copy()
            
            # Apply predicted mask to DSM predictions
            dsm_pred = dsm_pred * pred_mask
            
            # Apply ground truth mask to ground truth DSM
            dsm_tile = dsm_tile * gt_mask
            
            if verbose:
                logger.info(f"Applied binary segmentation masks:")
                logger.info(f"  - Prediction mask to predicted DSM")
                logger.info(f"  - Ground truth mask to ground truth DSM")
                
                # Calculate statistics on the predicted mask
                pred_masked_percentage = 100.0 * (1.0 - np.mean(pred_mask))
                logger.info(f"Masked out {pred_masked_percentage:.2f}% of pixels in predicted DSM")
                
                # Calculate statistics on the ground truth mask
                gt_masked_percentage = 100.0 * (1.0 - np.mean(gt_mask))
                logger.info(f"Masked out {gt_masked_percentage:.2f}% of pixels in ground truth DSM")
                
                # Calculate how many ground truth pixels had non-zero values in background
                nonzero_bg = np.sum((gt_mask == 0) & (dsm_tile_original > 0))
                if nonzero_bg > 0:
                    bg_percentage = 100.0 * nonzero_bg / np.sum(gt_mask == 0)
                    logger.info(f"Found {nonzero_bg} non-zero background pixels in ground truth ({bg_percentage:.2f}%)")

        # Calculate DSM metrics
        total_delta1, total_delta2, total_delta3, \
        total_mse, total_mae, total_rmse, _, _ = compute_dsm_metrics(
            verbose=verbose,
            logger=logger,
            total_delta1=total_delta1,
            total_delta2=total_delta2,
            total_delta3=total_delta3,
            total_mse=total_mse,
            total_mae=total_mae,
            total_rmse=total_rmse,
            dsm_tile=dsm_tile,
            dsm_pred=dsm_pred
        )

        # Keep track of computation time
        tile_time = end - start
        total_time += tile_time
        if verbose: logger.info("Tile time  : " + str(tile_time))
        
        if save_test:            
            # Save the predicted DSM 
            dsm_pred = Image.fromarray(dsm_pred)
            subfolder = (
                f"dsm_"
                f"{1 if sem_flag else 0}"
                f"{1 if norm_flag else 0}"
                f"{1 if edge_flag else 0}"
                f"{'+' if correction else ''}"
            )
            dsm_output_dir = f"./output/{dataset_name}/{sar_indicator}/{subfolder}/"
            if not os.path.exists(dsm_output_dir):
                os.makedirs(dsm_output_dir)
            dsm_file_path = os.path.join(dsm_output_dir, filename + '.tif')
            dsm_pred.save(dsm_file_path)
            
            # Save the predicted SEM with SAR indicator
            sem_pred = Image.fromarray(sem_pred)
            sem_output_dir = f"./output/{dataset_name}/{sar_indicator}/sem/"
            if not os.path.exists(sem_output_dir):
                os.makedirs(sem_output_dir)
            sem_file_path = os.path.join(sem_output_dir, filename + '.tif')
            sem_pred.save(sem_file_path)

    # Calculate final segmentation metrics from accumulated confusion matrix
    iou_per_class, f1_per_class, precision_per_class, recall_per_class, miou, overall_accuracy, FWIoU = \
        calculate_segmentation_metrics_from_confusion_matrix(confusion_matrix)

    # Calculate other averages
    avg_mse = total_mse / tilesLen
    avg_mae = total_mae / tilesLen
    avg_rmse = total_rmse / tilesLen
    avg_delta1 = total_delta1 / tilesLen
    avg_delta2 = total_delta2 / tilesLen
    avg_delta3 = total_delta3 / tilesLen
    
    if verbose:
        # Calculate means for all metrics
        dsm_metrics_str = (
            f"DSM Regression Metrics:\n"
            f"    MSE:    {avg_mse:.6f}\n"
            f"    MAE:    {avg_mae:.6f}\n"
            f"    RMSE:   {avg_rmse:.6f}\n"
            f"    Delta1: {avg_delta1:.6f}\n"
            f"    Delta2: {avg_delta2:.6f}\n"
            f"    Delta3: {avg_delta3:.6f}\n"
        )

        # Format all segmentation metrics using the utility function
        sem_metrics_str = format_segmentation_metrics(
            iou_per_class=iou_per_class,
            f1_per_class=f1_per_class,
            precision_per_class=precision_per_class,
            recall_per_class=recall_per_class,
            miou=miou,
            overall_accuracy=overall_accuracy,
            FWIoU=FWIoU
        )
        
        # Log all metrics with clear sections
        logger.info(
            f"\nEvaluation Results for {test_count} Test Samples\n"
            f"{'='*50}\n"
            f"{dsm_metrics_str}\n"
            f"{'='*50}\n"
            f"Semantic Segmentation Metrics:\n{sem_metrics_str}\n"
            f"{'='*50}\n"
            f"Test process finished in {total_time:.6f} sec.\n"
        )

    # Return a comprehensive set of metrics (both regression and segmentation)
    return {
        # Regression metrics
        "mse": avg_mse,
        "mae": avg_mae, 
        "rmse": avg_rmse,
        "delta1": avg_delta1,
        "delta2": avg_delta2,
        "delta3": avg_delta3,
        
        # Segmentation metrics - class metrics first (following config.py organization)
        "iou_per_class": iou_per_class,
        "precision_per_class": precision_per_class,
        "recall_per_class": recall_per_class,
        "f1_per_class": f1_per_class,
        # Scalar/overall metrics next
        "miou": miou,
        "overall_accuracy": overall_accuracy,
        "fwiou": FWIoU,
        
        # Other information
        "time": total_time,
        "count": test_count
    }


if __name__ == '__main__':
    metrics = test_dsm(mtl=None, dae=None, mode='test', save_test=True, verbose=True)

