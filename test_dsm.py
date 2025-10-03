#MAHDI ELHOUSNI, WPI 2020

import numpy as np
import cv2
import utils
import time
import matplotlib.pyplot as plt
import glob
import os

import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras.utils import *
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.applications.densenet import DenseNet121
from PIL import Image

from nets import *
from utils import *
from tifffile import *
from tqdm import tqdm

import sys
import logging
import rasterio
from rasterio.transform import from_bounds

datasetName='Vaihingen'

correction=True
sem_flag = True

cropSize=320

predCheckPointPath='./checkpoints/'+datasetName+'/mtl.weights.h5'
corrCheckPointPath='./checkpoints/'+datasetName+'/refinement.weights.h5'

val_rgb, val_dsm, val_sem = collect_tilenames("val",datasetName)

NUM_VAL_IMAGES = len(val_rgb)

print("number of validation samples " + str(NUM_VAL_IMAGES))

backboneNet=DenseNet121(weights='imagenet', include_top=False, input_tensor=Input(shape=(cropSize,cropSize,3)))

net = MTL(backboneNet, datasetName)
random_input = np.zeros((1, cropSize, cropSize, 3), dtype=np.float32)
net(random_input, training=False)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.debug(f"Loading model weights from {predCheckPointPath}")
net.load_weights(predCheckPointPath)
logger.debug("Model weights loaded successfully")

if(correction):
  autoencoder = Autoencoder()
  if(datasetName=='Vaihingen'):
    num_classes = 6
  else:
    num_classes = 20
  
  correction_input_channels = 1 + 3 + num_classes + 3
  sample_input = np.zeros((1, cropSize, cropSize, correction_input_channels), dtype=np.float32)
  autoencoder(sample_input, training=False)
  autoencoder.load_weights(corrCheckPointPath)

tile_mse   = 0
total_mse  = 0

tile_rmse  = 0
total_rmse = 0

tile_mae   = 0
total_mae  = 0

tile_time  = 0
total_time = 0

target_dir = '/home/rha/Documents/ibhou/bologna/qgis/bologna_center'
tif_files = sorted(glob.glob(target_dir + '/*.tif'))

output_dir = './output/Bologna/'
os.makedirs(output_dir, exist_ok=True)

tilesLen = len(tif_files)

if tilesLen == 0:
  logger.error(f"No TIF files found in {target_dir}")
  sys.exit(1)

logger.info(f"Found {tilesLen} TIF files to process")

for tif_path in tqdm(tif_files, desc="Processing TIF files"):
  logger.info(f"Processing {tif_path}")
  
  with rasterio.open(tif_path) as src:
    rgb_data = src.read()
    transform = src.transform
    crs = src.crs
    logger.debug(f"Read image shape: {rgb_data.shape}, CRS: {crs}, Transform: {transform}")

    if rgb_data.shape[0] <= 4:
      rgb_tile = np.transpose(rgb_data, (1, 2, 0))
      logger.debug(f"Transposed rgb_data to shape: {rgb_tile.shape}")
    else:
      rgb_tile = rgb_data
      logger.debug(f"rgb_tile shape (no transpose): {rgb_tile.shape}")

    if rgb_tile.shape[2] == 4:
      rgb_tile = rgb_tile[:, :, :3]
      logger.debug("Removed alpha channel from rgb_tile")
    elif rgb_tile.shape[2] == 1:
      rgb_tile = np.repeat(rgb_tile, 3, axis=2)
      logger.debug("Repeated single channel to 3 channels in rgb_tile")

    rgb_tile = rgb_tile.astype(np.float32) / 255.0
    logger.debug(f"Normalized rgb_tile to float32 in range [0,1]")

  h, w = rgb_tile.shape[:2]
  logger.debug(f"Tile size after normalization: {h}x{w}")
  if h < cropSize or w < cropSize:
    pad_h = max(0, cropSize - h)
    pad_w = max(0, cropSize - w)
    rgb_tile = np.pad(rgb_tile, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    h, w = rgb_tile.shape[:2]
    logger.debug(f"Padded rgb_tile to size: {h}x{w}")

  coordinates = []
  rgb_data = []
  # Use larger step size for faster processing (less precision but much faster)
  step_size = int(cropSize/2)  # Changed from cropSize/6 to cropSize/2 for ~9x fewer crops
  for x1, x2, y1, y2 in sliding_window(rgb_tile, step=step_size, window_size=(cropSize,cropSize)):
    coordinates.append([y1, y2, x1, x2])
    rgb_data.append(rgb_tile[y1:y2, x1:x2, :])
  logger.debug(f"Generated {len(rgb_data)} crops for sliding window inference with step size {step_size}")

  pred = np.zeros([2, h, w])
  sem_pred = np.zeros([2, h, w, num_classes])
  for crop in tqdm(range(len(rgb_data)), desc=f"Processing crops for {os.path.basename(tif_path)}", leave=False):
    cropRGB = rgb_data[crop]
    y1, y2, x1, x2 = coordinates[crop]
    prob_matrix = gaussian_kernel(cropRGB.shape[0], cropRGB.shape[1])
    logger.debug(f"Crop {crop}: coords=({y1},{y2},{x1},{x2}), crop shape={cropRGB.shape}")

    dsm_output, sem_output, norm_output = net.call(cropRGB[np.newaxis, ...], training=False)
    logger.debug(f"Network outputs shapes: dsm={dsm_output.shape}, sem={sem_output.shape}, norm={norm_output.shape}")

    if correction:
      correctionInput = tf.concat([dsm_output, norm_output, sem_output, cropRGB[np.newaxis, ...]], axis=-1)
      logger.debug(f"Correction input shape: {correctionInput.shape}")
      noise = autoencoder.call(correctionInput, training=False)
      logger.debug(f"Noise shape: {noise.shape}")
      dsm_output = dsm_output - noise
      logger.debug("Applied correction to dsm_output")

    dsm_output = dsm_output.numpy().squeeze()
    sem_output_np = sem_output.numpy().squeeze()
    
    # Accumulate DSM predictions
    pred[0, y1:y2, x1:x2] += np.multiply(dsm_output, prob_matrix)
    pred[1, y1:y2, x1:x2] += prob_matrix
    
    # Accumulate semantic predictions
    for c in range(num_classes):
      sem_pred[0, y1:y2, x1:x2, c] += np.multiply(sem_output_np[:, :, c], prob_matrix)
    sem_pred[1, y1:y2, x1:x2, :] += prob_matrix[:, :, np.newaxis]

  gaussian = pred[1]
  pred = np.divide(pred[0], gaussian)
  pred = pred[:h, :w]
  logger.debug(f"Final DSM prediction shape: {pred.shape}")
  
  # Process semantic segmentation
  sem_gaussian = sem_pred[1]
  sem_final = np.divide(sem_pred[0], sem_gaussian)
  sem_final = sem_final[:h, :w, :]
  
  # Get building class (assuming buildings are class 5 for Vaihingen dataset)
  if datasetName == 'Vaihingen':
    building_class_idx = 5  # Building class for Vaihingen
  else:
    building_class_idx = 1  # Adjust for other datasets
  
  # Create building footprint mask
  building_footprint = np.argmax(sem_final, axis=2) == building_class_idx
  logger.debug(f"Final semantic prediction shape: {sem_final.shape}")
  logger.debug(f"Building footprint shape: {building_footprint.shape}")

  filename = os.path.splitext(os.path.basename(tif_path))[0]
  output_tif_path = os.path.join(output_dir, filename + '.tif')
  building_output_path = os.path.join(output_dir, filename + '_buildings.tif')
  semantic_output_path = os.path.join(output_dir, filename + '_semantic_rgb.tif')

  # Save DSM with georeference
  with rasterio.open(
    output_tif_path,
    'w',
    driver='GTiff',
    height=pred.shape[0],
    width=pred.shape[1],
    count=1,
    dtype=rasterio.float32,
    crs=crs,
    transform=transform,
    compress='lzw',
    nodata=-9999) as dst:
    dst.write(pred.astype(np.float32), 1)
    logger.debug(f"Written DSM prediction to {output_tif_path}")

  # Save building footprints
  with rasterio.open(
    building_output_path,
    'w',
    driver='GTiff',
    height=building_footprint.shape[0],
    width=building_footprint.shape[1],
    count=1,
    dtype=rasterio.uint8,
    crs=crs,
    transform=transform,
    compress='lzw') as dst:
    dst.write(building_footprint.astype(np.uint8), 1)
    logger.debug(f"Written building footprints to {building_output_path}")

  # Save semantic RGB visualization
  if datasetName == 'Vaihingen':
    color_map = np.array([
      [255, 255, 255],  
      [0, 0, 255],      
      [0, 255, 255],    
      [0, 255, 0],      
      [255, 255, 0],    
      [255, 0, 0]       
    ])
  else:
    color_map = np.random.randint(0, 255, (num_classes, 3))
  
  class_map = np.argmax(sem_final, axis=2)
  semantic_rgb = color_map[class_map]
  
  with rasterio.open(
    semantic_output_path,
    'w',
    driver='GTiff',
    height=semantic_rgb.shape[0],
    width=semantic_rgb.shape[1],
    count=3,
    dtype=rasterio.uint8,
    crs=crs,
    transform=transform,
    compress='lzw') as dst:
    for i in range(3):
      dst.write(semantic_rgb[:, :, i].astype(np.uint8), i+1)
    logger.debug(f"Written semantic RGB to {semantic_output_path}")

  logger.info(f"Saved DSM: {output_tif_path}")
  logger.info(f"Saved buildings: {building_output_path}")  
  logger.info(f"Saved semantic RGB: {semantic_output_path}")

logger.info("Final MSE loss  : " + str(total_mse/tilesLen))
logger.info("Final MAE loss  : " + str(total_mae/tilesLen))
logger.info("Final RMSE loss : " + str(total_rmse/tilesLen))
