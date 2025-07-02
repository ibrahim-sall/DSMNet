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

target_dir = '/home/rha/Documents/ibhou/hambourg/TDOP_20'
jp2_files = sorted(glob.glob(target_dir + '/*.jp2'))

output_dir = './output/hambourg/'
os.makedirs(output_dir, exist_ok=True)

tilesLen = len(jp2_files)

for jp2_path in jp2_files:
  logger.info(f"Processing {jp2_path}")
  
  with rasterio.open(jp2_path) as src:
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
  for x1, x2, y1, y2 in sliding_window(rgb_tile, step=int(cropSize/6), window_size=(cropSize,cropSize)):
    coordinates.append([y1, y2, x1, x2])
    rgb_data.append(rgb_tile[y1:y2, x1:x2, :])
  logger.debug(f"Generated {len(rgb_data)} crops for sliding window inference")

  pred = np.zeros([2, h, w])
  for crop in range(len(rgb_data)):
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
    pred[0, y1:y2, x1:x2] += np.multiply(dsm_output, prob_matrix)
    pred[1, y1:y2, x1:x2] += prob_matrix

  gaussian = pred[1]
  pred = np.divide(pred[0], gaussian)
  pred = pred[:h, :w]
  logger.debug(f"Final prediction shape: {pred.shape}")

  filename = os.path.splitext(os.path.basename(jp2_path))[0]
  tif_path = os.path.join(output_dir, filename + '.tif')

  # Save with georeference
  with rasterio.open(
    tif_path,
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
    logger.debug(f"Written prediction to {tif_path}")

  logger.info(f"Saved {tif_path} with georeference")

logger.info("Final MSE loss  : " + str(total_mse/tilesLen))
logger.info("Final MAE loss  : " + str(total_mae/tilesLen))
logger.info("Final RMSE loss : " + str(total_rmse/tilesLen))
