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
logging.basicConfig(level=logging.DEBUG)
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
    
    if rgb_data.shape[0] <= 4:
      rgb_tile = np.transpose(rgb_data, (1, 2, 0))
    else:
      rgb_tile = rgb_data
    
    if rgb_tile.shape[2] == 4:
      rgb_tile = rgb_tile[:, :, :3] 
    elif rgb_tile.shape[2] == 1:
      rgb_tile = np.repeat(rgb_tile, 3, axis=2)
    
    rgb_tile = rgb_tile.astype(np.float32) / 255.0
    
  h, w = rgb_tile.shape[:2]
  if h < cropSize or w < cropSize:
    pad_h = max(0, cropSize - h)
    pad_w = max(0, cropSize - w)
    rgb_tile = np.pad(rgb_tile, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    h, w = rgb_tile.shape[:2]

  coordinates = []
  rgb_data = []
  for x1, x2, y1, y2 in sliding_window(rgb_tile, step=int(cropSize/6), window_size=(cropSize,cropSize)):
    coordinates.append([y1,y2,x1,x2])
    rgb_data.append(rgb_tile[y1:y2, x1:x2, :])

  pred = np.zeros([2, h, w])
  for crop in range(len(rgb_data)):
    cropRGB = rgb_data[crop]
    y1, y2, x1, x2 = coordinates[crop]
    prob_matrix = gaussian_kernel(cropRGB.shape[0], cropRGB.shape[1])
    dsm_output, sem_output, norm_output = net.call(cropRGB[np.newaxis, ...], training=False)
    
    if(correction):
      correctionInput = tf.concat([dsm_output, norm_output, sem_output, cropRGB[np.newaxis,...]], axis=-1)
      noise = autoencoder.call(correctionInput, training=False)
      dsm_output = dsm_output - noise
    
    dsm_output = dsm_output.numpy().squeeze()
    pred[0, y1:y2, x1:x2] += np.multiply(dsm_output, prob_matrix)
    pred[1, y1:y2, x1:x2] += prob_matrix

  gaussian = pred[1]
  pred = np.divide(pred[0], gaussian)
  pred = pred[:h, :w]
  
  # Scale output to meters
  # Adjust these values based on your training data range
  min_elevation = 0.0    # minimum elevation in your training area (meters)
  max_elevation = 100.0  # maximum elevation for Hamburg area (meters)
  
  # Scale from [0,1] back to actual elevation range in meters
  pred_meters = pred * (max_elevation - min_elevation) + min_elevation

  filename = os.path.splitext(os.path.basename(jp2_path))[0]
  tif_path = os.path.join(output_dir, filename + '.tif')
  
  # Save with georeference using rasterio (using float32 for elevation data)
  with rasterio.open(
    tif_path,
    'w',
    driver='GTiff',
    height=pred_meters.shape[0],
    width=pred_meters.shape[1],
    count=1,
    dtype=rasterio.float32,
    crs=crs,
    transform=transform,
    compress='lzw',
    nodata=-9999  # Set nodata value for elevation data
  ) as dst:
    dst.write(pred_meters.astype(np.float32), 1)
  
  logger.info(f"Saved {tif_path} with georeference")

logger.info("Final MSE loss  : " + str(total_mse/tilesLen))
logger.info("Final MAE loss  : " + str(total_mae/tilesLen))
logger.info("Final RMSE loss : " + str(total_rmse/tilesLen))
