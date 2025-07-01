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

datasetName='Vaihingen'

correction=False
sem_flag = True

cropSize=320

predCheckPointPath='./checkpoints/'+datasetName+'/mtl.weights.h5'
corrCheckPointPath='./checkpoints/'+datasetName+'/refinement.h5'

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
  autoencoder=Autoencoder()
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
  img = Image.open(jp2_path)
  if img.mode == 'RGBA':
    img = img.convert('RGB')
  elif img.mode == 'L':
    img = img.convert('RGB')
  rgb_tile = np.array(img) / 255.0
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
    dsm_output = dsm_output.numpy().squeeze()
    pred[0, y1:y2, x1:x2] += np.multiply(dsm_output, prob_matrix)
    pred[1, y1:y2, x1:x2] += prob_matrix

  # gaussian = pred[1]f
  pred = np.divide(pred[0], gaussian)
  pred = pred[:h, :w]

  filename = os.path.splitext(os.path.basename(jp2_path))[0]
  tif_path = os.path.join(output_dir, filename + '.tif')
  pred_img = Image.fromarray((pred * 255).astype(np.uint8))
  pred_img.save(tif_path)
  logger.info(f"Saved {tif_path}")

logger.info("Final MSE loss  : " + str(total_mse/tilesLen))
logger.info("Final MAE loss  : " + str(total_mae/tilesLen))
logger.info("Final RMSE loss : " + str(total_rmse/tilesLen))
