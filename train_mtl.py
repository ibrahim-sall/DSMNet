# MAHDI ELHOUSNI, WPI 2020
# Altered by Ahmad Naghavi, OzU 2024

from config import *  # Must come first to make metric_names available

import numpy as np
import random
import logging

from datetime import datetime
import time
from os import path
from skimage import io

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.utils import *
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.losses import *
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.losses import MeanSquaredError, CategoricalCrossentropy, Huber

from nets import *
from utils import *
from metrics import *  # Import the new metrics module
from test_dsm import test_dsm

import matplotlib
# Always use Agg backend for consistent non-interactive plotting
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format='%(asctime)s - %(levelname)s - %(message)s',  # Specify the format
    handlers=[
        logging.FileHandler(f"{dataset_name}_mtl_train_output.log", mode='w'),  # Log to file (w: overwrite mode; a: append mode)
        logging.StreamHandler()  # Also log to console
    ]
)
logger = logging.getLogger()

# Keep track of the computation time
total_start = time.time()
current_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
logger.info(f'\nMTL training on {dataset_name} just started at {current_datetime}!\n')

# Collect the required file addresses for training the MTL model
train_rgb, train_sar, train_dsm, train_sem, _ = collect_tilenames("train")
# Merge train and valid sets if there is no validation while training the model
if not train_valid_flag:
    valid_rgb, valid_sar, valid_dsm, valid_sem, _ = collect_tilenames("valid")
    train_rgb.extend(valid_rgb)
    train_sar.extend(valid_sar)
    train_dsm.extend(valid_dsm)
    train_sem.extend(valid_sem)

# Update number of training samples and training iterations for datasets that are not from tile_mode group
if not large_tile_mode:
    mtl_training_samples = len(train_rgb)
    mtl_train_iters = int(mtl_training_samples / mtl_batchSize)
    mtl_log_freq = int(mtl_train_iters / 5)

NUM_TRAIN_IMAGES = len(train_rgb)
logger.info(f"Number of training samples: {NUM_TRAIN_IMAGES}\n")

# Define the backbone of MTL serving as the encoder section
backboneNet = DenseNet121(
    weights='imagenet', 
    include_top=False, 
    input_tensor=Input(shape=(cropSize, cropSize, 3))
    )
# Define the MTL model with necessary parameters
mtl = MTL(
    backboneNet, 
    sem_flag=sem_flag, 
    norm_flag=norm_flag, 
    edge_flag=edge_flag
    )
# Load saved weights in the beginning if there is any
if mtl_preload: 
    mtl.load_weights(predCheckPointPath)
# Freeze the backbone weights if necessary to save computation time.
# In such a way, only the decoder section will be updating during training.
if mtl_bb_freeze:
    for layer in backboneNet.layers: layer.trainable = False

# Define the loss function for regression and classification tasks
if reg_loss == 'mse':
    REG_LOSS = MeanSquaredError()
elif reg_loss == 'huber':
    REG_LOSS = Huber(delta=huber_delta)
CCE = CategoricalCrossentropy()

# Plot train/valid errors if the case
if train_valid_flag:
    # Create combined list of all metrics
    all_metrics = metric_names.copy()
    
    # Add scalar segmentation metrics
    for metric in segmentation_scalar_metrics:
        if metric not in all_metrics:
            all_metrics.append(metric)
    
    # Add per-class metrics
    for metric in segmentation_class_metrics:
        for class_idx in range(len(semantic_label_map)):
            all_metrics.append(f"{metric}_class{class_idx}")
    
    # Initialize metrics dictionaries with all available metrics
    valid_metrics = {metric: [] for metric in all_metrics}
    if plot_train_error:
        train_metrics = {metric: [] for metric in all_metrics}
    
    # Only create plots for the selected metrics
    fig, axes, lines = plot_train_valid_metrics(
        epoch=0,
        train_metrics={metric: [] for metric in plot_metrics} if plot_train_error else None,
        valid_metrics={metric: [] for metric in plot_metrics},
        plot_train=plot_train_error,
        model_type='MTL'
    )

# Initiate training
for epoch in range(1, mtl_numEpochs + 1):

    epoch_start = time.time()
    logger.info(f'\nepoch {epoch}/{mtl_numEpochs} just started!\n')

    # Update learning rate based on the decaying option
    if (mtl_lr_decay and epoch > 1): mtl_lr = mtl_lr / 2
    # Set the model optimizer options
    optimizer = tf.keras.optimizers.Adam(learning_rate=mtl_lr, beta_1=0.9)

    logger.info("Current epoch: " + str(epoch))
    logger.info("Current LR:    " + str(mtl_lr))

    # Reseting error metrics at the beginning of every epoch
    error_total = 0.0
    error_L1 = 0.0
    error_L2 = 0.0
    error_L3 = 0.0
    error_L4 = 0.0
    error_rmse = 0.0  # Add new RMSE tracker

    # Feed the model and updating it afterwards with a batch of training samples
    for iter in range(1, mtl_train_iters + 1):

        rgb_batch, dsm_batch, sem_batch, norm_batch, edge_batch = \
        generate_training_batches(train_rgb, train_sar, train_dsm, train_sem, iter, mtl_flag=True)

        # Call the MTL model and compute the loss functions
        with tf.GradientTape() as tape:
            dsm_out, sem_out, norm_out, edge_out = mtl.call(rgb_batch, mtl_head_mode, training=True)
            L1 = REG_LOSS(dsm_batch.squeeze(), tf.squeeze(dsm_out))  # For gradient computation
            
            # Compute per-sample DSM RMSE for comparison with test metrics
            abs_diff = tf.abs(tf.squeeze(dsm_out) - dsm_batch.squeeze())
            mse_per_sample = tf.reduce_mean(tf.square(abs_diff), axis=[1, 2])  # Mean over H,W dimensions
            rmse_per_sample = tf.sqrt(mse_per_sample)
            batch_rmse = tf.reduce_mean(rmse_per_sample)  # Average RMSE across batch
            
            L2 = CCE(sem_batch, sem_out) if sem_flag else tf.constant(0, dtype=tf.float32)
            L3 = REG_LOSS(norm_batch, norm_out) if norm_flag else tf.constant(0, dtype=tf.float32)
            L4 = REG_LOSS(edge_batch.squeeze(), tf.squeeze(edge_out)) if edge_flag else tf.constant(0, dtype=tf.float32)

            # Check for NaN values in loss components
            if tf.math.is_nan(L1) or tf.math.is_nan(L2) or tf.math.is_nan(L3) or tf.math.is_nan(L4):
                logger.error(f"NaN detected in loss values:")
                logger.error(f"L1 (DSM loss): {L1}")
                logger.error(f"L2 (Semantic loss): {L2}")
                logger.error(f"L3 (Normal loss): {L3}")
                logger.error(f"L4 (Edge loss): {L4}")
                logger.error(f"Current batch statistics:")
                logger.error(f"DSM output range: {tf.reduce_min(dsm_out)} to {tf.reduce_max(dsm_out)}")
                logger.error(f"DSM target range: {tf.reduce_min(dsm_batch)} to {tf.reduce_max(dsm_batch)}")
                raise ValueError("Training halted due to NaN values in loss computation")

            # Compute the overall loss according to scaling factors
            total_loss = w1 * L1 + w2 * L2 + w3 * L3 + w4 * L4
            logger.info(f'epoch: {epoch}/{mtl_numEpochs}\ttrain_iter: {iter}/{mtl_train_iters}\t'
                       f'total_loss: {total_loss:.6f}\tbatch_rmse: {batch_rmse:.6f}')

        # Update the model parameters
        grads = tape.gradient(total_loss, mtl.trainable_variables)
        optimizer.apply_gradients(zip(grads, mtl.trainable_variables))

        # Update error metrics
        error_total = error_total + total_loss.numpy()
        error_L1 = error_L1 + L1.numpy()
        error_rmse = error_rmse + batch_rmse.numpy()  # Add new DSM RMSE tracker
        error_L2 = error_L2 + L2.numpy()  
        error_L3 = error_L3 + L3.numpy()
        error_L4 = error_L4 + L4.numpy()

        # Compute average loss values after a number of iterations and log them
        if should_compute_metrics(iter, mtl_train_iters, mtl_log_freq):
            mtl_min_loss, last_epoch_saved, error_total, error_L1, error_L2, error_L3, error_L4, error_rmse = \
                compute_mtl_metrics(iter, mtl_train_iters, mtl_log_freq, last_epoch_saved, epoch, mtl_min_loss, 
                                  error_total, error_L1, error_L2, error_L3, error_L4, error_rmse, 
                                  logger, train_valid_flag, mtl, predCheckPointPath)

    # Calculate epoch runtime
    epoch_end = time.time()
    epoch_time = epoch_end - epoch_start
    logger.info(f'\nepoch {epoch}/{mtl_numEpochs} just finished after {epoch_time:.8f} sec!\n')

    # Compute train and validation errors after every epoch
    if train_valid_flag:
        valid_avg_rmse, train_metrics, valid_metrics = compute_validation_metrics(
            plot_train_error=plot_train_error,
            epoch=epoch,
            train_metrics=train_metrics if plot_train_error else None,
            valid_metrics=valid_metrics,
            fig=fig,
            axes=axes,
            lines=lines,
            logger=logger,
            mtl=mtl,
            dae=None,
            model_type='MTL',
            test_dsm_func=test_dsm  # Pass the test_dsm function here
        )

        # Early stopping check (after validation)
        should_stop, best_metric, patience_counter = handle_early_stopping(
            early_stop_flag=early_stop_flag,
            current_metric=get_metric_value(valid_metrics, eval_metric), 
            best_metric=best_metric,
            patience_counter=patience_counter,
            early_stop_patience=early_stop_patience,
            early_stop_delta=early_stop_delta,
            model=mtl,
            checkpoint_path=predCheckPointPath,
            epoch=epoch,
            logger=logger
        )
        
        if should_stop:
            break

# Calculate the total execution time
total_end = time.time()
total_time_seconds = total_end - total_start

minutes, seconds = divmod(total_time_seconds, 60)
hours, minutes = divmod(minutes, 60)
days, hours = divmod(hours, 24)

current_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
logger.info(f'\nMTL training on {dataset_name} just finished at {current_datetime}. Total time: '
      f'{int(days)} day(s) {int(hours)} hour(s) {int(minutes)} minute(s) {seconds:.8f} second(s)\n')

if train_valid_flag:
    # Save final plot and clean up
    plt.savefig(f'{dataset_name}_mtl_train_valid_metrics.png')
    logger.info(f"Final metrics plot saved to {dataset_name}_mtl_train_valid_metrics.png")
    plt.close(fig)

