# MAHDI ELHOUSNI, WPI 2020
# Altered by Ahmad Naghavi, OzU 2024

from config import *  # Must come first to make metric_names available

import logging
import numpy as np
import random
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
from tensorflow.keras.losses import MeanSquaredError, Huber

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
        logging.FileHandler(f"{dataset_name}_dae_train_output.log", mode='w'),  # Log to file (w: overwrite mode; a: append mode)
        logging.StreamHandler()  # Also log to console
    ]
)
logger = logging.getLogger()

# Keep track of the computation time
total_start = time.time()
current_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
logger.info(f'\nAutoEncoder training on {dataset_name} just started at {current_datetime}!\n')

# Collect the required file addresses for training the DAE model
train_rgb, train_sar, train_dsm, _, _ = collect_tilenames("train")
# Merge train and valid sets if there is no validation while training the model
if not train_valid_flag:
    valid_rgb, valid_sar, valid_dsm, _, _ = collect_tilenames("valid")
    train_rgb.extend(valid_rgb)
    train_sar.extend(valid_sar)
    train_dsm.extend(valid_dsm)

# Update number of training samples and training iterations for datasets that are not from tile_mode group
if not large_tile_mode:
    dae_training_samples = len(train_rgb)
    dae_train_iters = int(dae_training_samples / dae_batchSize)
    dae_log_freq = int(dae_train_iters / 5)

NUM_TRAIN_IMAGES = len(train_rgb)
logger.info(f"Number of training samples: {NUM_TRAIN_IMAGES}\n")

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

# Define the loss function for regression and classification tasks
if reg_loss == 'mse':
    REG_LOSS = MeanSquaredError()
elif reg_loss == 'huber':
    REG_LOSS = Huber(delta=huber_delta)

# Define the DAE object for denoising the MTL outputs
dae = Autoencoder()
# Load saved weights in the beginning if there is any
if dae_preload: 
    dae.load_weights(corrCheckPointPath)

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
        model_type='DAE'
    )

# Initiate training
for epoch in range(1, dae_numEpochs + 1):

    epoch_start = time.time()
    logger.info(f'\nepoch {epoch}/{dae_numEpochs} just started!\n')

    # Update learning rate based on the decaying option
    if (dae_lr_decay and epoch > 1): dae_lr = dae_lr / 2
    # Set the model optimizer options
    optimizer = tf.keras.optimizers.Adam(learning_rate=dae_lr, beta_1=0.9)

    logger.info('Current epoch: ' + str(epoch))
    logger.info("Current LR:    " + str(dae_lr))

    # Reseting the error metric at the beginning of every epoch
    error_dae = 0.0
    error_rmse = 0.0  # Add RMSE tracker

    # Feed the model and updating it afterwards with a batch of training samples
    for iter in range(1, dae_train_iters + 1):

        rgb_batch, dsm_batch, _, _, _ = \
        generate_training_batches(train_rgb, train_sar, train_dsm, [], iter, mtl_flag=False)

        # Feed MTL with the selected batch and then feed its output to DAE
        dsm_out, sem_out, norm_out, edge_out = mtl.call(rgb_batch, mtl_head_mode, training=False)
        # Concatenate MTL outputs alongside the RGB input for the sake of DAE input
        correctionList = []
        if (sem_flag):
            correctionList.append(sem_out)
        if (norm_flag):
            correctionList.append(norm_out)
        if (edge_flag):
            correctionList.append(edge_out)
        correctionList = [dsm_out] + correctionList + [rgb_batch]
        correctionInput = tf.concat(correctionList, axis=-1)

        # Call the DAE model and compute the loss function 
        # DAE output is contemplated noise here for the MTL output which serves as the DSM first guess
        with tf.GradientTape() as tape:
            noise = dae.call(correctionInput, training=True)
            dsm_out = dsm_out - noise

            dae_loss = REG_LOSS(dsm_batch, dsm_out)
            
            # Calculate RMSE
            abs_diff = tf.abs(dsm_out - dsm_batch)
            mse_per_sample = tf.reduce_mean(tf.square(abs_diff), axis=[1, 2])
            rmse_per_sample = tf.sqrt(mse_per_sample)
            batch_rmse = tf.reduce_mean(rmse_per_sample)
            
            logger.info(f'epoch: {epoch}/{dae_numEpochs}\ttrain iter: {iter}/{dae_train_iters}\t'
                       f'DAE loss: {dae_loss:.6f}\tbatch_rmse: {batch_rmse:.6f}')

        # Update the model parameters
        grads = tape.gradient(dae_loss, dae.trainable_variables)
        optimizer.apply_gradients(zip(grads, dae.trainable_variables))

        # Update the error metric
        error_dae = error_dae + dae_loss.numpy()
        error_rmse = error_rmse + batch_rmse.numpy()

        # Compute average loss value for the denoised nDSM after a number of iterations and log it
        if should_compute_metrics(iter, dae_train_iters, dae_log_freq):
            dae_min_loss, last_epoch_saved, error_dae, error_rmse = \
                compute_dae_metrics(iter, dae_train_iters, dae_log_freq, last_epoch_saved, epoch, dae_min_loss,
                                  error_dae, error_rmse, logger, train_valid_flag, dae, corrCheckPointPath)

    # Calculate epoch runtime
    epoch_end = time.time()
    epoch_time = epoch_end - epoch_start
    logger.info(f'\nepoch {epoch}/{dae_numEpochs} just finished after {epoch_time:.8f} sec!\n')

    # Compute and plot train/valid errors
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
            mtl=None,
            dae=dae,
            model_type='DAE',
            test_dsm_func=test_dsm  # Pass the test_dsm function here
        )

        # Early stopping check
        should_stop, best_metric, patience_counter = handle_early_stopping(
            early_stop_flag=early_stop_flag,
            current_metric=get_metric_value(valid_metrics, eval_metric),
            best_metric=best_metric, 
            patience_counter=patience_counter,
            early_stop_patience=early_stop_patience,
            early_stop_delta=early_stop_delta,
            model=dae,
            checkpoint_path=corrCheckPointPath,
            epoch=epoch,
            logger=logger,
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
logger.info(f'\nAutoEncoder training on {dataset_name} just finished at {current_datetime}. Total time: '
      f'{int(days)} day(s) {int(hours)} hour(s) {int(minutes)} minute(s) {seconds:.8f} second(s)\n')

if train_valid_flag:
    # Save final plot and clean up
    plt.savefig(f'{dataset_name}_dae_train_valid_metrics.png')
    logger.info(f"Final metrics plot saved to {dataset_name}_dae_train_valid_metrics.png")
    plt.close(fig)

