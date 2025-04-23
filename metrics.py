# Metrics and evaluation utilities for DSMNet
# Ahmad Naghavi, OzU 2024

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import logging
from typing import Optional, List, Tuple
from config import *

def update_confusion_matrix(pred, target):
    """
    Update confusion matrix for semantic segmentation evaluation.
    Since inputs are already preprocessed by convert_sem_onehot_to_annotation,
    they will contain label values matching label_codes.
    
    Args:
        pred (np.ndarray): Predicted segmentation mask with values matching label_codes
        target (np.ndarray): Ground truth segmentation mask with values matching label_codes
        
    Returns:
        np.ndarray: Updated confusion matrix stats to be added to the main confusion matrix
    """
    num_classes = len(semantic_label_map)
    
    # Create inverse mapping from label values to indices
    label_to_idx = {v: k for k, v in semantic_label_map.items()}
    
    # Handle RGB tuples for datasets like Vaihingen
    if pred.ndim == 3:  # RGB case
        # Convert RGB arrays to tuples for mapping
        pred_tuples = [tuple(p) for p in pred.reshape(-1, 3)]
        target_tuples = [tuple(t) for t in target.reshape(-1, 3)]
        
        # Map RGB tuples to indices using the inverse semantic_label_map
        pred_idx = [label_to_idx[p] for p in pred_tuples]
        target_idx = [label_to_idx[t] for t in target_tuples]
    else:  # Single channel integer labels
        pred_idx = [label_to_idx[p] for p in pred.flat]
        target_idx = [label_to_idx[t] for t in target.flat]
    
    # Convert to numpy arrays for bincount operation
    pred_idx = np.array(pred_idx)
    target_idx = np.array(target_idx)
    
    # Update confusion matrix
    hist = np.bincount(
        num_classes * target_idx + pred_idx,
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)
    
    return hist


def calculate_segmentation_metrics_from_confusion_matrix(confusion_matrix, eps=1e-8):
    """
    Calculate segmentation metrics from a confusion matrix.
    
    Args:
        confusion_matrix (np.ndarray): Shape (num_classes, num_classes)
        eps (float): Small value to avoid division by zero
        
    Returns:
        tuple: 
            - iou_per_class (np.ndarray): IoU for each class
            - f1_per_class (np.ndarray): F1 score for each class
            - precision_per_class (np.ndarray): Precision for each class
            - recall_per_class (np.ndarray): Recall for each class
            - miou (float): Mean IoU across all classes
            - overall_accuracy (float): Overall accuracy metric
            - FWIoU (float): Frequency Weighted Intersection over Union
    """
    tp = np.diag(confusion_matrix)
    fp = confusion_matrix.sum(axis=0) - tp  # False positives
    fn = confusion_matrix.sum(axis=1) - tp  # False negatives
    
    union = tp + fp + fn
    intersection = tp
    
    iou_per_class = intersection / (union + eps)
    miou = np.nanmean(iou_per_class)
    
    precision_per_class = tp / (tp + fp + eps)
    recall_per_class = tp / (tp + fn + eps)
    
    f1_per_class = 2 * (precision_per_class * recall_per_class) / (precision_per_class + recall_per_class + eps)
    
    # Compute overall accuracy and frequency weighted IoU
    total = confusion_matrix.sum()
    overall_accuracy = np.diag(confusion_matrix).sum() / (total + eps)
    frequency = confusion_matrix.sum(axis=1) / (total + eps)
    FWIoU = (frequency * iou_per_class)[frequency > 0].sum()
    
    # Return vector metrics first, then scalar metrics at the end
    return iou_per_class, f1_per_class, precision_per_class, recall_per_class, miou, overall_accuracy, FWIoU


def format_segmentation_metrics(iou_per_class: np.ndarray,
                              f1_per_class: np.ndarray, 
                              precision_per_class: np.ndarray,
                              recall_per_class: np.ndarray,
                              miou: float,
                              overall_accuracy: float,
                              FWIoU: float) -> str:
    """
    Format segmentation metrics with proper class labels based on dataset type.
    
    Args:
        iou_per_class (np.ndarray): IoU values per class
        f1_per_class (np.ndarray): F1 scores per class
        precision_per_class (np.ndarray): Precision values per class
        recall_per_class (np.ndarray): Recall values per class
        miou (float): Mean IoU across all classes
        overall_accuracy (float): Overall accuracy metric
        FWIoU (float): Frequency Weighted Intersection over Union
        
    Returns:
        str: Formatted string containing both per-class and overall segmentation metrics
    """
    output_parts = []
    
    # Format per-class metrics
    output_parts.append("Per-class Metrics:")
    for cls_idx, label in semantic_label_map.items():
        if uses_rgb_labels:
            label_str = f"Class {cls_idx} RGB{label}"
        else:
            label_str = f"Class {cls_idx}"
            if isinstance(label, int):
                label_str += f" (Label {label})"
        
        metrics = [
            f"IoU: {iou_per_class[cls_idx]:.4f}",
            f"F1: {f1_per_class[cls_idx]:.4f}",
            f"Precision: {precision_per_class[cls_idx]:.4f}",
            f"Recall: {recall_per_class[cls_idx]:.4f}"
        ]
        
        output_parts.append(f"{label_str}:\n\t{' | '.join(metrics)}")
    
    # Add overall metrics
    output_parts.append("\nOverall Metrics:")
    output_parts.append(f"Mean IoU: {miou:.4f}")
    output_parts.append(f"Overall Accuracy: {overall_accuracy:.4f}")
    output_parts.append(f"Frequency Weighted IoU: {FWIoU:.4f}")
    
    return '\n'.join(output_parts)


def compute_dsm_metrics(
    verbose,
    logger,
    total_delta1,
    total_delta2,
    total_delta3,
    total_mse,
    total_mae,
    total_rmse,
    dsm_tile,
    dsm_pred
):
    """
    Compute Digital Surface Model (DSM) evaluation metrics for a single tile.
    This function calculates various error metrics between predicted and ground truth DSM tiles,
    including MSE, MAE, RMSE and delta accuracy metrics. It handles invalid values and updates
    running totals for batch processing.
    Args:
        verbose (bool): If True, logs detailed metrics for each tile
        logger: Logger object for output messages
        total_delta1 (float): Running total for delta1 accuracy metric
        total_delta2 (float): Running total for delta2 accuracy metric
        total_delta3 (float): Running total for delta3 accuracy metric
        total_mse (float): Running total for Mean Squared Error
        total_mae (float): Running total for Mean Absolute Error 
        total_rmse (float): Running total for Root Mean Squared Error
        dsm_tile (numpy.ndarray): Ground truth DSM tile
        dsm_pred (numpy.ndarray): Predicted DSM tile
    Returns:
        tuple:
            - total_delta1 (float): Updated running total for delta1
            - total_delta2 (float): Updated running total for delta2
            - total_delta3 (float): Updated running total for delta3
            - total_mse (float): Updated running total for MSE
            - total_mae (float): Updated running total for MAE
            - total_rmse (float): Updated running total for RMSE
            - dsm_tile (numpy.ndarray): Filtered ground truth DSM
            - dsm_pred (numpy.ndarray): Filtered predicted DSM
    Notes:
        - Delta metrics measure accuracy within thresholds of 1.25, 1.25^2, and 1.25^3
        - Zero or negative values are replaced with small positive values (1e-5)
        - Invalid pixels are filtered out before computation
    """
    # 1. Ensure both arrays have the same shape
    assert dsm_tile.shape == dsm_pred.shape, (
        f"Shape mismatch: dsm_tile shape {dsm_tile.shape}, "
        f"dsm_pred shape {dsm_pred.shape}"
    )

    # 2. Copy to avoid modifying the originals
    dsm_tile_ = dsm_tile.copy()
    dsm_pred_ = dsm_pred.copy()

    # 3. Handle invalid or zero values: the original approach based on the contest rules
    #    - Very small positive for zeros and negative in pred
    #    - Large sentinel (999) for negative pred -> Not used in this version!
    #    - Very small positive for non-positive ground truth

    # Calculate the average of zero and negative pixels for dsm_pred and dsm_tile
    zero_pixel_ratio_pred = len(dsm_pred_[dsm_pred_ == 0]) / dsm_pred_.size
    avg_neg_pred = np.mean(dsm_pred_[dsm_pred_ < 0])
    zero_pixel_ratio_tile = len(dsm_tile_[dsm_tile_ == 0]) / dsm_tile_.size
    avg_neg_tile = np.mean(dsm_tile_[dsm_tile_ < 0])
    
    # Log the ratios if verbose
    if verbose:
        logger.info(f"dsm_pred - ratio of zero pixels    : {zero_pixel_ratio_pred}")
        logger.info(f"dsm_pred - mean of negative pixels : {avg_neg_pred}")
        logger.info(f"dsm_tile - ratio of zero pixels    : {zero_pixel_ratio_tile}")
        logger.info(f"dsm_tile - mean of negative pixels : {avg_neg_tile}")

    # Replace zero or negative values to avoid division by zero or invalid ratios
    # dsm_pred_[dsm_pred_ == 0], dsm_pred_[dsm_pred_ < 0] = 1e-5, 999
    dsm_pred_[dsm_pred_ <= 0] = 1e-5
    dsm_tile_[dsm_tile_ <= 0] = 1e-5

    # 4. Create a valid mask (both arrays should have strictly positive values)
    valid_mask = (dsm_tile_ > 0) & (dsm_pred_ > 0)

    dsm_tile_ = dsm_tile_[valid_mask]
    dsm_pred_ = dsm_pred_[valid_mask]

    # 5. Check if there are any valid pixels left
    if len(dsm_tile_) == 0:
        if verbose:
            logger.warning("No valid pixels found in this tile!")
        return (
            total_delta1, total_delta2, total_delta3,
            total_mse, total_mae, total_rmse,
            dsm_tile_, dsm_pred_
        )
 
    # 6. Compute error metrics: MSE, MAE, RMSE
    abs_diff = np.abs(dsm_pred_ - dsm_tile_)
    tile_mse = np.mean(abs_diff ** 2)
    tile_mae = np.mean(abs_diff)
    tile_rmse = np.sqrt(tile_mse)

    # 7. Compute delta metrics
    #    max_ratio = max(pred / gt, gt / pred)
    max_ratio = np.maximum(dsm_pred_ / dsm_tile_, dsm_tile_ / dsm_pred_)
    tile_delta1 = np.mean(max_ratio < 1.25)
    tile_delta2 = np.mean(max_ratio < 1.25 ** 2)
    tile_delta3 = np.mean(max_ratio < 1.25 ** 3)

    # 8. Log tile-level metrics if verbose
    if verbose:
        logger.info(f"Tile MSE   : {tile_mse:.4f}")
        logger.info(f"Tile MAE   : {tile_mae:.4f}")
        logger.info(f"Tile RMSE  : {tile_rmse:.4f}")
        logger.info(f"Tile Delta1: {tile_delta1:.4f}")
        logger.info(f"Tile Delta2: {tile_delta2:.4f}")
        logger.info(f"Tile Delta3: {tile_delta3:.4f}")

    # 9. Update running totals
    total_mse  += tile_mse
    total_mae  += tile_mae
    total_rmse += tile_rmse

    total_delta1 += tile_delta1
    total_delta2 += tile_delta2
    total_delta3 += tile_delta3

    # 10. Return updated totals + filtered arrays
    return (
        total_delta1,
        total_delta2,
        total_delta3,
        total_mse,
        total_mae,
        total_rmse,
        dsm_tile_,
        dsm_pred_
    )


def compute_validation_metrics(
    plot_train_error,     # Whether to compute training metrics
    epoch,                # Current epoch number
    train_metrics,        # Dictionary of training metrics
    valid_metrics,        # Dictionary of validation metrics
    fig,                  # Current figure object
    axes,                 # List of subplot axes 
    lines,               # List of plot line objects
    logger,              # Logger instance
    mtl,                 # MTL model instance (or None for DAE validation)
    dae,                 # DAE model instance (or None for MTL validation)
    model_type='MTL',    # Type of model ('MTL' or 'DAE')
    test_dsm_func=None   # Function to compute DSM metrics
):
    """
    Compute and log validation metrics, update plots for model training.
    
    Args:
        plot_train_error (bool): Whether to compute training metrics
        epoch (int): Current epoch number
        train_metrics (dict): Dictionary of training metrics
        valid_metrics (dict): Dictionary of validation metrics
        fig (matplotlib.figure.Figure): Current figure object
        axes (list): List of subplot axes
        lines (list): List of plot line objects
        logger: Logger instance
        mtl: MTL model instance (or None for DAE validation)
        dae: DAE model instance (or None for MTL validation)
        model_type (str): Type of model ('MTL' or 'DAE')
        test_dsm_func (function): Function to compute DSM metrics
        
    Returns:
        tuple: Updated metrics dictionaries and validation RMSE
            - valid_avg_rmse (float): Validation RMSE for early stopping
            - train_metrics (dict): Updated training metrics
            - valid_metrics (dict): Updated validation metrics
    """
    logger.info(f"Computing average" +
            (f" train and " if plot_train_error else " ") +
            f"validation errors out of the partially trained model after epoch {epoch} ...\n")

    # Compute training metrics if enabled
    if plot_train_error:
        train_metrics_data = test_dsm_func(mtl, dae, mode='train')
        
        train_avg_mse = train_metrics_data['mse']
        train_avg_mae = train_metrics_data['mae']
        train_avg_rmse = train_metrics_data['rmse']
        train_miou = train_metrics_data['miou']
        train_iou_per_class = train_metrics_data['iou_per_class']
        train_avg_delta1 = train_metrics_data['delta1']
        train_avg_delta2 = train_metrics_data['delta2']
        train_avg_delta3 = train_metrics_data['delta3']
        train_time = train_metrics_data['time']
        train_count = train_metrics_data['count']
        
        # Format training metrics in a readable way
        train_metrics_str = (
            f"\n{'='*35}\n"
            f"Training Results (Epoch {epoch}) - {train_count} samples\n"
            f"{'='*35}\n"
            f"DSM Regression Metrics:\n"
            f"    MSE:    {train_avg_mse:.6f}\n"
            f"    MAE:    {train_avg_mae:.6f}\n"
            f"    RMSE:   {train_avg_rmse:.6f}\n"
            f"    Delta1: {train_avg_delta1:.6f}\n"
            f"    Delta2: {train_avg_delta2:.6f}\n"
            f"    Delta3: {train_avg_delta3:.6f}\n"
        )
        
        # Add segmentation metrics
        train_metrics_str += f"\nSegmentation Metrics:\n    Mean IoU: {train_miou:.6f}"
        
        # Add per-class IoU metrics if they exist
        if train_iou_per_class is not None and len(train_iou_per_class) > 0:
            if len(train_iou_per_class) <= 4:  # Only show per-class for small number of classes
                train_metrics_str += "\n    Per-class IoU: "
                for i, iou_val in enumerate(train_iou_per_class):
                    train_metrics_str += f"Class {i}: {iou_val:.4f}  "
        
        train_metrics_str += f"\n\nComputed in {train_time:.6f} sec"
        logger.info(train_metrics_str)

    # Compute validation metrics
    valid_metrics_data = test_dsm_func(mtl, dae, mode='valid')
    
    valid_avg_mse = valid_metrics_data['mse']
    valid_avg_mae = valid_metrics_data['mae']
    valid_avg_rmse = valid_metrics_data['rmse']
    valid_miou = valid_metrics_data['miou']
    valid_iou_per_class = valid_metrics_data['iou_per_class']
    valid_avg_delta1 = valid_metrics_data['delta1']
    valid_avg_delta2 = valid_metrics_data['delta2']
    valid_avg_delta3 = valid_metrics_data['delta3']
    valid_time = valid_metrics_data['time']
    valid_count = valid_metrics_data['count']
    
    # Format validation metrics in a readable way
    valid_metrics_str = (
        f"\n{'='*35}\n"
        f"Validation Results (Epoch {epoch}) - {valid_count} samples\n"
        f"{'='*35}\n"
        f"DSM Regression Metrics:\n"
        f"    MSE:    {valid_avg_mse:.6f}\n"
        f"    MAE:    {valid_avg_mae:.6f}\n"
        f"    RMSE:   {valid_avg_rmse:.6f}\n"
        f"    Delta1: {valid_avg_delta1:.6f}\n"
        f"    Delta2: {valid_avg_delta2:.6f}\n"
        f"    Delta3: {valid_avg_delta3:.6f}\n"
    )
    
    # Add segmentation metrics
    valid_metrics_str += f"\nSegmentation Metrics:\n    Mean IoU: {valid_miou:.6f}"
    
    # Add per-class IoU metrics if they exist (and if not too many classes)
    if valid_iou_per_class is not None and len(valid_iou_per_class) > 0:
        if len(valid_iou_per_class) <= 4:  # Only show per-class for small number of classes
            valid_metrics_str += "\n    Per-class IoU: "
            for i, iou_val in enumerate(valid_iou_per_class):
                valid_metrics_str += f"Class {i}: {iou_val:.4f}  "
        else:
            valid_metrics_str += f"\n    [{len(valid_iou_per_class)} classes - omitted for brevity]"
    
    valid_metrics_str += f"\n\nComputed in {valid_time:.6f} sec"
    logger.info(valid_metrics_str)

    # Update metrics with all available values
    if plot_train_error:
        # Update regression metrics
        train_metrics['mse'].append(train_avg_mse)
        train_metrics['mae'].append(train_avg_mae)
        train_metrics['rmse'].append(train_avg_rmse)
        train_metrics['delta1'].append(train_avg_delta1)
        train_metrics['delta2'].append(train_avg_delta2)
        train_metrics['delta3'].append(train_avg_delta3)
        
        # Update segmentation metrics
        train_metrics['miou'] = train_metrics.get('miou', []) + [train_miou]
        
        # Add per-class IoU metrics if they exist
        if train_iou_per_class is not None:
            for i, iou_val in enumerate(train_iou_per_class):
                train_metrics[f'iou_class{i}'] = train_metrics.get(f'iou_class{i}', []) + [iou_val]
    
    # Update validation metrics
    valid_metrics['mse'].append(valid_avg_mse)
    valid_metrics['mae'].append(valid_avg_mae)
    valid_metrics['rmse'].append(valid_avg_rmse)
    valid_metrics['delta1'].append(valid_avg_delta1)
    valid_metrics['delta2'].append(valid_avg_delta2)
    valid_metrics['delta3'].append(valid_avg_delta3)
    
    # Update segmentation metrics
    valid_metrics['miou'] = valid_metrics.get('miou', []) + [valid_miou]
    
    # Add per-class IoU metrics if they exist
    if valid_iou_per_class is not None:
        for i, iou_val in enumerate(valid_iou_per_class):
            valid_metrics[f'iou_class{i}'] = valid_metrics.get(f'iou_class{i}', []) + [iou_val]

    # Update plots
    fig, axes, lines = plot_train_valid_metrics(
        epoch=epoch,
        train_metrics=train_metrics if plot_train_error else None,
        valid_metrics=valid_metrics,
        fig=fig,
        axes=axes,
        plot_train=plot_train_error,
        model_type=model_type
    )
    
    # Always redraw plot and save figure
    fig.canvas.draw()
    plt.savefig(f'{dataset_name}_{model_type.lower()}_train_valid_metrics.png')
    logger.info(f"Metrics plot updated at {dataset_name}_{model_type.lower()}_train_valid_metrics.png")
        
    return valid_avg_rmse, train_metrics, valid_metrics


def compute_mtl_metrics(
    iter,                  # Current iteration number
    mtl_train_iters,      # Total number of training iterations
    mtl_log_freq,         # Frequency of logging
    last_epoch_saved,     # Last epoch when checkpoint was saved
    epoch,                # Current epoch number 
    mtl_min_loss,         # Minimum MTL loss recorded
    error_total,          # Total error accumulator
    error_L1,             # DSM error accumulator
    error_L2,             # Semantic error accumulator
    error_L3,             # Normal error accumulator
    error_L4,             # Edge error accumulator
    error_rmse,           # RMSE error accumulator
    logger,               # Logger instance
    train_valid_flag,     # Whether validation is enabled
    mtl,                  # MTL model instance
    predCheckPointPath    # Path to save model checkpoints
):
    """
    Compute and log MTL training metrics at specified intervals.
    
    Args:
        iter (int): Current iteration number
        mtl_train_iters (int): Total number of training iterations
        mtl_log_freq (int): Frequency of logging
        last_epoch_saved (int): Last epoch when checkpoint was saved
        epoch (int): Current epoch number
        mtl_min_loss (float): Minimum MTL loss recorded
        error_total (float): Total error accumulator
        error_L1 (float): DSM error accumulator
        error_L2 (float): Semantic error accumulator
        error_L3 (float): Normal error accumulator
        error_L4 (float): Edge error accumulator
        error_rmse (float): RMSE error accumulator
        logger: Logger instance
        train_valid_flag (bool): Whether validation is enabled
        mtl: MTL model instance
        predCheckPointPath (str): Path to save model checkpoints
        
    Returns:
        tuple: Updated values:
            mtl_min_loss (float): Updated minimum MTL loss
            last_epoch_saved (int): Updated last saved epoch
            error_total (float): Reset total error
            error_L1 (float): Reset DSM error
            error_L2 (float): Reset semantic error
            error_L3 (float): Reset normal error
            error_L4 (float): Reset edge error
            error_rmse (float): Reset RMSE error
    """
    # Check if this is the last validation round and adjust the validation frequency accordingly
    mtl_log_freq_ = get_logging_frequency(iter, mtl_train_iters, mtl_log_freq)

    logger.info(f"iteration no: {iter}")
    logger.info(f"###  MTL TRAIN AVERAGE LOSS VALUES FOR {mtl_log_freq_} BATCHES  ###")
    logger.info('average total wloss : {:.6f}'.format(error_total / mtl_log_freq_))
    logger.info('average DSM loss    : {:.6f}'.format(error_L1 / mtl_log_freq_) + 
                ('\tcurrent DSM minimum loss: {:.6f}'.format(mtl_min_loss) if not train_valid_flag else ''))
    logger.info('average DSM RMSE    : {:.6f}'.format(error_rmse / mtl_log_freq_))

    if (sem_flag):
        logger.info('average SEM loss    : {:.6f}'.format(error_L2 / mtl_log_freq_))
    if (norm_flag):
        logger.info('average NORM loss   : {:.6f}'.format(error_L3 / mtl_log_freq_))
    if (edge_flag):
        logger.info('average Edge loss   : {:.6f}'.format(error_L4 / mtl_log_freq_))

    # Save the model weights as checkpoints if the validation is disabled
    if not train_valid_flag:
        logger.info(f"Last MTL train checkpoint saved at epoch #{last_epoch_saved}")
        # Update the minimum loss threshold and save the checkpoint
        if error_L1 / mtl_log_freq_ < mtl_min_loss:
            mtl.save_weights(predCheckPointPath)
            mtl_min_loss = error_L1 / mtl_log_freq_
            last_epoch_saved = epoch
            logger.info(f'MTL train checkpoint saved at epoch {epoch}!')

    # Reset error metrics for further updating options
    error_total = 0.0
    error_L1 = 0.0
    error_L2 = 0.0
    error_L3 = 0.0
    error_L4 = 0.0
    error_rmse = 0.0  # Reset RMSE tracker
    
    return (
        mtl_min_loss,
        last_epoch_saved,
        error_total,
        error_L1,
        error_L2, 
        error_L3,
        error_L4,
        error_rmse
    )


def compute_dae_metrics(
    iter,                  # Current iteration number
    dae_train_iters,      # Total number of training iterations 
    dae_log_freq,         # Frequency of logging
    last_epoch_saved,     # Last epoch when checkpoint was saved
    epoch,                # Current epoch number
    dae_min_loss,         # Minimum DAE loss recorded
    error_dae,            # DAE error accumulator
    error_rmse,           # RMSE error accumulator
    logger,               # Logger instance
    train_valid_flag,     # Whether validation is enabled
    dae,                  # DAE model instance
    corrCheckPointPath    # Path to save model checkpoints
):
    """
    Compute and log DAE training metrics at specified intervals.
    
    Args:
        iter (int): Current iteration number
        dae_train_iters (int): Total number of training iterations
        dae_log_freq (int): Frequency of logging
        last_epoch_saved (int): Last epoch when checkpoint was saved
        epoch (int): Current epoch number
        dae_min_loss (float): Minimum DAE loss recorded
        error_dae (float): DAE error accumulator
        error_rmse (float): RMSE error accumulator 
        logger: Logger instance
        train_valid_flag (bool): Whether validation is enabled
        dae: DAE model instance
        corrCheckPointPath (str): Path to save model checkpoints
        
    Returns:
        tuple: Updated values:
            dae_min_loss (float): Updated minimum DAE loss
            last_epoch_saved (int): Updated last saved epoch
            error_dae (float): Reset DAE error
            error_rmse (float): Reset RMSE error
    """
    # Get the frequency of validation
    dae_log_freq_ = get_logging_frequency(iter, dae_train_iters, dae_log_freq)

    logger.info(f"iteration no: {iter}")
    logger.info(f"###  DAE TRAIN AVERAGE LOSS VALUES FOR {dae_log_freq_} BATCHES  ###")
    logger.info('average DAE loss : {:.6f}'.format(error_dae / dae_log_freq_) + 
                ('\tcurrent DAE minimum loss: {:.6f}'.format(dae_min_loss) if not train_valid_flag else ''))
    logger.info('average DAE RMSE : {:.6f}'.format(error_rmse / dae_log_freq_))

    # Save the model weights as checkpoints if the validation is disabled
    if not train_valid_flag:
        logger.info(f"Last DAE train checkpoint saved at epoch #{last_epoch_saved}")
        # Update the minimum loss threshold and save the checkpoint
        if error_dae / dae_log_freq_ < dae_min_loss:
            dae.save_weights(corrCheckPointPath)
            dae_min_loss = error_dae / dae_log_freq_
            last_epoch_saved = epoch
            logger.info(f'DAE train checkpoint saved at epoch {epoch}')

    # Reset the error metric for further updating options
    error_dae = 0.0
    error_rmse = 0.0
    
    return dae_min_loss, last_epoch_saved, error_dae, error_rmse


def get_metric_value(metrics_dict, metric_name):
    """Get the specified metric value from metrics dictionary"""
    metric_map = {
        'rmse': metrics_dict['rmse'][-1] if metrics_dict['rmse'] else 0.0,
        'mse': metrics_dict['mse'][-1] if metrics_dict['mse'] else 0.0,
        'mae': metrics_dict['mae'][-1] if metrics_dict['mae'] else 0.0,
        'delta1': metrics_dict['delta1'][-1] if metrics_dict['delta1'] else 0.0,
        'delta2': metrics_dict['delta2'][-1] if metrics_dict['delta2'] else 0.0,
        'delta3': metrics_dict['delta3'][-1] if metrics_dict['delta3'] else 0.0
    }
    return metric_map.get(metric_name, 0.0)


def should_compute_metrics(current_iter: int, total_iters: int, log_freq: int) -> bool:
    """Check if metrics should be computed at current iteration"""
    is_logging_step = (current_iter % log_freq == 0)
    is_final_step = (current_iter == total_iters)
    return is_logging_step or is_final_step


def get_logging_frequency(current_iter: int, total_iters: int, log_freq: int) -> int:
    """Get logging frequency, adjusted for final iteration if needed"""
    if current_iter == total_iters and current_iter % log_freq != 0:
        return total_iters % log_freq
    return log_freq


def plot_train_valid_metrics(epoch, train_metrics, valid_metrics, fig=None, axes=None, plot_train=False, model_type='MTL'):
    """Plot training and validation metrics during training."""
    # Determine which metrics to plot and their types
    metrics_to_plot = []
    for metric in plot_metrics:
        if metric in metric_names or metric in segmentation_scalar_metrics:
            metrics_to_plot.append((metric, 'scalar'))
        elif metric in segmentation_class_metrics:
            metrics_to_plot.append((metric, 'class'))
    
    if fig is None:
        # Create figure with subplots for each metric
        num_metrics = len(metrics_to_plot)
        fig, axes = plt.subplots(1, num_metrics, figsize=(6*num_metrics, 5))
        if num_metrics == 1:
            axes = [axes]
            
        # Setup plots for each metric
        lines = []
        for i, (metric, metric_type) in enumerate(metrics_to_plot):
            ax = axes[i]
            
            if metric_type == 'scalar':
                # Setup for scalar metrics (RMSE, Delta, mIoU, OA, FWIoU)
                if plot_train:
                    line1, = ax.plot([], [], label=f'Training {metric.upper()}', color='blue')
                    lines.append(line1)
                line2, = ax.plot([], [], label=f'Validation {metric.upper()}', color='orange')
                lines.append(line2)
                
                # Configure axes
                ax.set_xlabel('Epoch')
                ax.set_ylabel(metric.upper())
                ax.set_title(f'{metric.upper()} Progress')
                ax.legend()
                ax.grid(True)
                
                # Set initial axis limits
                ax.set_xlim(0, 10)
                if metric.startswith(('delta', 'iou', 'oa', 'fwiou')):  # Metrics with [0,1] range
                    ax.set_ylim(0, 1)
                else:
                    ax.set_ylim(0, 1)  # Will be adjusted later
            
            else:  # metric_type == 'class'
                # Setup for per-class metrics (IoU, Precision, Recall, F1)
                num_classes = len(semantic_label_map)
                colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
                
                # Create lines for each class
                for class_idx in range(num_classes):
                    if plot_train:
                        line1, = ax.plot([], [], 
                                       label=f'Train Class {class_idx}', 
                                       color=colors[class_idx], 
                                       linestyle='-')
                        lines.append(line1)
                    
                    line2, = ax.plot([], [], 
                                   label=f'Valid Class {class_idx}', 
                                   color=colors[class_idx], 
                                   linestyle='--')
                    lines.append(line2)
                
                # Configure axes
                ax.set_xlabel('Epoch')
                ax.set_ylabel(f'Class {metric.upper()}')
                ax.set_title(f'Per-class {metric.upper()} Progress')
                ax.legend(loc='upper right', fontsize='small')
                ax.grid(True)
                
                # Set initial axis limits
                ax.set_xlim(0, 10)
                ax.set_ylim(0, 1)  # IoU, Precision, Recall, F1 are all in [0, 1]
        
        # Add title
        section_title = 'Multi-task Learning (MTL)' if model_type == 'MTL' else 'Denoising Auto-Encoder (DAE)'
        plt.subplots_adjust(top=0.85)
        fig.suptitle(f'{dataset_name} Dataset - {section_title} Training Progress', 
                    fontsize=12, 
                    y=0.98)
                    
        plt.tight_layout()
        return fig, axes, lines

    # Update existing plots
    lines = []
    for i, (metric, metric_type) in enumerate(metrics_to_plot):
        ax = axes[i]
        # Clear any existing lines
        ax.clear()
        
        if metric_type == 'scalar':
            # Plot scalar metrics
            # Plot training data if available
            if plot_train and train_metrics and metric in train_metrics:
                x_data = list(range(1, epoch + 1))
                ax.plot(x_data, train_metrics[metric], label=f'Training {metric.upper()}', color='blue')
                
            # Plot validation data
            if valid_metrics and metric in valid_metrics:
                x_data = list(range(1, epoch + 1))
                ax.plot(x_data, valid_metrics[metric], label=f'Validation {metric.upper()}', color='orange')
                
            # Reconfigure axes
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.upper())
            ax.set_title(f'{metric.upper()} Progress')
            ax.legend()
            ax.grid(True)
            
            # Adjust axis limits
            ax.set_xlim(1, max(epoch + 1, 10))
            if metric.startswith(('delta', 'iou', 'oa', 'fwiou')):
                ax.set_ylim(0, 1)
            else:
                # For error metrics, adjust y-axis based on data
                metric_data = valid_metrics.get(metric, [])
                if plot_train and train_metrics and metric in train_metrics:
                    metric_data = train_metrics[metric] + metric_data
                if metric_data:  # Check if there's data to plot
                    min_val = min([x for x in metric_data if x is not None])
                    max_val = max([x for x in metric_data if x is not None])
                    padding = 0.1 * (max_val - min_val)
                    ax.set_ylim(max(0, min_val - padding), max_val + padding)
        
        else:  # metric_type == 'class'
            # Plot per-class metrics
            num_classes = len(semantic_label_map)
            colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
            
            for class_idx in range(num_classes):
                class_metric_key = f"{metric}_class{class_idx}"
                
                if plot_train and train_metrics and class_metric_key in train_metrics:
                    x_data = list(range(1, epoch + 1))
                    ax.plot(x_data, train_metrics[class_metric_key], 
                           label=f'Train Class {class_idx}', 
                           color=colors[class_idx], 
                           linestyle='-')
                
                if valid_metrics and class_metric_key in valid_metrics:
                    x_data = list(range(1, epoch + 1))
                    ax.plot(x_data, valid_metrics[class_metric_key], 
                           label=f'Valid Class {class_idx}', 
                           color=colors[class_idx], 
                           linestyle='--')
            
            # Reconfigure axes
            ax.set_xlabel('Epoch')
            ax.set_ylabel(f'Class {metric.upper()}')
            ax.set_title(f'Per-class {metric.upper()} Progress')
            ax.legend(loc='upper right', fontsize='small')
            ax.grid(True)
            
            # Adjust axis limits
            ax.set_xlim(1, max(epoch + 1, 10))
            ax.set_ylim(0, 1)  # Keep [0, 1] range for classification metrics

    # Maintain layout and draw
    fig.tight_layout()
    plt.draw()
    
    return fig, axes, lines








