# MAHDI ELHOUSNI, WPI 2020
# Altered by Ahmad Naghavi, OzU 2024

import os
# Set up GPU environment variables for CUDA device management
# Ensure CUDA devices are ordered by PCI bus ID for consistent behavior across sessions
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# Specify which GPU to use; here, GPU with index 1 is selected
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Change this to the desired GPU index
# Set TensorFlow log level to a specific level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all messages, 1 = filter out INFO, 2 = filter out INFO & WARNINGS, 3 = only ERROR messages

# Define the dataset to be used for training and testing
# Options include Vaihingen, Vaihingen_crp256, DFC2018, DFC2018_crp256, DFC2019_crp256, DFC2019_crp256_bin, DFC2019_crp512, 
# and DFC2023 derivatives as follows:
# DFC2023A (Ahmad's splitting), DFC2023Asmall, DFC2023Amini, and DFC2023S (Sinan's splitting) datasets
dataset_name = 'DFC2019_crp256'  # Change this to the desired dataset name

# Shortcut path to the datasets parent folder
# Because these files may be voluminous, thus you may put them inside another folder to be 
# globally available to other projects as well. You should end the path with a '/'
shortcut_path = '../datasets/'  # Change this to the desired path

# Whether the input image tile is large, thus random patches are selected out of that, (DFC2018 and Vaihingen)
# Or the input image is like a normal patch, thus as a whole could be fed to the model, (DFC2023)
large_tile_data = ['Vaihingen', 'DFC2018']
large_tile_mode = dataset_name in large_tile_data

# Define datasets that use RGB triplets for semantic labels
rgb_label_datasets = ['Vaihingen', 'Vaihingen_crp256']  # Add other RGB-triplet datasets here
uses_rgb_labels = dataset_name in rgb_label_datasets

# Define a combined dictionary with (cropSize, batchSize) tuples for each dataset
dataset_configs = {
    'Vaihingen': (320, 4),
    'DFC2018': (320, 4),
    'Vaihingen_crp256': (256, 10),
    'DFC2018_crp256': (256, 10),
    'DFC2019_crp256': (256, 10),
    'DFC2019_crp512': (512, 2),
    'DFC2023': (512, 2),
}

# Get cropSize and batchSize based on dataset name, with fallback logic
# First try exact match, then prefix match, then default to (256, 2)
if dataset_name in dataset_configs:
    cropSize, batch_size = dataset_configs[dataset_name]
else:
    # Find by prefix match if exact match not found
    matching_datasets = [d for d in dataset_configs.keys() if dataset_name.startswith(d)]
    if matching_datasets:
        # Sort by length descending to get the most specific match
        best_match = sorted(matching_datasets, key=len, reverse=True)[0]
        cropSize, batch_size = dataset_configs[best_match]
    else:
        # Default values if no match found
        cropSize, batch_size = 256, 2

# Define the flag for synthetic aperture radar (SAR) channel for the input tensor.
# This could be the case for DFC2023 in which the input RGB and the 1-channel SAR images are fused together 
# to provide the model with more precise information.
# However, in such a case, as the DenseNet architecture cannot capture more than 3 channels for 
# its input, thus we should use a convolution layer prior to that to convert the 4-channel input 
# to a 3-channel one.
datasets_with_sar = ['DFC2023']  # List of datasets that support SAR mode
sar_path_indicator = dataset_name.startswith(tuple(datasets_with_sar))
sar_mode = False

# Normalization flag for input RGB, DSM, etc
normalize_flag = False

# Parameters for the Multitask Learning (MTL) component
mtl_lr_decay = False  # Flag to enable/disable learning rate decay
mtl_lr = 0.0002  # Initial learning rate for the MTL network
mtl_batchSize = batch_size  # Batch size for training the MTL network, now dynamic based on dataset
mtl_numEpochs = 1000  # Number of epochs for training the MTL network

# Total number of training samples available for MTL generated out of data augmentation technique for large tiles, 
# o.w. for input data as patches, the true number of training samples will be used accordingly
mtl_training_samples = 10000
# Calculate the total number of iterations required for training based on batch size and samples count
mtl_train_iters = int(mtl_training_samples / mtl_batchSize)
mtl_log_freq = int(mtl_train_iters / 5)  # Frequency at which evaluation metrics are calculated during training
mtl_min_loss = float('inf')  # Minimum DSM loss threshold to save the MTL network weights as checkpoints

# Parameters for the Denoising AutoEncoder (DAE) component defined as the same way for MTL
dae_lr_decay = False
dae_lr = 0.0002
dae_batchSize = batch_size  # Batch size for training the DAE network, now dynamic based on dataset
dae_numEpochs = 1000

# Total number of training samples available for DAE generated out of data augmentation technique for large tiles, 
# o.w. for input data as patches, the true number of training samples will be used accordingly
dae_training_samples = 10000
dae_train_iters = int(dae_training_samples / dae_batchSize)
dae_log_freq = int(dae_train_iters / 5)
dae_min_loss = float('inf')  # Minimum loss (DSM noise) threshold to save the DAE network weights as checkpoints

# MTL saved weights preloading mode. If True, then all MTL model will be initialized with saved weights before training
mtl_preload = False
# MTL backbone frozen mode. If True, then the MTL backbone weights will not get updated during training to save time
mtl_bb_freeze = False

# DAE saved weights preloading mode. If True, then all DAE model will be initialized with saved weights before training
dae_preload = False

# Define the status and the path to save checkpoints for MTL and Unet
# Only add SAR mode indicator for DFC2023 datasets
sar_indicator = ('+sar' if sar_mode else '-sar') if sar_path_indicator else '.'
predCheckPointPath = f'./checkpoints/{dataset_name}/{sar_indicator}/mtl'  # MTL checkpoints path
corrCheckPointPath = f'./checkpoints/{dataset_name}/{sar_indicator}/refinement'  # DAE checkpoints path

# Initialize the epoch counter for the last saved weights when validation is disabled
last_epoch_saved = None 

# Set flag to either calculate the train/valid error after every epoch or just ignore it
# If ignored, then the train and valid sets will be merged to form one unique train set
# !!! If set to True, then be careful about the 'correction' flag as it will affect the train/valid error computations
train_valid_flag = True

# Early stopping configuration only if train_valid_flag is set to True
early_stop_flag = True  # Enable/disable early stopping 
early_stop_patience = 10  # Number of epochs to wait for improvement before stopping
early_stop_delta = 1e-2  # Minimum change in monitored value to qualify as an improvement

# Evaluation metric configuration
# Define available metrics first, since other configs depend on it
metric_names = ['mse', 'mae', 'rmse', 'delta1', 'delta2', 'delta3']

# Categorize metrics into error metrics (lower is better) and accuracy metrics (higher is better)
height_error_metrics = ['mse', 'mae', 'rmse']
height_accuracy_metrics = ['delta1', 'delta2', 'delta3']

# Segmentation metrics separated by type
segmentation_class_metrics = ['iou', 'precision', 'recall', 'f1_score']  # Per-class metrics
segmentation_scalar_metrics = ['miou', 'oa', 'fwiou']  # Overall metrics
segmentation_accuracy_metrics = segmentation_class_metrics + segmentation_scalar_metrics

eval_metric = 'rmse'  # Options: must be one of metric_names
# Automatically determine if lower is better based on metric type
eval_metric_lower_better = eval_metric in height_error_metrics

# Initialize early stopping variables based on metric configuration
best_metric = float('inf') if eval_metric_lower_better else float('-inf')
patience_counter = 0

# Plot configuration for train/valid errors
plot_train_error = False  # Flag to enable/disable calculating and plotting training errors
# Example: Add 'miou' to see segmentation metrics
plot_metrics = ['rmse', 'delta1', 'iou', 'miou']  # List of metrics to plot from metric_names + segmentation_accuracy_metrics

# Set the regression loss mode, either MSE or Huber
reg_loss = 'mse'  # 'mse' or 'huber'
huber_delta = 0.1  # Huber loss hyperparameter, delta

# Edgemap configuration
# Threshold for potential rooftops out of nDSM (pixel values are supposed to be in [0, 255])
roof_height_threshold = 50
# Canny edge detection algorithm low and high thresholds for detecting potential edges for rooftops
canny_lt, canny_ht = 50, 150

# Set flags for additive heads of MTL, viz semantic segmentation, surface normals, and edgemaps
sem_flag, norm_flag, edge_flag = True, True, False

# Set flag for MTL heads interconnection mode, either fully intertwined ('full') or just for the DSM head ('dsm')
mtl_head_mode = 'dsm'  # 'full' or 'dsm'

# Set flag for applying denoising autoencoder during testing. 
# Note: If set to True, this will affect train/valid error computations
correction = True

# Define label codes for semantic segmentation task, and
# scaling factors (weights) for different types of loss functions in MTL
# Note: You may change the scaling factors based on your discretion.
if 'Vaihingen' in dataset_name:
    label_codes = [(255, 255, 255), (0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 255, 0), (255, 0, 0)]
    w1, w2, w3, w4 = (1e-4, 1e-1, 1e-5, 0.001)  # weights for: dsm, sem, norm, edge

elif 'DFC2018' in dataset_name:
    label_codes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    w1, w2, w3, w4 = (100.0, 1.0, 10.0, 100.0)  # weights for: dsm, sem, norm, edge

elif dataset_name.startswith('DFC2019'):
    if dataset_name.endswith('bin'):
        label_codes = [0, 1]
        w1, w2, w3, w4 = (1e-2, 1e-1, 1e-5, 100.0)  # weights for: dsm, sem, norm, edge
    else:
        label_codes = [2, 5, 6, 9, 17, 65]
        w1, w2, w3, w4 = (1e-2, 1e-1, 1e-5, 100.0)  # weights for: dsm, sem, norm, edge

elif dataset_name.startswith('DFC2023'):
    label_codes = [0, 1]
    w1, w2, w3, w4 = (1e-3, 1.0, 1e-5, 1e-3)  # weights for: dsm, sem, norm, edge

# Create dictionary and indicator for semantic label codes
semantic_label_map = {k: v for k, v in enumerate(label_codes)}
sem_k = len(semantic_label_map)

# Check if the dataset is binary classification based on label codes
binary_classification_flag = len(label_codes) == 2 and set(label_codes) == {0, 1}
