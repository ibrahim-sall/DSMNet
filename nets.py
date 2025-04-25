# MAHDI ELHOUSNI, WPI 2020 
# Altered by Ahmad Naghavi, OzU 2024

import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras.utils import *
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *

from utils import *
from config import *


## Multi-task Learning (MTL) architecture
class MTL(tf.keras.Model):
    def __init__(self, net, sem_flag=True, norm_flag=True, edge_flag=True):
        super(MTL, self, ).__init__()

        # Determine the initial convolution layer to turn the RGB+SAR 4-channel input to a 3-channel tensor.
        # This is mandatory for the sake of DenseNet121 input as the number of channels shall be three.
        if sar_mode:
            # Define the input shape explicitly as (crop_size, crop_size, 4)
            self.conv_sar = Conv2D(3, 3, padding='same', input_shape=(cropSize, cropSize, 4))
            self.do_sar = Dropout(0.5)

        # Determine other model parameters
        self.encoder = net
        self.sem_flag = sem_flag
        self.norm_flag = norm_flag
        self.edge_flag = edge_flag

        self.bn0 = BatchNormalization()

        # Establish decoding layers for SEM
        if self.sem_flag:
            self.deconv1_sem = Conv2DTranspose(1024, 3, strides=2, padding='same')
            self.deconv2_sem = Conv2DTranspose(512, 3, strides=2, padding='same')
            self.deconv3_sem = Conv2DTranspose(256, 3, strides=2, padding='same')
            self.deconv4_sem = Conv2DTranspose(64, 3, strides=2, padding='same')
            self.deconv5_sem = Conv2DTranspose(32, 3, strides=2, padding='same')

            self.conv_1_sem = Conv2D(1024, 3, strides=1, padding='same')
            self.conv_2_sem = Conv2D(1024, 3, strides=1, padding='same')
            self.conv_3_sem = Conv2D(512, 3, strides=1, padding='same')
            self.conv_4_sem = Conv2D(512, 3, strides=1, padding='same')
            self.conv_5_sem = Conv2D(256, 3, strides=1, padding='same')
            self.conv_6_sem = Conv2D(256, 3, strides=1, padding='same')
            self.conv_7_sem = Conv2D(64, 3, strides=1, padding='same')
            self.conv_8_sem = Conv2D(64, 3, strides=1, padding='same')
            self.conv_9_sem = Conv2D(32, 3, strides=1, padding='same')
            self.conv_10_sem = Conv2D(32, 3, strides=1, padding='same')
            
            self.conv_sem = Conv2D(sem_k, 3, strides=1, padding='same')

            self.do1_sem = Dropout(0.5)
            self.do2_sem = Dropout(0.5)
            self.do3_sem = Dropout(0.5)
            self.do4_sem = Dropout(0.5)
            self.do5_sem = Dropout(0.5)

        # Establish decoding layers for Normals
        if self.norm_flag:
            self.deconv1_norm = Conv2DTranspose(1024, 3, strides=2, padding='same')
            self.deconv2_norm = Conv2DTranspose(512, 3, strides=2, padding='same')
            self.deconv3_norm = Conv2DTranspose(256, 3, strides=2, padding='same')
            self.deconv4_norm = Conv2DTranspose(64, 3, strides=2, padding='same')
            self.deconv5_norm = Conv2DTranspose(32, 3, strides=2, padding='same')

            self.conv_1_norm = Conv2D(1024, 3, strides=1, padding='same')
            self.conv_2_norm = Conv2D(1024, 3, strides=1, padding='same')
            self.conv_3_norm = Conv2D(512, 3, strides=1, padding='same')
            self.conv_4_norm = Conv2D(512, 3, strides=1, padding='same')
            self.conv_5_norm = Conv2D(256, 3, strides=1, padding='same')
            self.conv_6_norm = Conv2D(256, 3, strides=1, padding='same')
            self.conv_7_norm = Conv2D(64, 3, strides=1, padding='same')
            self.conv_8_norm = Conv2D(64, 3, strides=1, padding='same')
            self.conv_9_norm = Conv2D(32, 3, strides=1, padding='same')
            self.conv_10_norm = Conv2D(32, 3, strides=1, padding='same')

            self.conv_norm = Conv2D(3, 3, strides=1, padding='same')

            self.do1_norm = Dropout(0.5)
            self.do2_norm = Dropout(0.5)
            self.do3_norm = Dropout(0.5)
            self.do4_norm = Dropout(0.5)
            self.do5_norm = Dropout(0.5)

        # Establish decoding layers for EdgeMap
        if self.edge_flag:
            self.deconv1_edge = Conv2DTranspose(1024, 3, strides=2, padding='same')
            self.deconv2_edge = Conv2DTranspose(512, 3, strides=2, padding='same')
            self.deconv3_edge = Conv2DTranspose(256, 3, strides=2, padding='same')
            self.deconv4_edge = Conv2DTranspose(64, 3, strides=2, padding='same')
            self.deconv5_edge = Conv2DTranspose(32, 3, strides=2, padding='same')

            self.conv_1_edge = Conv2D(1024, 3, strides=1, padding='same')
            self.conv_2_edge = Conv2D(1024, 3, strides=1, padding='same')
            self.conv_3_edge = Conv2D(512, 3, strides=1, padding='same')
            self.conv_4_edge = Conv2D(512, 3, strides=1, padding='same')
            self.conv_5_edge = Conv2D(256, 3, strides=1, padding='same')
            self.conv_6_edge = Conv2D(256, 3, strides=1, padding='same')
            self.conv_7_edge = Conv2D(64, 3, strides=1, padding='same')
            self.conv_8_edge = Conv2D(64, 3, strides=1, padding='same')
            self.conv_9_edge = Conv2D(32, 3, strides=1, padding='same')
            self.conv_10_edge = Conv2D(32, 3, strides=1, padding='same')

            self.conv_edge = Conv2D(1, 3, strides=1, padding='same')

            self.do1_edge = Dropout(0.5)
            self.do2_edge = Dropout(0.5)
            self.do3_edge = Dropout(0.5)
            self.do4_edge = Dropout(0.5)
            self.do5_edge = Dropout(0.5)

        # Establish decoding layers for DSM
        self.deconv1_dsm = Conv2DTranspose(1024, 3, strides=2, padding='same')
        self.deconv2_dsm = Conv2DTranspose(512, 3, strides=2, padding='same')
        self.deconv3_dsm = Conv2DTranspose(256, 3, strides=2, padding='same')
        self.deconv4_dsm = Conv2DTranspose(64, 3, strides=2, padding='same')
        self.deconv5_dsm = Conv2DTranspose(32, 3, strides=2, padding='same')

        self.conv_1_dsm = Conv2D(1024, 3, strides=1, padding='same')
        self.conv_2_dsm = Conv2D(1024, 3, strides=1, padding='same')
        self.conv_3_dsm = Conv2D(512, 3, strides=1, padding='same')
        self.conv_4_dsm = Conv2D(512, 3, strides=1, padding='same')
        self.conv_5_dsm = Conv2D(256, 3, strides=1, padding='same')
        self.conv_6_dsm = Conv2D(256, 3, strides=1, padding='same')
        self.conv_7_dsm = Conv2D(64, 3, strides=1, padding='same')
        self.conv_8_dsm = Conv2D(64, 3, strides=1, padding='same')
        self.conv_9_dsm = Conv2D(32, 3, strides=1, padding='same')
        self.conv_10_dsm = Conv2D(32, 3, strides=1, padding='same')

        self.conv_dsm = Conv2D(1, 3, strides=1, padding='same')

        self.do1_dsm = Dropout(0.5)
        self.do2_dsm = Dropout(0.5)
        self.do3_dsm = Dropout(0.5)
        self.do4_dsm = Dropout(0.5)
        self.do5_dsm = Dropout(0.5)


    def call(self, x, head_mode='dsm', training=True):

        # Fuse RGB and SAR inputs if the case
        if sar_mode:
            x = self.conv_sar(x)
            x = self.do_sar(x, training=training)

        # Applying the MTL encoder on the input
        x = self.encoder(x)
        x = self.bn0(x, training=training)

        # Decoding the encoded input based on the amount of selected interconnection among MTL heads
        if head_mode == 'dsm':
            x_dsm, x_sem, x_norm, x_edge = self.call_dsm(x, training)
        if head_mode == 'full':
            x_dsm, x_sem, x_norm, x_edge = self.call_full(x, training)

        return x_dsm, x_sem, x_norm, x_edge


    def call_dsm(self, x, training=True):

        x_sem, x_norm, x_edge = [], [], []

        # SEM decoding layers
        if (self.sem_flag):
            x_sem = self.deconv1_sem(x)
            x3_sem = tf.nn.relu(x_sem)
            x_sem = self.do1_sem(x3_sem, training=training)
            x_sem = self.deconv2_sem(x_sem)
            x4_sem = tf.nn.relu(x_sem)
            x_sem = self.do2_sem(x4_sem, training=training)
            x_sem = self.deconv3_sem(x_sem)
            x5_sem = tf.nn.relu(x_sem)
            x_sem = self.do3_sem(x5_sem, training=training)
            x_sem = self.deconv4_sem(x_sem)
            x6_sem = tf.nn.relu(x_sem)
            x_sem = self.do4_sem(x6_sem, training=training)
            x_sem = self.deconv5_sem(x_sem)
            x7_sem = tf.nn.relu(x_sem)
            x_sem = self.do5_sem(x7_sem, training=training)
            x_sem = self.conv_sem(x_sem)
            x_sem = tf.nn.softmax(x_sem)

        # Normal decoding layers
        if (self.norm_flag):
            x_norm = self.deconv1_norm(x)
            x3_norm = tf.nn.relu(x_norm)
            x_norm = self.do1_norm(x3_norm, training=training)
            x_norm = self.deconv2_norm(x_norm)
            x4_norm = tf.nn.relu(x_norm)
            x_norm = self.do2_norm(x4_norm, training=training)
            x_norm = self.deconv3_norm(x_norm)
            x5_norm = tf.nn.relu(x_norm)
            x_norm = self.do3_norm(x5_norm, training=training)
            x_norm = self.deconv4_norm(x_norm)
            x6_norm = tf.nn.relu(x_norm)
            x_norm = self.do4_norm(x6_norm, training=training)
            x_norm = self.deconv5_norm(x_norm)
            x7_norm = tf.nn.relu(x_norm)
            x_norm = self.do5_norm(x7_norm, training=training)
            x_norm = self.conv_norm(x_norm)

        # EdgeMap decoding layers
        if (self.edge_flag):
            x_edge = self.deconv1_edge(x)
            x3_edge = tf.nn.relu(x_edge)
            x_edge = self.do1_edge(x3_edge, training=training)
            x_edge = self.deconv2_edge(x_edge)
            x4_edge = tf.nn.relu(x_edge)
            x_edge = self.do2_edge(x4_edge, training=training)
            x_edge = self.deconv3_edge(x_edge)
            x5_edge = tf.nn.relu(x_edge)
            x_edge = self.do3_edge(x5_edge, training=training)
            x_edge = self.deconv4_edge(x_edge)
            x6_edge = tf.nn.relu(x_edge)
            x_edge = self.do4_edge(x6_edge, training=training)
            x_edge = self.deconv5_edge(x_edge)
            x7_edge = tf.nn.relu(x_edge)
            x_edge = self.do5_edge(x7_edge, training=training)
            x_edge = self.conv_edge(x_edge)
            x_edge = tf.nn.relu(x_edge)

        ####### DSM head decoding layers intertwined with other heads #######
        #### DSM head decoding layer 1 ####
        x_dsm = self.deconv1_dsm(x)
        x_dsm = tf.nn.relu(x_dsm)
        if (self.sem_flag):
            x_dsm = concatenate([x_dsm, x3_sem], axis=3)
        if (self.norm_flag):
            x_dsm = concatenate([x_dsm, x3_norm], axis=3)
        if (self.edge_flag):
            x_dsm = concatenate([x_dsm, x3_edge], axis=3)
        x_dsm = self.conv_1_dsm(x_dsm)
        x_dsm = tf.nn.relu(x_dsm)
        x_dsm = self.conv_2_dsm(x_dsm)
        x_dsm = tf.nn.relu(x_dsm)
        x_dsm = self.do1_dsm(x_dsm, training=training)

        #### DSM head decoding layer 2 ####
        x_dsm = self.deconv2_dsm(x_dsm)
        x_dsm = tf.nn.relu(x_dsm)
        if (self.sem_flag):
            x_dsm = concatenate([x_dsm, x4_sem], axis=3)
        if (self.norm_flag):
            x_dsm = concatenate([x_dsm, x4_norm], axis=3)
        if (self.edge_flag):
            x_dsm = concatenate([x_dsm, x4_edge], axis=3)
        x_dsm = self.conv_3_dsm(x_dsm)
        x_dsm = tf.nn.relu(x_dsm)
        x_dsm = self.conv_4_dsm(x_dsm)
        x_dsm = tf.nn.relu(x_dsm)
        x_dsm = self.do2_dsm(x_dsm, training=training)

        #### DSM head decoding layer 3 ####
        x_dsm = self.deconv3_dsm(x_dsm)
        x_dsm = tf.nn.relu(x_dsm)
        if (self.sem_flag):
            x_dsm = concatenate([x_dsm, x5_sem], axis=3)
        if (self.norm_flag):
            x_dsm = concatenate([x_dsm, x5_norm], axis=3)
        if (self.edge_flag):
            x_dsm = concatenate([x_dsm, x5_edge], axis=3)
        x_dsm = self.conv_5_dsm(x_dsm)
        x_dsm = tf.nn.relu(x_dsm)
        x_dsm = self.conv_6_dsm(x_dsm)
        x_dsm = tf.nn.relu(x_dsm)
        x_dsm = self.do3_dsm(x_dsm, training=training)

        #### DSM head decoding layer 4 ####
        x_dsm = self.deconv4_dsm(x_dsm)
        x_dsm = tf.nn.relu(x_dsm)
        if (self.sem_flag):
            x_dsm = concatenate([x_dsm, x6_sem], axis=3)
        if (self.norm_flag):
            x_dsm = concatenate([x_dsm, x6_norm], axis=3)
        if (self.edge_flag):
            x_dsm = concatenate([x_dsm, x6_edge], axis=3)
        x_dsm = self.conv_7_dsm(x_dsm)
        x_dsm = tf.nn.relu(x_dsm)
        x_dsm = self.conv_8_dsm(x_dsm)
        x_dsm = tf.nn.relu(x_dsm)
        x_dsm = self.do4_dsm(x_dsm, training=training)

        #### DSM head decoding layer 5 ####
        x_dsm = self.deconv5_dsm(x_dsm)
        x_dsm = tf.nn.relu(x_dsm)
        if (self.sem_flag):
            x_dsm = concatenate([x_dsm, x7_sem], axis=3)
        if (self.norm_flag):
            x_dsm = concatenate([x_dsm, x7_norm], axis=3)
        if (self.edge_flag):
            x_dsm = concatenate([x_dsm, x7_edge], axis=3)
        x_dsm = self.conv_9_dsm(x_dsm)
        x_dsm = tf.nn.relu(x_dsm)
        x_dsm = self.conv_10_dsm(x_dsm)
        x_dsm = tf.nn.relu(x_dsm)
        x_dsm = self.do5_dsm(x_dsm, training=training)

        x_dsm = self.conv_dsm(x_dsm)

        return x_dsm, x_sem, x_norm, x_edge


    def call_full(self, x, training=True):

        x_sem, x_norm, x_edge = [], [], []

        ####### Decoding layers intertwined all with each other #######

        ####    Layer 1     ####
        # DSM head decoding layer 1
        x_dsm = self.deconv1_dsm(x)
        x_concat = tf.nn.relu(x_dsm)
        if (self.sem_flag):
            x_sem = self.deconv1_sem(x)
            x_sem = tf.nn.relu(x_sem)
            x_concat = concatenate([x_concat, x_sem], axis=3)
        if (self.norm_flag):
            x_norm = self.deconv1_norm(x)
            x_norm = tf.nn.relu(x_norm)
            x_concat = concatenate([x_concat, x_norm], axis=3)
        if (self.edge_flag):
            x_edge = self.deconv1_edge(x)
            x_edge = tf.nn.relu(x_edge)
            x_concat = concatenate([x_concat, x_edge], axis=3)
        x_dsm = self.conv_1_dsm(x_concat)
        x_dsm = tf.nn.relu(x_dsm)
        x_dsm = self.conv_2_dsm(x_dsm)
        x_dsm = tf.nn.relu(x_dsm)
        x_dsm = self.do1_dsm(x_dsm, training=training)

        # SEM head decoding layer 1
        if (self.sem_flag):
            x_sem = self.conv_1_sem(x_concat)
            x_sem = tf.nn.relu(x_sem)
            x_sem = self.conv_2_sem(x_sem)
            x_sem = tf.nn.relu(x_sem)
            x_sem = self.do1_sem(x_sem, training=training)

        # Norm head decoding layer 1
        if (self.norm_flag):
            x_norm = self.conv_1_norm(x_concat)
            x_norm = tf.nn.relu(x_norm)
            x_norm = self.conv_2_norm(x_norm)
            x_norm = tf.nn.relu(x_norm)
            x_norm = self.do1_norm(x_norm, training=training)

        # Edge head decoding layer 1
        if (self.edge_flag):
            x_edge = self.conv_1_edge(x_concat)
            x_edge = tf.nn.relu(x_edge)
            x_edge = self.conv_2_edge(x_edge)
            x_edge = tf.nn.relu(x_edge)
            x_edge = self.do1_edge(x_edge, training=training)

        ####    Layer 2     ####
        # DSM head decoding layer 2
        x_dsm = self.deconv2_dsm(x_dsm)
        x_concat = tf.nn.relu(x_dsm)
        if (self.sem_flag):
            x_sem = self.deconv2_sem(x_sem)
            x_sem = tf.nn.relu(x_sem)
            x_concat = concatenate([x_concat, x_sem], axis=3)
        if (self.norm_flag):
            x_norm = self.deconv2_norm(x_norm)
            x_norm = tf.nn.relu(x_norm)
            x_concat = concatenate([x_concat, x_norm], axis=3)
        if (self.edge_flag):
            x_edge = self.deconv2_edge(x_edge)
            x_edge = tf.nn.relu(x_edge)
            x_concat = concatenate([x_concat, x_edge], axis=3)
        x_dsm = self.conv_3_dsm(x_concat)
        x_dsm = tf.nn.relu(x_dsm)
        x_dsm = self.conv_4_dsm(x_dsm)
        x_dsm = tf.nn.relu(x_dsm)
        x_dsm = self.do2_dsm(x_dsm, training=training)

        # SEM head decoding layer 2
        if (self.sem_flag):
            x_sem = self.conv_3_sem(x_concat)
            x_sem = tf.nn.relu(x_sem)
            x_sem = self.conv_4_sem(x_sem)
            x_sem = tf.nn.relu(x_sem)
            x_sem = self.do2_sem(x_sem, training=training)

        # Norm head decoding layer 2
        if (self.norm_flag):
            x_norm = self.conv_3_norm(x_concat)
            x_norm = tf.nn.relu(x_norm)
            x_norm = self.conv_4_norm(x_norm)
            x_norm = tf.nn.relu(x_norm)
            x_norm = self.do2_norm(x_norm, training=training)

        # Edge head decoding layer 2
        if (self.edge_flag):
            x_edge = self.conv_3_edge(x_concat)
            x_edge = tf.nn.relu(x_edge)
            x_edge = self.conv_4_edge(x_edge)
            x_edge = tf.nn.relu(x_edge)
            x_edge = self.do2_edge(x_edge, training=training)

        ####    Layer 3    ####
        # DSM head decoding layer 3
        x_dsm = self.deconv3_dsm(x_dsm)
        x_concat = tf.nn.relu(x_dsm)
        if (self.sem_flag):
            x_sem = self.deconv3_sem(x_sem)
            x_sem = tf.nn.relu(x_sem)
            x_concat = concatenate([x_concat, x_sem], axis=3)
        if (self.norm_flag):
            x_norm = self.deconv3_norm(x_norm)
            x_norm = tf.nn.relu(x_norm)
            x_concat = concatenate([x_concat, x_norm], axis=3)
        if (self.edge_flag):
            x_edge = self.deconv3_edge(x_edge)
            x_edge = tf.nn.relu(x_edge)
            x_concat = concatenate([x_concat, x_edge], axis=3)
        x_dsm = self.conv_5_dsm(x_concat)
        x_dsm = tf.nn.relu(x_dsm)
        x_dsm = self.conv_6_dsm(x_dsm)
        x_dsm = tf.nn.relu(x_dsm)
        x_dsm = self.do3_dsm(x_dsm, training=training)

        # SEM head decoding layer 3
        if (self.sem_flag):
            x_sem = self.conv_5_sem(x_concat)
            x_sem = tf.nn.relu(x_sem)
            x_sem = self.conv_6_sem(x_sem)
            x_sem = tf.nn.relu(x_sem)
            x_sem = self.do3_sem(x_sem, training=training)

        # Norm head decoding layer 3
        if (self.norm_flag):
            x_norm = self.conv_5_norm(x_concat)
            x_norm = tf.nn.relu(x_norm)
            x_norm = self.conv_6_norm(x_norm)
            x_norm = tf.nn.relu(x_norm)
            x_norm = self.do3_norm(x_norm, training=training)

        # Edge head decoding layer 3
        if self.edge_flag:
            x_edge = self.conv_5_edge(x_concat)
            x_edge = tf.nn.relu(x_edge)
            x_edge = self.conv_6_edge(x_edge)
            x_edge = tf.nn.relu(x_edge)
            x_edge = self.do3_edge(x_edge, training=training)

        ####    Layer 4     ####
        # DSM head decoding layer 4
        x_dsm = self.deconv4_dsm(x_dsm)
        x_concat = tf.nn.relu(x_dsm)
        if (self.sem_flag):
            x_sem = self.deconv4_sem(x_sem)
            x_sem = tf.nn.relu(x_sem)
            x_concat = concatenate([x_concat, x_sem], axis=3)
        if (self.norm_flag):
            x_norm = self.deconv4_norm(x_norm)
            x_norm = tf.nn.relu(x_norm)
            x_concat = concatenate([x_concat, x_norm], axis=3)
        if (self.edge_flag):
            x_edge = self.deconv4_edge(x_edge)
            x_edge = tf.nn.relu(x_edge)
            x_concat = concatenate([x_concat, x_edge], axis=3)
        x_dsm = self.conv_7_dsm(x_concat)
        x_dsm = tf.nn.relu(x_dsm)
        x_dsm = self.conv_8_dsm(x_dsm)
        x_dsm = tf.nn.relu(x_dsm)
        x_dsm = self.do4_dsm(x_dsm, training=training)

        # SEM head decoding layer 4
        if (self.sem_flag):
            x_sem = self.conv_7_sem(x_concat)
            x_sem = tf.nn.relu(x_sem)
            x_sem = self.conv_8_sem(x_sem)
            x_sem = tf.nn.relu(x_sem)
            x_sem = self.do4_sem(x_sem, training=training)

        # Norm head decoding layer 4
        if (self.norm_flag):
            x_norm = self.conv_7_norm(x_concat)
            x_norm = tf.nn.relu(x_norm)
            x_norm = self.conv_8_norm(x_norm)
            x_norm = tf.nn.relu(x_norm)
            x_norm = self.do4_norm(x_norm, training=training)

        # Edge head decoding layer 4
        if (self.edge_flag):
            x_edge = self.conv_7_edge(x_concat)
            x_edge = tf.nn.relu(x_edge)
            x_edge = self.conv_8_edge(x_edge)
            x_edge = tf.nn.relu(x_edge)
            x_edge = self.do4_edge(x_edge, training=training)

        ####    Layer 5     ####
        # DSM head decoding layer 5
        x_dsm = self.deconv5_dsm(x_dsm)
        x_concat = tf.nn.relu(x_dsm)
        if (self.sem_flag):
            x_sem = self.deconv5_sem(x_sem)
            x_sem = tf.nn.relu(x_sem)
            x_concat = concatenate([x_concat, x_sem], axis=3)
        if (self.norm_flag):
            x_norm = self.deconv5_norm(x_norm)
            x_norm = tf.nn.relu(x_norm)
            x_concat = concatenate([x_concat, x_norm], axis=3)
        if (self.edge_flag):
            x_edge = self.deconv5_edge(x_edge)
            x_edge = tf.nn.relu(x_edge)
            x_concat = concatenate([x_concat, x_edge], axis=3)
        x_dsm = self.conv_9_dsm(x_concat)
        x_dsm = tf.nn.relu(x_dsm)
        x_dsm = self.conv_10_dsm(x_dsm)
        x_dsm = tf.nn.relu(x_dsm)
        x_dsm = self.do5_dsm(x_dsm, training=training)

        x_dsm = self.conv_dsm(x_dsm)

        # SEM head decoding layer 5
        if (self.sem_flag):
            x_sem = self.conv_9_sem(x_concat)
            x_sem = tf.nn.relu(x_sem)
            x_sem = self.conv_10_sem(x_sem)
            x_sem = tf.nn.relu(x_sem)
            x_sem = self.do5_sem(x_sem, training=training)

            x_sem = self.conv_sem(x_sem)
            x_sem = tf.nn.softmax(x_sem)

        # Norm head decoding layer 5
        if (self.norm_flag):
            x_norm = self.conv_9_norm(x_concat)
            x_norm = tf.nn.relu(x_norm)
            x_norm = self.conv_10_norm(x_norm)
            x_norm = tf.nn.relu(x_norm)
            x_norm = self.do5_norm(x_norm, training=training)

            x_norm = self.conv_norm(x_norm)

        # Edge head decoding layer 5
        if (self.edge_flag):
            x_edge = self.conv_9_edge(x_concat)
            x_edge = tf.nn.relu(x_edge)
            x_edge = self.conv_10_edge(x_edge)
            x_edge = tf.nn.relu(x_edge)
            x_edge = self.do5_edge(x_edge, training=training)

            x_edge = self.conv_edge(x_edge)
            x_edge = tf.nn.relu(x_edge)

        return x_dsm, x_sem, x_norm, x_edge


## Denoising Autoencoder architecture
class Autoencoder(tf.keras.Model):
    def __init__(self, random_noise_size=100):
        super(Autoencoder, self, ).__init__()

        # Encoding layers
        self.conv1_0 = Conv2D(64, 3, padding='same')
        self.conv1 = Conv2D(64, 3, padding='same')
        self.pool1 = MaxPooling2D(pool_size=(2, 2))

        self.conv2_0 = Conv2D(128, 3, padding='same')
        self.conv2 = Conv2D(128, 3, padding='same')
        self.pool2 = MaxPooling2D(pool_size=(2, 2))

        self.conv3_0 = Conv2D(256, 3, padding='same')
        self.conv3 = Conv2D(256, 3, padding='same')
        self.pool3 = MaxPooling2D(pool_size=(2, 2))

        self.conv4_0 = Conv2D(512, 3, padding='same')
        self.conv4 = Conv2D(512, 3, padding='same')
        self.pool4 = MaxPooling2D(pool_size=(2, 2))

        # Bridge between encoder and decoder
        self.conv5_0 = Conv2D(1024, 3, padding='same')
        self.conv5 = Conv2D(1024, 3, padding='same')

        # Decoding layers
        self.up6 = Conv2DTranspose(512, 2, strides=2, padding='same')
        self.conv6_0 = Conv2D(512, 3, padding='same')
        self.conv6 = Conv2D(512, 3, padding='same')

        self.up7 = Conv2DTranspose(256, 2, strides=2, padding='same')
        self.conv7_0 = Conv2D(256, 3, padding='same')
        self.conv7 = Conv2D(256, 3, padding='same')

        self.up8 = Conv2DTranspose(128, 2, strides=2, padding='same')
        self.conv8_0 = Conv2D(128, 3, padding='same')
        self.conv8 = Conv2D(128, 3, padding='same')

        self.up9 = Conv2DTranspose(64, 2, strides=2, padding='same')
        self.conv9_0 = Conv2D(64, 3, padding='same')
        self.conv9 = Conv2D(64, 3, padding='same')

        self.out = Conv2D(1, 3, padding='same')

        self.do1 = Dropout(0.5)
        self.do2 = Dropout(0.5)
        self.do3 = Dropout(0.5)
        self.do4 = Dropout(0.5)


    def call(self, x, training=True):
        # Call the encoding layers
        x = self.conv1_0(x)
        x = tf.nn.relu(x)
        x = self.conv1(x)
        x_1 = tf.nn.relu(x)
        x = self.pool1(x_1)

        x = self.conv2_0(x)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x_2 = tf.nn.relu(x)
        x = self.pool2(x_2)

        x = self.conv3_0(x)
        x = tf.nn.relu(x)
        x = self.conv3(x)
        x_3 = tf.nn.relu(x)
        x = self.pool3(x_3)

        x = self.conv4_0(x)
        x = tf.nn.relu(x)
        x = self.conv4(x)
        x_4 = tf.nn.relu(x)
        x = self.pool4(x_4)

        # Process the bridge connection
        x = self.conv5_0(x)
        x = tf.nn.relu(x)
        x = self.conv5(x)
        x = tf.nn.relu(x)

        # Call the decoding layers
        x = self.up6(x)
        x = tf.nn.relu(x)
        x = concatenate([x_4, x], axis=3)
        x = self.conv6_0(x)
        x = tf.nn.relu(x)
        x = self.conv6(x)
        x = tf.nn.relu(x)
        x = self.do1(x, training=training)

        x = self.up7(x)
        x = tf.nn.relu(x)
        x = concatenate([x_3, x], axis=3)
        x = self.conv7_0(x)
        x = tf.nn.relu(x)
        x = self.conv7(x)
        x = tf.nn.relu(x)
        x = self.do2(x, training=training)

        x = self.up8(x)
        x = tf.nn.relu(x)
        x = concatenate([x_2, x], axis=3)
        x = self.conv8_0(x)
        x = tf.nn.relu(x)
        x = self.conv8(x)
        x = tf.nn.relu(x)
        x = self.do3(x, training=training)

        x = self.up9(x)
        x = tf.nn.relu(x)
        x = concatenate([x_1, x], axis=3)
        x = self.conv9_0(x)
        x = tf.nn.relu(x)
        x = self.conv9(x)
        x = tf.nn.relu(x)
        x = self.do4(x, training=training)

        x = self.out(x)

        return x

