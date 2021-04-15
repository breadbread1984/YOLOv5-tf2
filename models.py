#!/usr/bin/python3

import tensorflow as tf;
import tensorflow_addons as tfa;

def ConvBlockMish(input_shape, filters, kernel_size, strides = (1, 1), activate = True, bn = True):

  padding = 'valid' if strides == (2,2) else 'same';
  inputs = tf.keras.Input(input_shape);
  # NOTE: use bias when batchnorm is not used
  results = tf.keras.layers.Conv2D(filters, kernel_size = kernel_size, strides = strides, padding = padding, use_bias = not bn)(inputs);
  if bn == True: results = tf.keras.layers.BatchNormalization()(results);
  # NOTE: mish
  if activate == True: results = tf.keras.layers.Lambda(lambda x: tfa.activations.mish(x))(results);
  return tf.keras.Model(inputs = inputs, outputs = results);

def ConvBlockLeakyReLU(input_shape, filters, kernel_size, strides = (1, 1), activate = True, bn = True):
  
  padding = 'valid' if strides == (2,2) else 'same';
  inputs = tf.keras.Input(input_shape);
  # NOTE: use bias when batchnorm is not used
  results = tf.keras.layers.Conv2D(filters, kernel_size = kernel_size, strides = strides, padding = padding, use_bias = not bn)(inputs);
  if bn == True: results = tf.keras.layers.BatchNormalization()(results);
  # NOTE: leak relu
  if activate == True: results = tf.keras.layers.LeakyReLU(alpha = 0.1)(results);
  return tf.keras.Model(inputs = inputs, outputs = results);

def ResBlock(input_shape, filters, blocks, output_filters, output_kernel):

  inputs = tf.keras.Input(shape = input_shape);
  results = ConvBlockMish(inputs.shape[1:], filters // 2, (1, 1))(inputs);
  short = ConvBlockMish(inputs.shape[1:], filters // 2, (1, 1))(inputs);
  for i in range(blocks):
    block_short = results;
    results = ConvBlockMish(results.shape[1:], filters // 2, (1, 1))(results);
    results = ConvBlockMish(results.shape[1:], filters // 2, (3, 3))(results);
    results = tf.keras.layers.Add()([results, block_short]);
  results = ConvBlockMish(results.shape[1:], filters // 2, (1, 1))(results);
  results = tf.keras.layers.Concatenate(axis = -1)([results, short]);
  results = ConvBlockMish(results.shape[1:], output_filters, output_kernel)(results);
  return tf.keras.Model(inputs = inputs, outputs = results);

def Body(input_shape):

  inputs = tf.keras.Input(shape = input_shape);
  cb = ConvBlockMish(inputs.shape[1:], 32, (3, 3))(inputs); # cb.shape = (batch, h, w, 32)
  cb = ConvBlockMish(cb.shape[1:], 64, (1, 1))(cb); # cb.shape = (batch, h, w, 64)
  rb1 = ResBlock(cb.shape[1:], filters = 64, blocks = 1, output_filters = 128, output_kernel = (3, 3))(cb); # rb1.shape = (batch, h, w, 128)
  rb2 = ResBlock(rb1.shape[1:], filters = 128, blocks = 3, output_filters = 256, output_kernel = (3, 3))(rb1); # rb2.shape = (batch, h, w, 256)
  rb3 = ResBlock(rb2.shape[1:], filters = 256, blocks = 3, output_filters = 512, output_kernel = (3, 3))(rb2); # rb3.shape = (batch, h, w, 512)
  return tf.keras.Model(inputs = inputs, outputs = (rb1, rb2, rb3));

def YOLOv5(input_shape = (608, 608, 3), class_num = 80, anchor_num = 3):

  inputs = tf.keras.Input(shape = input_shape);
  small, middle, large = Body(inputs.shape[1:])(inputs);
  pool1 = tf.keras.layers.MaxPool2D(pool_size = (13, 13), strides = (1, 1), padding = 'same')(large);
  pool2 = tf.keras.layers.MaxPool2D(pool_size = (9, 9), strides = (1, 1), padding = 'same')(large);
  pool3 = tf.keras.layers.MaxPool2D(pool_size = (5, 5), strides = (1, 1), padding = 'same')(large);
  results = tf.keras.layers.Concatenate(axis = -1)([pool1, pool2, pool3, large]);
  results = ConvBlockMish(results.shape[1:], 512, (1, 1))(results);
  results = ResBlock(results.shape[1:], filters = 512, blocks = 1, output_filters = 256, output_kernel = (1, 1))(results);
  
  
