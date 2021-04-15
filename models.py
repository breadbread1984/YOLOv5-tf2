#!/usr/bin/python3

import tensorflow as tf;
import tensorflow_addons as tfa;

def ConvBlockMish(input_shape, filters, kernel_size, strides = (1, 1), bn = True):

  padding = 'valid' if strides == (2,2) else 'same';
  inputs = tf.keras.Input(input_shape);
  # NOTE: use bias when batchnorm is not used
  results = tf.keras.layers.Conv2D(filters, kernel_size = kernel_size, strides = strides, padding = padding, use_bias = not bn)(inputs);
  if bn == True:
    results = tf.keras.layers.BatchNormalization()(results);
  # NOTE: mish
  results = tf.keras.layers.Lambda(lambda x: tfa.activations.mish(x))(results);
  return tf.keras.Model(inputs = inputs, outputs = results);

def ConvBlockLeakyReLU(input_shape, filters, kernel_size, strides = (1, 1), bn = True):
  
  padding = 'valid' if strides == (2,2) else 'same';
  inputs = tf.keras.Input(input_shape);
  # NOTE: use bias when batchnorm is not used
  results = tf.keras.layers.Conv2D(filters, kernel_size = kernel_size, strides = strides, padding = padding, use_bias = not bn)(inputs);
  if bn == True:
    results = tf.keras.layers.BatchNormalization()(results);
  # NOTE: leak relu
  results = tf.keras.layers.LeakyReLU(alpha = 0.1)(results);
  return tf.keras.Model(inputs = inputs, outputs = results);

def ResBlock(input_shape, filters, blocks):

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
  return tf.keras.Model(inputs = inputs, outputs = results);

def Body(input_shape):

  inputs = tf.keras.Input(shape = input_shape);
  # TODO

def YOLOv5(input_shape = (608, 608, 3), class_num = 80, anchor_num = 3):

  inputs = tf.keras.Input(shape = input_shape);
