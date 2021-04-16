#!/usr/bin/python3

import tensorflow as tf;
import tensorflow_addons as tfa;

def ConvBlockMish(input_shape, filters, kernel_size, strides = (1, 1), activate = True, bn = True):

  inputs = tf.keras.Input(input_shape);
  results = inputs;
  if strides == (2, 2):
    padding = 'valid'
    pad_h, pad_w = (kernel_size[0] - 2) // 2 + 1, (kernel_size[1] - 2) // 2 + 1;
    results = tf.keras.layers.ZeroPadding2D(padding = ((pad_h, pad_h),(pad_w, pad_w)))(results);
  else:
    padding = 'same';
  # NOTE: use bias when batchnorm is not used
  results = tf.keras.layers.Conv2D(filters, kernel_size = kernel_size, strides = strides, padding = padding, use_bias = not bn)(results);
  if bn == True: results = tf.keras.layers.BatchNormalization()(results);
  # NOTE: mish
  if activate == True: results = tf.keras.layers.Lambda(lambda x: tfa.activations.mish(x))(results);
  return tf.keras.Model(inputs = inputs, outputs = results);

def ConvBlockLeakyReLU(input_shape, filters, kernel_size, strides = (1, 1), activate = True, bn = True):
  
  inputs = tf.keras.Input(input_shape);
  results = inputs;
  if strides == (2, 2):
    padding = 'valid'
    pad_h, pad_w = (kernel_size[0] - 2) // 2 + 1, (kernel_size[1] - 2) // 2 + 1;
    results = tf.keras.layers.ZeroPadding2D(padding = ((pad_h, pad_h),(pad_w, pad_w)))(results);
  else:
    padding = 'same';
  # NOTE: use bias when batchnorm is not used
  results = tf.keras.layers.Conv2D(filters, kernel_size = kernel_size, strides = strides, padding = padding, use_bias = not bn)(inputs);
  if bn == True: results = tf.keras.layers.BatchNormalization()(results);
  # NOTE: leak relu
  if activate == True: results = tf.keras.layers.LeakyReLU(alpha = 0.1)(results);
  return tf.keras.Model(inputs = inputs, outputs = results);

def ResBlock(input_shape, filters, blocks, output_filters, output_kernel, downsample = False):

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
  results = ConvBlockMish(results.shape[1:], output_filters, output_kernel, strides = (2, 2) if downsample else (1, 1))(results);
  return tf.keras.Model(inputs = inputs, outputs = results);

def Body(input_shape):

  inputs = tf.keras.Input(shape = input_shape);
  cb = ConvBlockMish(inputs.shape[1:], 32, (3, 3))(inputs); # cb.shape = (batch, h, w, 32)
  cb = ConvBlockMish(cb.shape[1:], 64, (1, 1), strides = (2, 2))(cb); # cb.shape = (batch, h / 2, w / 2, 64)
  rb1 = ResBlock(cb.shape[1:], filters = 64, blocks = 1, output_filters = 128, output_kernel = (3, 3), downsample = True)(cb); # rb1.shape = (batch, h / 4, w / 4, 128)
  rb2 = ResBlock(rb1.shape[1:], filters = 128, blocks = 3, output_filters = 256, output_kernel = (3, 3), downsample = True)(rb1); # rb2.shape = (batch, h / 8, w / 8, 256)
  rb3 = ResBlock(rb2.shape[1:], filters = 256, blocks = 3, output_filters = 512, output_kernel = (3, 3), downsample = True)(rb2); # rb3.shape = (batch, h / 16, w / 16, 512)
  return tf.keras.Model(inputs = inputs, outputs = (rb1, rb2, rb3));

def YOLOv5(input_shape = (608, 608, 3), class_num = 80, anchor_num = 3):

  inputs = tf.keras.Input(shape = input_shape);
  small, middle, large = Body(inputs.shape[1:])(inputs);
  pool1 = tf.keras.layers.MaxPool2D(pool_size = (13, 13), strides = (1, 1), padding = 'same')(large);
  pool2 = tf.keras.layers.MaxPool2D(pool_size = (9, 9), strides = (1, 1), padding = 'same')(large);
  pool3 = tf.keras.layers.MaxPool2D(pool_size = (5, 5), strides = (1, 1), padding = 'same')(large);
  results = tf.keras.layers.Concatenate(axis = -1)([pool1, pool2, pool3, large]);
  results = ConvBlockMish(results.shape[1:], 512, (1, 1), (2, 2))(results);
  large_feature = ResBlock(results.shape[1:], filters = 512, blocks = 1, output_filters = 256, output_kernel = (1, 1))(results);
  results = tf.keras.layers.UpSampling2D(2, interpolation = 'nearest')(large_feature);
  raw_middle_feature = ConvBlockLeakyReLU(results.shape[1:], 256, (1, 1))(results);
  results = tf.keras.layers.Concatenate(axis = -1)([results, raw_middle_feature]);
  results = ConvBlockLeakyReLU(results.shape[1:], 256, (1, 1))(results);
  middle_feature = ResBlock(results.shape[1:], filters = 256, blocks = 1, output_filters = 128, output_kernel = (1, 1))(results);
  results = tf.keras.layers.UpSampling2D(2, interpolation = 'nearest')(middle_feature);
  raw_small_feature = ConvBlockLeakyReLU(results.shape[1:], 128, (1, 1))(results);
  results = tf.keras.layers.Concatenate(axis = -1)([results, raw_small_feature]);
  small_feature = ConvBlockLeakyReLU(results.shape[1:], 128, (1, 1))(results);
  results = ResBlock(small_feature.shape[1:], filters = 128, blocks = 1, output_filters = 128, output_kernel = (1, 1), downsample = True)(small_feature);
  # 1) output predicts of all scales
  # output predicts for small scale targets
  small_predicts = ConvBlockLeakyReLU(results.shape[1:], 3 * (class_num + 5), (1, 1), activate = False, bn = False)(results);
  small_predicts = tf.keras.layers.Reshape((input_shape[0] // 8, input_shape[1] // 8, anchor_num, 5 + class_num), name = 'output3')(small_predicts);
  # output predicts for middle scale targets
  results = ConvBlockLeakyReLU(small_feature.shape[1:], 128, (3, 3), strides = (2, 2))(small_feature);
  middle_results = ConvBlockLeakyReLU(middle.shape[1:], 256, (1, 1))(middle);
  results = tf.keras.layers.Concatenate(axis = -1)([results, middle_results]);
  middle_feature = ConvBlockLeakyReLU(results.shape[1:], 256, (1, 1))(results);
  results = ResBlock(middle_feature.shape[1:], filters = 256, blocks = 1, output_filters = 256, output_kernel = (1, 1), downsample = True)(middle_feature);
  middle_predicts = ConvBlockLeakyReLU(results.shape[1:], 3 * (class_num + 5), (1, 1), activate = False, bn = False)(results);
  middle_predicts = tf.keras.layers.Reshape((input_shape[0] // 16, input_shape[1] // 16, anchor_num, 5 + class_num), name = 'output2')(middle_predicts);
  # output predicts for large scale targets
  results = ConvBlockLeakyReLU(middle_feature.shape[1:], 256, (3, 3), strides = (2, 2))(middle_feature);
  large_results = ConvBlockLeakyReLU(large.shape[1:], 512, (1, 1))(large);
  results = tf.keras.layers.Concatenate(axis = -1)([results, large_results]);
  large_feature = ConvBlockLeakyReLU(results.shape[1:], 512, (1, 1))(results);
  results = ResBlock(large_feature.shape[1:], filters = 512, blocks = 1, output_filters = 512, output_kernel = (1, 1), downsample = True)(large_feature);
  large_predicts = ConvBlockLeakyReLU(results.shape[1:], 3 * (class_num + 5), (1, 1), activate = False, bn = False)(results);
  large_predicts = tf.keras.layers.Reshape((input_shape[0] // 32, input_shape[1] // 32, anchor_num, 5 + class_num), name = 'output1')(large_predicts);
  return tf.keras.Model(inputs = inputs, outputs = (large_predicts, middle_predicts, small_predicts));

if __name__ == "__main__":
  yolov5 = YOLOv5();
  yolov5.save('yolov5.h5');
