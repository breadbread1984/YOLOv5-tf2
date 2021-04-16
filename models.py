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
  results = tf.keras.layers.Conv2D(filters, kernel_size = kernel_size, strides = strides, padding = padding, use_bias = not bn)(results);
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
  downsampled = ConvBlockMish(results.shape[1:], output_filters, output_kernel, strides = (2, 2) if downsample else (1, 1))(results);
  return tf.keras.Model(inputs = inputs, outputs = (results, downsampled));

def Body(input_shape, init_width_size = 32, init_depth_size = 1):

  inputs = tf.keras.Input(shape = input_shape);
  cb = ConvBlockMish(inputs.shape[1:], init_width_size, (3, 3))(inputs); # cb.shape = (batch, h, w, 32)
  cb = ConvBlockMish(cb.shape[1:], 2 * init_width_size, (1, 1), strides = (2, 2))(cb); # cb.shape = (batch, h / 2, w / 2, 64)
  _, rb1 = ResBlock(cb.shape[1:], filters = 2 * init_width_size, blocks = init_depth_size, output_filters = 4 * init_width_size, output_kernel = (3, 3), downsample = True)(cb); # rb1.shape = (batch, h / 4, w / 4, 128)
  _, rb2 = ResBlock(rb1.shape[1:], filters = 4 * init_width_size, blocks = 3 * init_depth_size, output_filters = 8 * init_width_size, output_kernel = (3, 3), downsample = True)(rb1); # rb2.shape = (batch, h / 8, w / 8, 256)
  _, rb3 = ResBlock(rb2.shape[1:], filters = 8 * init_width_size, blocks = 3 * init_depth_size, output_filters = 16 * init_width_size, output_kernel = (3, 3), downsample = True)(rb2); # rb3.shape = (batch, h / 16, w / 16, 512)
  return tf.keras.Model(inputs = inputs, outputs = (rb1, rb2, rb3));

def YOLOv5(input_shape = (608, 608, 3), class_num = 80, anchor_num = 3, init_width_size = 32, init_depth_size = 1):

  inputs = tf.keras.Input(shape = input_shape); # inputs.shape = (batch, h, w, 3)
  small, middle, large = Body(inputs.shape[1:], init_width_size, init_depth_size)(inputs); # small.shape = (batch, h / 4, w / 4, 128), middle.shape = (batch, h / 8, w / 8, 256), large.shape = (batch, h / 16, w / 16, 512)
  pool1 = tf.keras.layers.MaxPool2D(pool_size = (13, 13), strides = (1, 1), padding = 'same')(large); # pool1.shape = (batch, h / 16, w / 16, 512)
  pool2 = tf.keras.layers.MaxPool2D(pool_size = (9, 9), strides = (1, 1), padding = 'same')(large); # pool2.shape = (batch, h / 16, w / 16, 512)
  pool3 = tf.keras.layers.MaxPool2D(pool_size = (5, 5), strides = (1, 1), padding = 'same')(large); # pool3.shape = (batch, h / 16, w / 16, 512)
  results = tf.keras.layers.Concatenate(axis = -1)([pool1, pool2, pool3, large]); # results.shape = (batch, h / 16, w / 16, 2048)
  results = ConvBlockMish(results.shape[1:], 16 * init_width_size, (1, 1))(results); # results.shape = (batch, h / 16, w / 16, 512)
  large_feature, results = ResBlock(results.shape[1:], filters = 16 * init_width_size, blocks = init_depth_size, output_filters = 8 * init_width_size, output_kernel = (1, 1))(results); # large_feature.shape = (batch, h / 8, w / 8, 256), results.shape = (batch, h / 16, w / 16, 256)
  results = tf.keras.layers.UpSampling2D(2, interpolation = 'nearest')(results); # results.shape = (batch, h / 8, w / 8, 256)
  raw_middle_feature = ConvBlockLeakyReLU(results.shape[1:], 8 * init_width_size, (1, 1))(middle); # raw_middle_feature.shape = (batch, h / 8, w / 8, 256)
  results = tf.keras.layers.Concatenate(axis = -1)([results, raw_middle_feature]); # results.shape = (batch, h / 8, w / 8, 512)
  results = ConvBlockLeakyReLU(results.shape[1:], 8 * init_width_size, (1, 1))(results); # results.shape = (batch, h / 8, w / 8, 256)
  middle_feature, results = ResBlock(results.shape[1:], filters = 8 * init_width_size, blocks = init_depth_size, output_filters = 4 * init_width_size, output_kernel = (1, 1))(results); # middle_feature.shape = (batch, h / 4, w / 4,128), results.shape = (batch, h / 8, w / 8, 128)
  results = tf.keras.layers.UpSampling2D(2, interpolation = 'nearest')(results); # results.shape = (batch, h / 4, w / 4, 128)
  raw_small_feature = ConvBlockLeakyReLU(results.shape[1:], 4 * init_width_size, (1, 1))(small); # raw_small_feature.shape = (batch, h / 4, w / 4, 128)
  results = tf.keras.layers.Concatenate(axis = -1)([results, raw_small_feature]); # results.shape = (batch, h / 4, w / 4, 256)
  results = ConvBlockLeakyReLU(results.shape[1:], 4 * init_width_size, (1, 1))(results); # results.shape = (batch, h / 4, w / 4, 128)
  small_feature, results = ResBlock(results.shape[1:], filters = 4 * init_width_size, blocks = init_depth_size, output_filters = 4 * init_width_size, output_kernel = (1, 1), downsample = True)(results); # small_feature.shape = (batch, h / 4, w / 4, 128), results.shape = (batch, h / 8, w / 8, 128)
  # 1) output predicts of all scales
  # output predicts for small scale targets
  small_predicts = ConvBlockLeakyReLU(results.shape[1:], 3 * (class_num + 5), (1, 1), activate = False, bn = False)(results); # small_predicts.shape = (batch, h / 8, w / 8, 3 * 85)
  small_predicts = tf.keras.layers.Reshape((input_shape[0] // 8, input_shape[1] // 8, anchor_num, 5 + class_num), name = 'output3')(small_predicts); # small_predicts.shape = (batch, h / 8, w / 8, 3, 85)
  # output predicts for middle scale targets
  results = ConvBlockLeakyReLU(small_feature.shape[1:], 4 * init_width_size, (3, 3), strides = (2, 2))(small_feature);
  middle_results = ConvBlockLeakyReLU(middle.shape[1:], 8 * init_width_size, (1, 1))(middle);
  results = tf.keras.layers.Concatenate(axis = -1)([results, middle_results]);
  results = ConvBlockLeakyReLU(results.shape[1:], 8 * init_width_size, (1, 1))(results);
  middle_feature, results = ResBlock(results.shape[1:], filters = 8 * init_width_size, blocks = init_depth_size, output_filters = 8 * init_width_size, output_kernel = (1, 1), downsample = True)(results);
  middle_predicts = ConvBlockLeakyReLU(results.shape[1:], 3 * (class_num + 5), (1, 1), activate = False, bn = False)(results);
  middle_predicts = tf.keras.layers.Reshape((input_shape[0] // 16, input_shape[1] // 16, anchor_num, 5 + class_num), name = 'output2')(middle_predicts);
  # output predicts for large scale targets
  results = ConvBlockLeakyReLU(middle_feature.shape[1:], 8 * init_width_size, (3, 3), strides = (2, 2))(middle_feature);
  large_results = ConvBlockLeakyReLU(large.shape[1:], 16 * init_width_size, (1, 1))(large);
  results = tf.keras.layers.Concatenate(axis = -1)([results, large_results]);
  results = ConvBlockLeakyReLU(results.shape[1:], 16 * init_width_size, (1, 1))(results);
  large_feature, results = ResBlock(results.shape[1:], filters = 16 * init_width_size, blocks = init_depth_size, output_filters = 16 * init_width_size, output_kernel = (1, 1), downsample = True)(results);
  large_predicts = ConvBlockLeakyReLU(results.shape[1:], 3 * (class_num + 5), (1, 1), activate = False, bn = False)(results);
  large_predicts = tf.keras.layers.Reshape((input_shape[0] // 32, input_shape[1] // 32, anchor_num, 5 + class_num), name = 'output1')(large_predicts);
  return tf.keras.Model(inputs = inputs, outputs = (large_predicts, middle_predicts, small_predicts));

def YOLOv5_small(input_shape = (608, 608, 3), class_num = 80, anchor_num = 3):
  return YOLOv5(input_shape, class_num, anchor_num, 32, 1);

def YOLOv5_middle(input_shape = (608, 608, 3), class_num = 80, anchor_num = 3):
  return YOLOv5(input_shape, class_num, anchor_num, 48, 2);

def YOLOv5_large(input_shape = (608, 608, 3), class_num = 80, anchor_num = 3):
  return YOLOv5(input_shape, class_num, anchor_num, 64, 3);

def YOLOv5_extend(input_shape = (608, 608, 3), class_num = 80, anchor_num = 3):
  return YOLOv5(input_shape, class_num, anchor_num, 80, 4);

if __name__ == "__main__":
  yolov5s = YOLOv5_small();
  yolov5s.save('yolov5s.h5');
  yolov5m = YOLOv5_middle();
  yolov5m.save('yolov5m.h5');
  yolov5l = YOLOv5_large();
  yolov5l.save('yolov5l.h5');
  yolov5x = YOLOv5_extend();
  yolov5x.save('yolov5x.h5');
