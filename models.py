#!/usr/bin/python3

import tensorflow as tf;

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
  if activate == True: results = tf.keras.layers.Lambda(lambda x: x * tf.math.tanh(tf.math.softplus(x)))(results);
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

def OutputParser(input_shape, img_shape, anchors):

  # feats.shape = batch x grid h x grid w x anchor_num x (1(delta x) + 1(delta y) + 1(width scale) + 1(height scale) + 1(object mask) + class_num(class probability))
  # NOTE: box center absolute x = delta x + prior box upper left x, box center absolute y = delta y + prior box upper left y
  # NOTE: width scale = box width / anchor width, height scale = box height / anchor height
  tf.debugging.Assert(tf.math.logical_and(tf.equal(tf.shape(input_shape)[0],4), tf.equal(input_shape[2], 3)), [input_shape]);
  tf.debugging.Assert(tf.equal(tf.shape(img_shape)[0],3), [img_shape]);
  # anchors.shape = (3,2)
  tf.debugging.Assert(tf.math.logical_and(tf.equal(tf.shape(anchors)[0], 3), tf.equal(tf.shape(anchors)[1], 2)), [anchors]);
  feats = tf.keras.Input(input_shape);
  # [x,y] = meshgrid(x,y) get the upper left positions of prior boxes
  # grid.shape = (grid h, grid w, 1, 2)
  grid_y = tf.keras.layers.Lambda(lambda x: tf.tile(tf.reshape(tf.range(tf.cast(tf.shape(x)[1], dtype = tf.float32), dtype = tf.float32), (-1, 1, 1, 1)), (1, tf.shape(x)[2], 1, 1)))(feats);
  grid_x = tf.keras.layers.Lambda(lambda x: tf.tile(tf.reshape(tf.range(tf.cast(tf.shape(x)[2], dtype = tf.float32), dtype = tf.float32), (1, -1, 1, 1)), (tf.shape(x)[1], 1, 1, 1)))(feats);
  grid = tf.keras.layers.Concatenate(axis = -1)([grid_x, grid_y]);
  # box center proportional position = (delta x, delta y) + (priorbox upper left x,priorbox upper left y) / (feature map.width, feature map.height)
  # box_xy.shape = (batch, grid h, grid w, anchor_num, 2)
  box_xy = tf.keras.layers.Lambda(lambda x: (tf.math.sigmoid(x[0][...,0:2]) + x[1]) / tf.cast([tf.shape(x[1])[1], tf.shape(x[1])[0]], dtype = tf.float32))([feats, grid]);
  # box proportional size = (width scale, height scale) * (anchor width, anchor height) / (image.width, image.height)
  # box_wh.shape = (batch, grid h, grid w, anchor_num, 2)
  box_wh = tf.keras.layers.Lambda(lambda x, y, z: tf.math.exp(x[...,2:4]) * y / tf.cast([z[1], z[0]], dtype = tf.float32), arguments = {'y': anchors, 'z': img_shape})(feats);
  # confidence of being an object
  box_confidence = tf.keras.layers.Lambda(lambda x: tf.math.sigmoid(x[..., 4]))(feats);
  # class confidence
  box_class_probs = tf.keras.layers.Lambda(lambda x: tf.math.sigmoid(x[..., 5:]))(feats);
  return tf.keras.Model(inputs = feats, outputs = (box_xy, box_wh, box_confidence, box_class_probs));

def giou_loss_fn(y_true, y_pred, mode = "giou"):
    """Implements the GIoU loss function.
    GIoU loss was first introduced in the
    [Generalized Intersection over Union:
    A Metric and A Loss for Bounding Box Regression]
    (https://giou.stanford.edu/GIoU.pdf).
    GIoU is an enhancement for models which use IoU in object detection.
    Args:
        y_true: true targets tensor. The coordinates of the each bounding
            box in boxes are encoded as [y_min, x_min, y_max, x_max].
        y_pred: predictions tensor. The coordinates of the each bounding
            box in boxes are encoded as [y_min, x_min, y_max, x_max].
        mode: one of ['giou', 'iou'], decided to calculate GIoU or IoU loss.
    Returns:
        GIoU loss float `Tensor`.
    """
    if mode not in ["giou", "iou"]:
        raise ValueError("Value of mode should be 'iou' or 'giou'")
    y_pred = tf.convert_to_tensor(y_pred)
    if not y_pred.dtype.is_floating:
        y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.cast(y_true, y_pred.dtype)
    giou = tf.squeeze(_calculate_giou(y_pred, y_true, mode))

    return 1 - giou

def _calculate_giou(b1, b2, mode = "giou"):
    """
    Args:
        b1: bounding box. The coordinates of the each bounding box in boxes are
            encoded as [y_min, x_min, y_max, x_max].
        b2: the other bounding box. The coordinates of the each bounding box
            in boxes are encoded as [y_min, x_min, y_max, x_max].
        mode: one of ['giou', 'iou'], decided to calculate GIoU or IoU loss.
    Returns:
        GIoU loss float `Tensor`.
    """
    zero = tf.convert_to_tensor(0.0, b1.dtype)
    b1_ymin, b1_xmin, b1_ymax, b1_xmax = tf.unstack(b1, 4, axis=-1)
    b2_ymin, b2_xmin, b2_ymax, b2_xmax = tf.unstack(b2, 4, axis=-1)
    b1_width = tf.maximum(zero, b1_xmax - b1_xmin)
    b1_height = tf.maximum(zero, b1_ymax - b1_ymin)
    b2_width = tf.maximum(zero, b2_xmax - b2_xmin)
    b2_height = tf.maximum(zero, b2_ymax - b2_ymin)
    b1_area = b1_width * b1_height
    b2_area = b2_width * b2_height

    intersect_ymin = tf.maximum(b1_ymin, b2_ymin)
    intersect_xmin = tf.maximum(b1_xmin, b2_xmin)
    intersect_ymax = tf.minimum(b1_ymax, b2_ymax)
    intersect_xmax = tf.minimum(b1_xmax, b2_xmax)
    intersect_width = tf.maximum(zero, intersect_xmax - intersect_xmin)
    intersect_height = tf.maximum(zero, intersect_ymax - intersect_ymin)
    intersect_area = intersect_width * intersect_height

    union_area = b1_area + b2_area - intersect_area
    iou = tf.math.divide_no_nan(intersect_area, union_area)
    if mode == "iou":
        return iou

    enclose_ymin = tf.minimum(b1_ymin, b2_ymin)
    enclose_xmin = tf.minimum(b1_xmin, b2_xmin)
    enclose_ymax = tf.maximum(b1_ymax, b2_ymax)
    enclose_xmax = tf.maximum(b1_xmax, b2_xmax)
    enclose_width = tf.maximum(zero, enclose_xmax - enclose_xmin)
    enclose_height = tf.maximum(zero, enclose_ymax - enclose_ymin)
    enclose_area = enclose_width * enclose_height
    giou = iou - tf.math.divide_no_nan((enclose_area - union_area), enclose_area)
    return giou

def Loss(img_shape, layer, class_num = 80, ignore_thresh = 0.5):

  # outputs is a tuple
  # outputs.shape[layer] = batch x h x w x anchor_num x (1(delta x) + 1(delta y) + 1(width scale) + 1(height scale) + 1(object mask) + class_num(class probability))
  # labels is a tuple
  # labels.shape[layer] = batch x h x w x anchor_num x (1(proportional x) + 1 (proportional y) + 1(proportional width) + 1(proportional height) + 1(object mask) + class_num(class probability))
  # NOTE: the info carried by the output and the label is different.
  assert layer in [0, 1, 2];
  tf.debugging.Assert(tf.equal(tf.shape(img_shape)[0], 3), [img_shape]);
  anchors = {2: [[10, 13], [16, 30], [33, 23]], 1: [[30, 61], [62, 45], [59, 119]], 0: [[116, 90], [156, 198], [373, 326]]};
  input_shapes = [
    (img_shape[0] // 32, img_shape[1] // 32, 3, 5 + class_num),
    (img_shape[0] // 16, img_shape[1] // 16, 3, 5 + class_num),
    (img_shape[0] // 8, img_shape[1] // 8, 3, 5 + class_num)
  ];
  input_shape_of_this_layer = input_shapes[layer];
  anchors_of_this_layer = anchors[layer];
  input_of_this_layer = tf.keras.Input(input_shape_of_this_layer);
  label_of_this_layer = tf.keras.Input(input_shape_of_this_layer);
  # 1) preprocess prediction
  pred_xy, pred_wh, pred_box_confidence, pred_class = OutputParser(input_shape_of_this_layer, img_shape, anchors_of_this_layer)(input_of_this_layer);
  pred_half_wh = tf.keras.layers.Lambda(lambda x: x / 2)(pred_wh);
  pred_upperleft = tf.keras.layers.Lambda(lambda x: x[0] - x[1])([pred_xy, pred_half_wh]); # pred_upperleft.shape = (batch, grid h, grid w, anchor_num, 2) in sequence of (xmin, ymin)
  pred_bottomright = tf.keras.layers.Lambda(lambda x: x[0] + x[1])([pred_xy, pred_half_wh]); # pred_bottomright.shape = (batch, grid h, grid w, anchor_num, 2) in sequence of (xmax, ymax)
  pred_bbox = tf.keras.layers.Lambda(lambda x: tf.concat([tf.reverse(x[0], axis = [-1]), tf.reverse(x[1], axis = [-1])], axis = -1))([pred_upperleft, pred_bottomright]); # pred_bbox.shape = (batch, grid h, grid w, anchor_num, 4) in sequence of (ymin, xmin, ymax, xmax)
  # 2) preprocess label
  true_position = tf.keras.layers.Lambda(lambda x: x[..., 0:4])(label_of_this_layer); # true_box.shape = (batch, grid h, grid w, anchor_num, 4) in sequence of (center x, center y, w, h)
  true_xy = tf.keras.layers.Lambda(lambda x: x[..., 0:2])(true_position); # true_xy.shape = (batch, grid h, grid w, anchor_num, 2)
  true_wh = tf.keras.layers.Lambda(lambda x: x[..., 2:4])(true_position); # true_wh.shape = (batch, grid h, grid w, anchor_num, 2)
  true_half_wh = tf.keras.layers.Lambda(lambda x: x[..., 2:4] / 2)(true_position); # true_half_wh.shape = (batch, grid h, grid w, anchor_num, 2)
  true_upperleft = tf.keras.layers.Lambda(lambda x: x[0] - x[1])([true_xy, true_half_wh]); # true_upperleft.shape = (batch, grid h, grid w, anchor_num, 2) in sequence of (xmin, ymin)
  true_bottomright = tf.keras.layers.Lambda(lambda x: x[0] + x[1])([true_xy, true_half_wh]); # true_bottomright.shape = (batch, grid h, grid w, anchor_num, 2) in sequence of (xmax, ymax)
  object_mask = tf.keras.layers.Lambda(lambda x: x[..., 4])(label_of_this_layer); # object_mask.shape = (batch, grid h, grid w, anchor_num)
  object_mask_bool = tf.keras.layers.Lambda(lambda x: tf.cast(x, dtype = tf.bool))(object_mask); # object_mask_bool.shape = (batch, grid h, grid w, anchor_num)
  true_bbox = tf.keras.layers.Lambda(lambda x: tf.concat([tf.reverse(x[0], axis = [-1]), tf.reverse(x[1], axis = [-1])], axis = -1))([true_upperleft, true_bottomright]); # true_bbox.shape = (batch, grid h, grid w, anchor_num, 4) in sequence of (ymin, xmin, ymax, xmax)
  true_class = tf.keras.layers.Lambda(lambda x: x[..., 5:])(label_of_this_layer); # true_class.shape = (batch, grid h, grid w, anchor_num, class_num)
  loss_scale = tf.keras.layers.Lambda(lambda x: 2 - x[..., 2] * x[..., 3])(true_position); # loss_scale.shape = (batch, grid h, grid w, anchor_num) punish harshly for smaller targets
  # 3) ignore mask
  def body(x):
    true_bbox, object_mask_bool, pred_bbox = x;
    # true_bbox.shape = (grid h, grid w, anchor_num, 4)
    # object_mask_bool.shape = (grid h, grid w, anchor_num)
    # pred_bbox.shape = (grid h, grid w, anchor_num, 4)
    true_bbox_list = tf.boolean_mask(true_bbox, object_mask_bool); # true_bbox_list.shape = (obj_num, 4)
    shape = tf.shape(pred_bbox)[:-1];
    pred_bbox_list = tf.reshape(pred_bbox, (-1, 4));
    bbox1_hw = true_bbox_list[..., 2:4] - true_bbox_list[..., 0:2]; # bbox1_hw.shape = (obj_num1, 2)
    bbox1_area = bbox1_hw[..., 0] * bbox1_hw[..., 1]; # bbox1_area.shape = (obj_num1)
    bbox2_hw = pred_bbox_list[..., 2:4] - pred_bbox_list[..., 0:2]; # bbox2_hw.shape = (obj_num2, 2)
    bbox2_area = bbox2_hw[..., 0] * bbox2_hw[..., 1]; # bbox2_area.shape = (obj_num2)
    intersect_min = tf.maximum(tf.expand_dims(true_bbox_list[..., 0:2], axis = 1), tf.expand_dims(pred_bbox_list[..., 0:2], axis = 0)); # intersect_min.shape = (obj_num1, obj_num2, 2)
    intersect_max = tf.minimum(tf.expand_dims(true_bbox_list[..., 2:4], axis = 1), tf.expand_dims(pred_bbox_list[..., 2:4], axis = 0)); # intersect_max.shape = (obj_num1, obj_num2, 2)
    intersect_hw = tf.maximum(intersect_max - intersect_min, 0); # intersect_hw.shape = (obj_num1, obj_num2, 2)
    intersect_area = intersect_hw[..., 0] * intersect_hw[..., 1]; # intersect_area.shape = (obj_num1, obj_num2)
    iou = intersect_area / tf.maximum(tf.expand_dims(bbox1_area, axis = 1) + tf.expand_dims(bbox2_area, axis = 0) - intersect_area, 1e-5); # iou.shape = (obj_num1, obj_num2)
    iou = tf.reshape(iou, tf.concat([tf.shape(true_bbox_list)[0:1], tf.shape(pred_bbox)[:-1]], axis = 0)); # iou.shape = (obj_num, grid h, grid w, anchor_num)
    best_iou = tf.math.reduce_max(iou, axis = 0); # iou.shape = (grid h, grid w, anchor_num)
    ignore_mask = tf.where(tf.math.less(best_iou, ignore_thresh), tf.ones_like(best_iou), tf.zeros_like(best_iou)); # ignore_mask.shape = (grid h, grid w, anchor_num)
    return ignore_mask;
  ignore_mask = tf.keras.layers.Lambda(lambda x, s: tf.map_fn(body, x, dtype = tf.float32), arguments = {'s': input_shape_of_this_layer})([true_bbox, object_mask_bool, pred_bbox]); # ignore_mask.shape = (batch, grid h, grid w, anchor_num)
  # 4) position loss
  # NOTE: only punish foreground area
  # NOTE: punish smaller foreground targets more harshly
  giou_loss = tf.keras.layers.Lambda(lambda x: x[0] * x[1] * giou_loss_fn(x[2], x[3], mode = 'iou'))([object_mask, loss_scale, true_bbox, pred_bbox]); # giou_loss.shape = (batch, grid h, grid w, anchor_num)
  # 5) confidence loss
  # NOTE: punish foreground area which is miss classified
  # NOTE: and punish background area which is far from foreground area and miss classified
  confidence_loss = tf.keras.layers.Lambda(lambda x: tf.math.pow(x[0] - x[1], 2) * (
      x[0] * tf.keras.losses.BinaryCrossentropy(from_logits = False, reduction = tf.keras.losses.Reduction.NONE)(
        tf.expand_dims(x[0], axis = -1), tf.expand_dims(x[1], axis = -1)
      ) + \
      (1. - x[0]) * x[2] * tf.keras.losses.BinaryCrossentropy(from_logits = False, reduction = tf.keras.losses.Reduction.NONE)(
        tf.expand_dims(x[0], axis = -1), tf.expand_dims(x[1], axis = -1)
      ))
    )([object_mask, pred_box_confidence, ignore_mask]); # confidence_loss.shape = (batch, grid h, grid w, anchor_num)
  # 6) class loss
  # NOTE: only punish foreground area
  class_loss = tf.keras.layers.Lambda(lambda x: 
      x[0] * tf.math.reduce_mean(tf.keras.losses.BinaryCrossentropy(from_logits = False, reduction = tf.keras.losses.Reduction.NONE)(
        tf.expand_dims(x[1], axis = -1), 
        tf.expand_dims(x[2], axis = -1)
      ), axis = -1)
    )([object_mask, true_class, pred_class]); # class_loss.shape = (batch, grid h, grid w, anchor_num)
  # 7) total
  loss = tf.keras.layers.Lambda(lambda x: tf.math.reduce_mean(tf.math.reduce_sum(tf.math.add_n(x), axis = [1,2,3]), axis = [0]))([giou_loss, confidence_loss, class_loss]); # loss.shape = ()
  return tf.keras.Model(inputs = (input_of_this_layer, label_of_this_layer), outputs = loss);

if __name__ == "__main__":
  yolov5s = YOLOv5_small();
  yolov5s.save_weights('yolov5s', save_format = 'tf');
  yolov5m = YOLOv5_middle();
  yolov5m.save_weights('yolov5m', save_format = 'tf');
  yolov5l = YOLOv5_large();
  yolov5l.save_weights('yolov5l', save_format = 'tf');
  yolov5x = YOLOv5_extend();
  yolov5x.save_weights('yolov5x', save_format = 'tf');
