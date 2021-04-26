#!/usr/bin/python3

from os import environ, listdir;
from os.path import join, exists;
import numpy as np;
import cv2;
import tensorflow as tf;
from models import YOLOv5_large, Loss;
from Predictor import Predictor;
from create_dataset import parse_function_generator, parse_function;

environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1';
#environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices --tf_xla_auto_jit=2'
#os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3';
#os.environ['CUDA_VISIBLE_DEVICES'] = '';
batch_size = 4; # images of different sizes can't be stack into a batch
trainset_size = 118287;
testset_size = 5000;

def main():

  yolov5l = YOLOv5_large((608,608,3,), 80);
  loss1 = Loss((608,608,3,), 0, 80);
  loss2 = Loss((608,608,3,), 1, 80);
  loss3 = Loss((608,608,3,), 2, 80);
  if exists('./checkpoints/ckpt'): yolov5l = tf.keras.models.load_model('./checkpoints/ckpt', custom_objects = {'tf': tf});
  optimizer = tf.keras.optimizers.Adam(1e-4);
  yolov5l.compile(optimizer = optimizer, loss = {'output1': lambda labels, outputs: loss1([outputs, labels]),
                                                                     'output2': lambda labels, outputs: loss2([outputs, labels]),
                                                                     'output3': lambda labels, outputs: loss3([outputs, labels])});

  # load downloaded dataset
  trainset_filenames = [join('trainset', filename) for filename in listdir('trainset')];
  testset_filenames = [join('testset', filename) for filename in listdir('testset')];
  trainset = tf.data.TFRecordDataset(trainset_filenames).map(parse_function_generator(80)).repeat(-1).shuffle(batch_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE);
  testset = tf.data.TFRecordDataset(testset_filenames).map(parse_function_generator(80)).repeat(-1).shuffle(batch_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE);
  callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir = './checkpoints'),
    tf.keras.callbacks.ModelCheckpoint(filepath = './checkpoints/ckpt', save_freq = 10000),
  ];
  yolov5l.fit(trainset, steps_per_epoch = ceil(trainset_size / batch_size), epochs = 100, validation_data = testset, validation_steps = ceil(testset_size / batch_size), callbacks = callbacks);
  yolov5l.save('yolov5l.h5');

if __name__ == "__main__":
  
  main();

