#!/usr/bin/python3

from shutil import rmtree;
from os import mkdir;
from os.path import exists;
import tensorflow as tf;
from models import YOLOv5_large;

def main():

  yolov5l = YOLOv5_large((608, 608, 3), 80);
  yolov5l.load_weights('./checkpoints/ckpt');
  if exists('trained_model'): rmtree('trained_model');
  mkdir('trained_model');
  yolov5l.save_weights('trained_model/yolov5l', save_format = 'tf');

if __name__ == "__main__":

  assert tf.executing_eagerly();
  main();

