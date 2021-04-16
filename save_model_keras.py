#!/usr/bin/python3

import tensorflow as tf;
from models import YOLOv5_large;

def main():

  yolov5l = YOLOv5_large((608, 608, 3), 80);
  yolov5l.load_weights('./checkpoints/ckpt/variables/variables');
  yolov5l.save('yolov5l.h5');
  yolov5l.save_weights('yolov5l_weights.h5');

if __name__ == "__main__":

  assert tf.executing_eagerly();
  main();

