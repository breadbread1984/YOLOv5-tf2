#!/usr/bin/python3

import tensorflow as tf;
from models import YOLOv5_large;

def main():

  yolov5l = YOLOv5_large((608, 608, 3), 80);
  optimizer = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.ExponentialDecay(1e-4, decay_steps = 110000, decay_rate = 0.99));
  checkpoint = tf.train.Checkpoint(model = yolov5l, optimizer = optimizer);
  checkpoint.restore(tf.train.latest_checkpoint('checkpoints'));
  yolov5l.save('yolov5l.h5');
  yolov5l.save_weights('yolov5l_weights.h5');

if __name__ == "__main__":

  assert tf.executing_eagerly();
  main();

