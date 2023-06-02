import tensorflow as tf
import numpy as np
import cv2
import train
import os
import sys

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6  # 0.6 sometimes works better for folks
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))

ckeckpoint_dir = './checkpoint'

def predict_weight(image):
    sess = tf.Session()
    saver = tf.train.import_meta_graph('./checkpoint/variable-64800.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./checkpoint'))
    graph = tf.get_default_graph()
    X = graph.get_tensor_by_name("X:0")
    keep_prob = graph.get_tensor_by_name("keep_prob:0")
    result = graph.get_tensor_by_name("y:0")
    predict_data = cv2.resize(image, (80,324))
    feed_dict={X:predict_data, keep_prob:1}
    weight = str(sess.run(result, feed_dict)[0][0])
    return weight