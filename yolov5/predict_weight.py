import tensorflow as tf
import numpy as np
import cv2
import train
import os
import sys



ckeckpoint_dir = '../Image_weighing/checkpoint'

def get_weight(image):
    sess = tf.Session()
    saver = tf.train.import_meta_graph('../Image_weighing/checkpoint/variable-2998800.meta')
    saver.restore(sess, tf.train.latest_checkpoint('../Image_weighing/checkpoint'))
    graph = tf.get_default_graph()
    X = graph.get_tensor_by_name("X:0")
    keep_prob = graph.get_tensor_by_name("keep_prob:0")
    result = graph.get_tensor_by_name("y:0")
    image = image.transpose(2,1,0)
    predict_data = image.reshape(-1,253,80)
    feed_dict={X:predict_data, keep_prob:1}
    weight = str(sess.run(result, feed_dict)[0][0])
    return weight