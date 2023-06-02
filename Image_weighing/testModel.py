import tensorflow as tf

sess = tf.Session()
saver = tf.train.import_meta_graph('.\checkpoint\variable-0.meta')
saver.restore(sess,'.\checkpoint\variable-0.index')

inputs = sess.graph.get_tensor_by_name('input_tensor_name:0')
outputs = sess.graph.get_tensor_by_name('output_tensor_name:0')

image = tf.read_file('.\duck2\1\10.jpg')

