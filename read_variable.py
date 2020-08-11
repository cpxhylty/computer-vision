import tensorflow as tf
import numpy as np

W1 = tf.Variable(np.arange(216).reshape((3, 3, 3, 8)), dtype=tf.float32, name="W1")
W2 = tf.Variable(np.arange(1152).reshape((3, 3, 8, 16)), dtype=tf.float32, name="W2")

saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, "./model.ckpt")
    print("weights:", sess.run(W1[0][0]))
    #print("biases:", sess.run(W2))