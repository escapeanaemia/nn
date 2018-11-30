# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 09:29:53 2018

@author: jungkeechul
"""

import tensorflow as tf

no_feature = 2
no_output = 1
learning_iter = 1000 
learning_rate = 0.1


train_data = [[0., 0.], [0., 1.], [1., 0.] , [1., 1.]]
train_label = [[0.], [0.], [0.], [1.]]

data = tf.placeholder(tf.float32, [None, 2])
label = tf.placeholder(tf.float32, [None, 1])

w_val = tf.Variable(tf.zeros([no_feature, no_output]))
b_val = tf.Variable(tf.zeros([no_output]))

activation_result = tf.sigmoid(tf.matmul(data, w_val) + b_val)

cost = tf.square(tf.subtract(activation_result, label))

train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

predicted_label = tf.cast(activation_result > 0.5, dtype=tf.float32)

accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted_label, label), dtype=tf.float32))
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(learning_iter):
        sess.run(train, feed_dict={data:train_data, label:train_label})
        if step % 100 == 0:
            print(sess.run(accuracy, feed_dict={data:train_data, label:train_label}))

    r, p = sess.run([activation_result, predicted_label], feed_dict={data:train_data, label:train_label})

    print('Activation_Result: ', r)
    print('Predicted_Label: ', p)