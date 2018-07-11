
import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf 
import numpy as np
# Set seed
seed = 42
np.random.seed(seed)
tf.set_random_seed(seed)

x = tf.placeholder("float",[None,784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x,W) + b)

y_ = tf.placeholder("float", [None,10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
# minimize cross entropy

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
# initialization
init=tf.global_variables_initializer()

# Start Graphs
sess=tf.Session()
sess.run(init)

for step in range(1000):
  	batch_xs, batch_ys = mnist.train.next_batch(100)
  	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys}) # Sampling
  	if step%20==1:
  		print(step,sess.run([cross_entropy], feed_dict={x: batch_xs, y_: batch_ys}))



correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))