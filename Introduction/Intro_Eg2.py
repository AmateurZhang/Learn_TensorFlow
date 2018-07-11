import tensorflow as tf 
import numpy as np 

# Basic instructions
matrix_1=tf.constant([[3.,3.]])
matrix_2=tf.constant([[2.],[2.]])
product=tf.multiply(matrix_1,matrix_2)

with tf.Session() as sess:
	with tf.device("/gpu:1"):
		result=sess.run([product])
		print(result)

# Interactive session
with tf.InteractiveSession() as sess:
#sess=tf.InteractiveSession()
	x=tf.Variable([3.,2.])
	b=tf.constant([1.,5.])
	x.initializer.run()
	sub_1=tf.subtract(x,b)
	print(sub_1.eval())


