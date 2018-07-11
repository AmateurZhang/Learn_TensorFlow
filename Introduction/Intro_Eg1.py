import tensorflow as tf
import numpy as np 

# Set seed
seed = 42
np.random.seed(seed)
tf.set_random_seed(seed)

# Generate phony data
x_data=np.float32(np.random.rand(2,100))
# np.random.rand: return random [d0,d1] size matrix from [0,1]
y_data=np.dot([0.100,0.240],x_data)+0.300

# Build Linear model
b=tf.Variable(tf.zeros([1]))
# tf.zeros: [matrix] size zeros
W=tf.Variable(tf.random_uniform([1,2],-1.0,1.0))
y=tf.matmul(W,x_data)+b


# loss function: LSE
loss= tf.reduce_mean(tf.square(y-y_data))
optimizer=tf.train.GradientDescentOptimizer(0.1)
train=optimizer.minimize(loss)

# initialization
init=tf.global_variables_initializer()

# Start Graphs
sess=tf.Session()
sess.run(init)

for step in range(0,201):
	sess.run(train)
	if step%20==0:
		print(step,sess.run(W),sess.run(b))
