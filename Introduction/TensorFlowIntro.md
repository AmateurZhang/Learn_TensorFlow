# TensorFlow Introduction

## Brief Introduction

- Example 1

  The following code solves a basic problem estimating the linear parameter using tensorflow.

```python
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

```

```shell
# Result
2018-07-09 10:30:06.738569: I d:\build\tensorflow\tensorflow-r1.7\tensorflow\core\platform\cpu_feature_guard.cc:140] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
0 [[0.6376071 0.113034 ]] [0.18991522]
20 [[0.20242016 0.24315514]] [0.24729991]
40 [[0.12348525 0.24923587]] [0.2834314]
60 [[0.10607232 0.24420407]] [0.29475936]
80 [[0.10171665 0.24153213]] [0.29833746]
100 [[0.10051303 0.24051736]] [0.29947183]
120 [[0.10015812 0.24016918]] [0.29983208]
140 [[0.10004953 0.24005453]] [0.29994658]
160 [[0.10001566 0.24001749]] [0.299983]
180 [[0.10000495 0.24000557]] [0.29999462]
200 [[0.10000158 0.2400018 ]] [0.29999828]
```

Gradient Descent Optimizer is applied here.

### Use certain GPU/CPU 

```python
import tensorflow as tf 
import numpy as np 
# Basic instructions
matrix_1=tf.constant([[3.,3.]])
matrix_2=tf.constant([[2.],[2.]])
product=tf.multiply(matrix_1,matrix_2)
# GPU:1
with tf.Session() as sess:
	with tf.device("/gpu:1"):
		result=sess.run([product])
		print(result)

```

### Interactive Session

```python
# Interactive session
sess=tf.InteractiveSession()
x=tf.Variable([3.,2.])
b=tf.constant([1.,5.])
x.initializer.run()
sub_1=tf.subtract(x,b)
print(sub_1.eval())

```

