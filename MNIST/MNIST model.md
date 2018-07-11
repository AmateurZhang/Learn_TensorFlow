# MNIST model

### Import Data

```python
import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

```

### CNN

#### initialize

```python
import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf 
import numpy as np
# Set seed
seed = 42
np.random.seed(seed)
tf.set_random_seed(seed)

# Initialize the weights and bias to avoid zeros
def weight_variable(shape):
	initial=tf.truncated_normal(shape,stddev=0.1)
	# tf.truncated_normal: normal distribution with shape={shape},
	# stddev={stddev}, mean={mean}
	return tf.Variable(initial)

def bias_variabl(shape):
	initial=tf.constant(0.1,shape=shape)
	return tf.Variable(initial)
```

#### Convolution and Pooling

```python
# Convolution and Pooling
# Convolution: filter, Pooling: Sampling(Maximum or Average)
def conv2d(x,W):
	return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],
		padding='SAME')
```

##### tf.nn.conv2d

```shell
Help on function conv2d in module tensorflow.python.ops.gen_nn_ops:
conv2d(input, filter, strides, padding, use_cudnn_on_gpu=True, data_format='NHWC', dilations=[1, 1, 1, 1], name=None)
    Computes a 2-D convolution given 4-D `input` and `filter` tensors.
    Given an input tensor of shape `[batch, in_height, in_width, in_channels]
    and a filter / kernel tensor of shape
    `[filter_height, filter_width, in_channels, out_channels]`, this op
    performs the following:    
    1. Flattens the filter to a 2-D matrix with shape
       `[filter_height * filter_width * in_channels, output_channels]`.
    2. Extracts image patches from the input tensor to form a *virtual*
       tensor of shape `[batch, out_height, out_width,
       filter_height * filter_width * in_channels]`.
    3. For each patch, right-multiplies the filter matrix and the image patch
       vector.
    In detail, with the default NHWC format,
        output[b, i, j, k] =
            sum_{di, dj, q} input[b, strides[1] * i + di, strides[2] * j + dj, q] 
                            filter[di, dj, q, k]
    Must have `strides[0] = strides[3] = 1`.  For the most common case of the same
    horizontal and vertices strides, `strides = [1, stride, stride, 1]`.
    Args:
      input: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`.
        A 4-D tensor. The dimension order is interpreted according to the value    
        of `data_format`, see below for details.
      filter: A `Tensor`. Must have the same type as `input`.
        A 4-D tensor of shape   
        `[filter_height, filter_width, in_channels, out_channels]`
      strides: A list of `ints`.
        1-D tensor of length 4.  The stride of the sliding window for each   
        dimension of `input`. The dimension order is determined by the value of  
        `data_format`, see below for details.
      padding: A `string` from: `"SAME", "VALID"`.
        The type of padding algorithm to use.
      use_cudnn_on_gpu: An optional `bool`. Defaults to `True`.
      data_format: An optional `string` from: `"NHWC", "NCHW"`. Defaults to `"NHWC"`.
        Specify the data format of the input and output data. With the   
        default format "NHWC", the data is stored in the order of: 
            [batch, height, width, channels].
        Alternatively, the format could be "NCHW", the data storage order of:
            [batch, channels, height, width].
      dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1]`.
        1-D tensor of length 4.  The dilation factor for each dimension of
        `input`. If set to k > 1, there will be k-1 skipped cells between each
        filter element on that dimension. The dimension order is determined by the
        value of `data_format`, see above for details. Dilations in the batch and
        depth dimensions must be 1.
      name: A name for the operation (optional).
    Returns:
      A `Tensor`. Has the same type as `input`.

```

