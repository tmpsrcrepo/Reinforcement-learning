from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

import gym
import universe
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


"""env = gym.make('flashgames.DuskDrive-v0')
env.configure(remotes=1) #automatically creates a local docker container
observation_n = env.reset()

while True:
	action_n = [[('KeyEvent', 'ArrowUp', True)] for ob in observation_n] # your agent here
	observation_n, reward_n, done_n, info = env.step(action_n)
	env.render()
"""


# goal: create a cnn network (duplicate the setting mentioned in the A3C paper p12) -> retrieve the features

"""
Input processing:
raw Atari frames, which are 210 × 160 pixel images with a 128 color palette

Preprocessing for reducing the input dimensionality:

1. converting their RGB representation to gray-scale
2. down-sampling it to a 110×84 image. 
3. cropping an 84 × 84 region of the image that roughly captures the playing area. ??? (1. how to crop? 2. only for GPU or?)
4. (apply the reprocessing to the last 4 frames of a history + stacks them to produce the input for Q-function)

"""
BATCH_SIZE = 200

INPUT_WIDTH = 210
INPUT_HEIGHT = 160
INPUT_CHANNELS = 3

NUM_FILTERS_1 = 16 #
KERNEL_WIDTH_1 = 8
KERNEL_HEIGHT_1 = 8
STRIDE_SIZE_1 = 4
POOL_WIDTH_1 = 2
POOL_HEIGHT_1 = 2

NUM_FILTERS_2 = 32
KERNEL_WIDTH_2 = 4
KERNEL_HEIGHT_2 = 4
STRIDE_SIZE_2 = 2
POOL_WIDTH_2 = 2
POOL_HEIGHT_2 = 2

PADDING = 1

HIDDEN_UNITS_1 = 256
ACTION_LAYER_UNITS = 125
VALUE_LAYER_UNITS = 125


NUM_REWARDS = 6

NUM_STATE = 6 # number actions

LSTM_SIZE = 256


""" def LSTM_model(embedding_matrix):
	lstm = tf.contrib.rnn.BasicLSTMCell(LSTM_SIZE)
	# initial state of the LSTM memory
	# [batch_size, lstm.state_size]
	state = tf.zeros([BATCH_SIZE, NUM_STATE])
	probs = []
	loss = 0.0
	for current_batch_of_words in words_in_dataset:
	    # The value of state is updated after processing each batch of words.
	    output, state = lstm(current_batch_of_words, state)

	    # The LSTM output can be used to make next word predictions
	    logits = tf.matmul(output, softmax_w) + softmax_b
	    probabilities.append(tf.nn.softmax(logits))
	    loss += loss_function(probabilities, target_words)

"""

def create_variable(name, shape, initializer = None, trainable = True):
	with tf.device('/cpu:0'):
		dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
		var = tf.get_variable(name, shape, initializer=initializer, trainable = trainable, dtype)
	return var

def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

#mostly derive from https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/examples/tutorials/layers/cnn_mnist.py
def cnn_inference(images):
	# input layer dimension: [batch_size, width, height, channels]

	# conv layer 1
	with tf.variable_scope('conv') as scope:
		
		kernel = create_variable('weights', [KERNEL_WIDTH_1, KERNEL_HEIGHT_1, INPUT_CHANNELS, NUM_FILTERS_1])
		conv = tf.nn.conv2d(images, kernel, [STRIDE_SIZE_1, STRIDE_SIZE_1, STRIDE_SIZE_1, STRIDE_SIZE_1], padding = 'SAME')
		biases = create_variable('biases', [NUM_FILTERS_1], initializer = tf.constant_initializer(0.0))
		pre_activation = tf.nn.bias_add(conv, biases)
    	conv1 = tf.nn.relu(pre_activation, name=scope.name)
    	_activation_summary(conv1)

		# conv layer 2
		kernel = create_variable('weights', [KERNEL_WIDTH_2, KERNEL_HEIGHT_2, NUM_FILTERS_1, NUM_FILTERS_2])
		conv = tf.nn.conv2d(images, kernel, [STRIDE_SIZE_2, STRIDE_SIZE_2, STRIDE_SIZE_2, STRIDE_SIZE_2], padding = 'SAME')
		biases = create_variable('biases', [NUM_FILTERS_2], initializer = tf.constant_initializer(0.0))
		pre_activation = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv2)

  	# fully connected
  	# Move everything into depth so we can perform a single matrix multiply.
  	reshape = tf.reshape(conv2, [FLAGS.batch_size, -1])
  	weights = create_variable('weights', shape=[dim, HIDDEN_UNITS_1])
  	biases = create_variable('biases', [HIDDEN_UNITS_1], initializer = tf.constant_initializer(0.1))
  	dense1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
 		_activation_summary(dense1)

  	# softmax layer -> probability for selecting action P(a_t|st; theta) -> for choosing
  	# We don't apply softmax here because
  	# tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
  	# and performs the softmax internally for efficiency.

  	# this is policy layer => can be replaced by LSTM
  	reg_constant = 0.01
   	with tf.variable_scope('softmax', regularizer=tf.contrib.layers.l2_regularizer(reg_constant)) as scope:
   		weights = create_variable('weights', [HIDDEN_UNITS_1, NUM_STATE])
   		biases = create_variable('biases', [ACTION_LAYER_UNITS], initializer = tf.constant_initializer(0.1))
   		softmax_linear = tf.add(tf.matmul(dense1, weights), biases, name=scope.name)
   		_activation_summary(softmax_linear)

   	# 1 linear output = Value function (V(s_t, theta_v))
   	with tf.variable_scope('linear_value', regularizer=tf.contrib.layers.l2_regularizer(reg_constant)) as scope:
   		weights = create_variable('weights', [HIDDEN_UNITS_1, 1])
   		biases = create_variable('biases', [1], initializer = tf.constant_initializer(0.1))
   		linear = tf.add(tf.matmul(dense1, weights), biases, name=scope.name)
   		_activation_summary(linear)

   	return softmax_linear, linear #logits, single value

# for the softmax layer fitting -> P(action|)
def get_policy_loss(logits, labels):
	labels = tf.cast(labels, tf.int64)
	#regularization
	reg_keys = tf.GraphKeys.REGULARIZATION_LOSSES
	#reg1 = tf.reduce_sum(tf.get_collection(reg_keys, "softmax")) * self.config.l2
	reg1 = tf.reduce_sum(tf.get_collection(reg_keys, "softmax")) * 0.01
    reg2 = tf.reduce_sum(tf.get_collection(reg_keys, "linear_value")) * 0.01
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
		labels=labels, logits=logits, name='cross_entropy_per_example') * (rewards - values)
	cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy') + reg1 + reg2
	tf.add_to_collection('losses', cross_entropy_mean)

	# The total loss is defined as the cross entropy loss plus all of the weight
	# decay terms (L2 loss).
	return tf.add_n(tf.get_collection('losses'), name='total_loss')


# this is for the simple version (just another layer after dense layer CNN, can be replaced by LSTM, next step)
# loss = derivative of log(P(a|s,theta)) * (R - V)
def train_policy(loss, name, global_step):
	# get track of losses
	loss_averages_op = _add_loss_summaries(loss)
	lr = 0.01 # can do learning rate update later

	# compute gradients
	with tf.control_dependencies([loss_averages_op]):
    	opt = tf.train.AdamOptimizer(lr)
    	grads = opt.compute_gradients(total_loss)

    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  	# Track the moving averages of all trainable variables.
  	variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
  	variables_averages_op = variable_averages.apply(tf.trainable_variables())

  	with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    	train_op = tf.no_op(name=name)

  	return train_op


def get_value_loss(reward):
	predicted_val = 


def __init__():


