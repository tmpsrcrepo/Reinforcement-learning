import inspect
import time

import numpy as np
import tensorflow as tf
from utils import *


#FLAGS = flags.FLAGS

class PolicyConfig(object):
	policy_lr = 0.01
	num_steps = 20 # max number of steps
	batch_size = 1 #20
	hidden_size = 100
	forget_bias = 0.0
	num_layers = 1
	input_size = 125 # hidden vector dim (state/observation vector dimension)
	max_grad_norm = 5 #?
	height, width, input_channels = 210, 160, 3


class LSTMPolicy():
	def __init__(self, n_actions, input_size, config = PolicyConfig()):
	#def __init__(self, n_actions, width, height, input_channels, input_size, config = PolicyConfig()):
		# input placeholder: frame
		
		self._image = tf.placeholder(tf.float32, (config.height, config.width, config.input_channels))
		hidden_size = config.hidden_size
		
		self.seqLen = tf.placeholder(tf.int64)
		# config.batch_size
		# [batchsize, sequence, input dim (feature length)]
		self._input = cnn_processing(self._image, config.height, config.width, config.input_channels, tf.get_variable_scope())
		#self._input = tf.placeholder(tf.float32, (config.batch_size, None, input_size))
		self.cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
		# initial state
		#self._initial_state = tf.Variable(self.cell.zero_state(config.batch_size, tf.float32), trainable = False)

		# Initialize state (start as zero)
		zero_state= self.cell.zero_state(config.batch_size, tf.float32)
		self._c = tf.Variable(zero_state[0], trainable = False)
		self._h = tf.Variable(zero_state[1], trainable = False)
		#self._c = tf.placeholder(tf.float32, zero_state[0].get_shape())
		#self._h = tf.placeholder(tf.float32, zero_state[0].get_shape())
		self._initial_state = tf.contrib.rnn.LSTMStateTuple(self._c, self._h)

		# get ouput (for calculating logits) & final output
		output, state =  tf.nn.dynamic_rnn(self.cell, self._input, initial_state = self._initial_state, sequence_length = self.seqLen,
			time_major = False)

		self._output = tf.reshape(output, [-1, hidden_size])
		self._final_state = state
		# action logits & value prediction
		self._action_logits = self.getActionLogits(hidden_size, n_actions)
		self._value_pred = self.getValue(hidden_size, n_actions)
		self._train_op = None # optimizer.applygradients + global steps (why it requires global steps)
		self._cost = 0
		self.tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

	def getActionLogits(self, hidden_size, n_actions):
		action_w = tf.get_variable("action_w", [hidden_size, n_actions], dtype=tf.float32,
			initializer = tf.contrib.layers.xavier_initializer())
		action_b = tf.get_variable("action_b", [n_actions], dtype=tf.float32,
			initializer = tf.contrib.layers.xavier_initializer())
		return tf.matmul(self._output, action_w) + action_b

	def getValue(self, hidden_size, n_actions):
		val_w = tf.get_variable("value_w", [hidden_size, 1], dtype=tf.float32,
			initializer = tf.contrib.layers.xavier_initializer())
		val_b = tf.get_variable("value_b", [1], dtype=tf.float32,
			initializer = tf.contrib.layers.xavier_initializer())
		return tf.matmul(self._output, val_w) + val_b

	def get_action_and_probs(self, session, input_, seqLen, c, h):
		with tf.control_dependencies([tf.assign(self._c, c), tf.assign(self._h, h)]):
			softmax = tf.nn.softmax(self._action_logits)
			best_action = tf.reduce_max(softmax, 1)
			best_prob = tf.argmax(softmax, 1)

		return session.run([best_action, best_prob, self.final_state],
			feed_dict = {self._image: input_, self.seqLen: seqLen})


	@property
	def final_state(self):
	    return self._final_state

  	@property
  	def cost(self):
  		return self._cost

	@property
	def train_op(self):
	    return self._train_op

	@property
	def initial_state(self):
		return self._initial_state

	@property
	def c(self):
		return self._c

	@property
	def h(self):
		return self._h

def test_1():
	print PolicyConfig().batch_size
	#X = np.random.randn(PolicyConfig().batch_size, 1, 10)	
	import gym
	env = gym.make("Pong-v0")
	X = env.reset()

	#tf.nn.rnn_cell.LSTMCell(num_units=64, state_is_tuple=True)
	lstmPolicy =  LSTMPolicy(6, X.shape[2])

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		#logits, action_logits = sess.run([lstmPolicy._output, lstmPolicy._action_logits],
		#	feed_dict = {lstmPolicy._input: X, lstmPolicy.seqLen: X.shape[1]})

		#softmax = tf.nn.softmax(lstmPolicy._action_logits)
		#output_softmax, actions, output, initial  = sess.run([tf.reduce_max(softmax, 1), tf.argmax(softmax, 1), lstmPolicy._output,
		#	lstmPolicy.initial_state, lstmPolicy.final_state],
		#	feed_dict = {lstmPolicy._input: X, lstmPolicy.seqLen: X.shape[1]})
		#prev_state = sess.run(lstmPolicy.initial_state)
		c_new = h_new =  None
		for i in xrange(20):
			action_probs, actions, (c_new, h_new) = lstmPolicy.get_action_and_probs(sess, X, 1, lstmPolicy.c, lstmPolicy.h)

			print lstmPolicy.get_action_and_probs(sess, X, X.shape[1], c_new, h_new)

			X = env.step(actions[0])

			#print output_softmax, actions, output, initial

test_1()
