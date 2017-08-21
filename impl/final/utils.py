import numpy as np
import tensorflow as tf




class CNN_Config(object):
	BATCH_SIZE = 200

	INPUT_WIDTH = 210
	INPUT_HEIGHT = 160
	INPUT_CHANNELS = 3

	NUM_FILTERS_1 = 16 #
	KERNEL_WIDTH_1 = 4
	KERNEL_HEIGHT_1 = 4
	STRIDE_SIZE_1 = 1
	POOL_WIDTH_1 = 2
	POOL_HEIGHT_1 = 2

	NUM_FILTERS_2 = 32
	KERNEL_WIDTH_2 = 4
	KERNEL_HEIGHT_2 = 4
	STRIDE_SIZE_2 = 1
	POOL_WIDTH_2 = 2
	POOL_HEIGHT_2 = 2

	PADDING = 1

	HIDDEN_UNITS_1 = 125
	#ACTION_LAYER_UNITS = 125
	#VALUE_LAYER_UNITS = 125

# convert color palette to gray scale
def rgb2gray_func(frame):
	# Y' =  0.2989 * R + 0.5870 * G + 0.1140 * B'
	new_frame =  0.2989 * frame[:,:,0] + 0.5870 * frame[:,:,1] + 0.1140 * frame[:,:,2]
	return np.expand_dims(new_frame, axis = 0)

def epsilon_greedy(action_probs, episilon, n_actions):
	random_ = random.uniform(0, 1)
	if episilon > random_:
		return tf.constant(np.random.randint(n_actions))
	else:
		return tf.argmax(action_probs)

#mostly derive from https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/examples/tutorials/layers/cnn_mnist.py
class CNN_feature():
	# input layer dimension: [batch_size, width, height, channels]
	# conv layer 1
	#with tf.variable_scope('conv', reuse=reuse) as scope:
	def __init__(self, width, height, input_channels, config = CNN_Config()):
		self.input_ = tf.placeholder(tf.float32, (height, width, input_channels))
		image = tf.reshape(self.input_, [-1, width, height, input_channels])
		kernel = tf.get_variable('W1',[config.KERNEL_WIDTH_1, config.KERNEL_HEIGHT_1, input_channels, config.NUM_FILTERS_1],
			dtype=tf.float32, initializer = tf.contrib.layers.xavier_initializer())
		conv = tf.nn.conv2d(image, kernel,
			[1, config.STRIDE_SIZE_1, config.STRIDE_SIZE_1, 1], padding = 'SAME')
		biases = tf.get_variable('b1', [config.NUM_FILTERS_1], initializer = tf.constant_initializer(0.0), dtype=tf.float32)
		pre_activation = tf.nn.bias_add(conv, biases)
		self.conv1 = tf.nn.relu(pre_activation, name = "conv1")

		# conv layer 2
		kernel = tf.get_variable('W2',
			[config.KERNEL_WIDTH_2, config.KERNEL_HEIGHT_2, config.NUM_FILTERS_1, config.NUM_FILTERS_2], dtype=tf.float32,
			initializer = tf.contrib.layers.xavier_initializer())
		conv = tf.nn.conv2d(self.conv1, kernel,
			[1, config.STRIDE_SIZE_2, config.STRIDE_SIZE_2, 1], padding = 'SAME')
		biases = tf.get_variable('b2',
			[config.NUM_FILTERS_2], initializer = tf.constant_initializer(0.0), dtype=tf.float32)
		pre_activation = tf.nn.bias_add(conv, biases)
		self.conv2 = tf.nn.relu(pre_activation, name = "conv2")
		x = tf.nn.elu(self.conv2, "feature")
		# flatten x
		self.feature = tf.expand_dims(tf.reshape(x, [-1]), [0])

def cnn_processing(input_, width, height, input_channels, scope_name, config = CNN_Config()):
	with tf.variable_scope(scope_name):
		image = tf.reshape(input_, [-1, width, height, input_channels])
		kernel = tf.get_variable('W1',[config.KERNEL_WIDTH_1, config.KERNEL_HEIGHT_1, input_channels, config.NUM_FILTERS_1],
			dtype=tf.float32, initializer = tf.contrib.layers.xavier_initializer())
		conv = tf.nn.conv2d(image, kernel,
			[1, config.STRIDE_SIZE_1, config.STRIDE_SIZE_1, 1], padding = 'SAME')
		biases = tf.get_variable('b1', [config.NUM_FILTERS_1], initializer = tf.constant_initializer(0.0), dtype=tf.float32)
		pre_activation = tf.nn.bias_add(conv, biases)
		conv1 = tf.nn.relu(pre_activation, name = "conv1")

		# conv layer 2
		kernel = tf.get_variable('W2',
			[config.KERNEL_WIDTH_2, config.KERNEL_HEIGHT_2, config.NUM_FILTERS_1, config.NUM_FILTERS_2], dtype=tf.float32,
			initializer = tf.contrib.layers.xavier_initializer())
		conv = tf.nn.conv2d(conv1, kernel,
			[1, config.STRIDE_SIZE_2, config.STRIDE_SIZE_2, 1], padding = 'SAME')
		biases = tf.get_variable('b2',
			[config.NUM_FILTERS_2], initializer = tf.constant_initializer(0.0), dtype=tf.float32)
		pre_activation = tf.nn.bias_add(conv, biases)
		conv2 = tf.nn.relu(pre_activation, name = "conv2")
		x = tf.nn.elu(conv2)
		# flatten x
		#print x.get_shape()[1]
		#return tf.expand_dims(tf.reshape(x, [-1]), [0])		
		return tf.reshape(x, [1, 1, -1])


def test():
	import gym
	env = gym.make("Pong-v0")
	frame = env.reset()
	frame = rgb2gray_func(frame)
	height, width, depth = frame.shape
	#frames = [env.step(env.action_space.sample()) for i in xrange(20)]
	
	#cnn_feature = CNN_feature(width, height, depth)
	image = tf.placeholder(tf.float32, (height, width, depth))
	feature_ = cnn_processing(image, width, height, depth, tf.get_variable_scope())

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		
		vec = sess.run(feature_, feed_dict = {image: frame})
		#vec = sess.run(cnn_feature.feature, feed_dict = {cnn_feature.input_: frame})
		print vec.shape

