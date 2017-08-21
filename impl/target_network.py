import gym
import universe
import tensorflow as tf

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

# cnn part codes are deried from cifar10.py from tensorflow tutorials
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/target_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 10, #10000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

### These should go to config
# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 20.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.
NUM_CLASSES = # number of actions

FITLER_HEIGHT = 5
FITLER_WIDTH = 5
IN_CHANNELS = 1
NUM_FILTERS =  10

def _get_var(name, shape, initializer):
	with tf.device('/cpu:0'):
		dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    	var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  	return var

def _weights_var(name, shape, stddev):
	dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
	var = _get_var(name, shape, initializer = tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
	# ignor weight decay for a while, will add it back

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


class target_network(object):
	def __init__(self):
		self.env = gym.make('AirRaid-v0') #atari (250 * 160 * 3), 256 color palettes
		self.history = [] #a list of processed 84*84 pixel images
		#self.model = 

	#for each frame in the atari game
	def preprocessing(self, frame):
		
		print "dimensions of action space", env.action_space
		print "dimensions of observation space:", env.observation_space

		gray_scale = rgb2gray_func(frame)

	def inference(self, images):
		# convolution layer 1
		with tf.variable_scope('conv1') as scope:
			#shape: filter_height, filter_width, in_channels, out_channels]
			filters = _weights_var("weights", [8, 8, IN_CHANNELS, 16], stddev=5e-2)
			conv = tf.nn.conv2d(images, filters, [4, 4, 4, 4], padding = "SAME")
			biases = _get_var("biases", [16], tf.constant_initializer(0.0))
			pre_activation = tf.nn.bias_add(conv, biases)
			conv1 = tf.nn.relu(pre_activation, name = scope.name)
			_activation_summary(conv1)

		# convolution layer 2
		with tf.variable_scope('conv2') as scope:
			filters = _weights_var("weights", [4, 4, 16, 32], stddev=5e-2)
			conv = tf.nn.conv2d(images, filters, [2, 2, 2, 2], padding = "SAME")
			biases = _get_var("biases", [32], tf.constant_initializer(0.0))
			pre_activation = tf.nn.bias_add(conv, biases)
			conv2 = tf.nn.relu(pre_activation, name = scope.name)
			 _activation_summary(conv2)

		# hidden layer
		  # local3
  		with tf.variable_scope('local3') as scope:
    		# Move everything into depth so we can perform a single matrix multiply.
    		reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
    		dim = reshape.get_shape()[1].value
    		weights = _weights_var('weights', shape=[dim, 256],
                                          stddev=0.04, wd=0.004)
    		biases = _get_var('biases', [256], tf.constant_initializer(0.1))
    		local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    		_activation_summary(local3)

		# softmax P(a|s,theta) following the full-connected layer
		  with tf.variable_scope('local4') as scope:
    		weights = _weights_var('weights', shape=[256, NUM_CLASSES],
                                          stddev=0.04, wd=0.004)
    		biases = _get_var('biases', [NUM_CLASSES], tf.constant_initializer(0.1))
    		local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
    		_activation_summary(local4)

		# valuation function V(a, s) following the ?



	# convert color palette to gray scale
	def rgb2gray_func(frame):
		# Y' =  0.2989 * R + 0.5870 * G + 0.1140 * B'
		return 0.2989 * frame[:,:,0] + 0.5870 * frame[:,:,1] + 0.1140 * frame[:,:,2]

"""

The Atari experiments used the same input preprocessing as (Mnih et al., 2015) 
and an action repeat of 4. The agents used the network architecture from (Mnih et al., 2013).

The network used 
	1. a convolutional layer with 16 filters of size 8  8 with stride 4,
	2. followed by a convolutional layer with with 32 filters of size 4 * 4 with stride 2, 
	3. followed by a fully connected layer with 256 hidden units. 
	(All three hidden layers were followed by a rectifier nonlinearity). 

The value-based methods had a single linear output unit for each action representing the action-value. 

"""