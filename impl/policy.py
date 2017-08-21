
#LSTM -> sequential action-value pair (after the final hidden layer)
import tensorflow as tf
import numpy as np


#inspect.getargspec(class.__init__).args -> useful function to print out the args

def data_type():
  return tf.float16 if FLAGS.use_fp16 else tf.float32

# config object
class SmallConfig(object):
  """Small config."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 20
  hidden_size = 200
  max_epoch = 4
  max_max_epoch = 8#13
  keep_prob = 0.6 #1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000



class Policy()

# read data
raw_data = reader.ptb_raw_data(FLAGS.data_path)
train_data, valid_data, test_data, _ = raw_data

# get config values
config = SmallConfig()
evalConfig = SmallConfig()
eval_config.batch_size = 1
eval_config.num_steps = 1

#use the train data
input_ = PTBInput(config=config, data=train_data, name="TrainInput")

batch_size = input_.batch_size
num_steps = input_.num_steps
size = config.hidden_size
vocab_size = config.vocab_size

# define cells
attn_cell = tf.contrib.rnn.BasicLSTMCell(
	size, forget_bias=0.0, state_is_tuple=True,
	reuse=tf.get_variable_scope().reuse)

# add dropout layer
def attn_cell():
    return tf.contrib.rnn.DropoutWrapper(
    lstm_cell(), output_keep_prob=config.keep_prob)

cell = attn_cell()
#cell = tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(config.num_layers)], state_is_tuple=True)

# define the initial state
_initial_state = cell.zero_state(batch_size, data_type())

# create embedding


# lstm (fixed time steps)
outputs = []
state = self._initial_state
with tf.variable_scope("RNN"):
	inputs = tf.unstack(inputs, num=num_steps, axis=1)
	outputs, state = tf.contrib.rnn.static_rnn(ell, inputs, initial_state=self._initial_state)
