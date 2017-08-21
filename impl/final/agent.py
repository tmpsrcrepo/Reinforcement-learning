from utils import *
from LstmPolicy import *
import gym

class Value_Config(object):
	batch_size = 20
	T_max = 1000
	gamma = 0.99
	env = gym.make("Pong-v0")
	action_space = env.action_space.n
	epsilon = 0.8 # epsilon greedy
	value_lr = 0.01
	policy_lr = 0.01
	value_input_size = 125
	max_grad_norm = 40.0
	num_steps = 20
# single thread version

class ActorCriticAgent(object):
	def __init__(self, server, config):
		# local number of steps
		self.local_num_steps = config.num_steps
		# local counter starting point
		self.t = 0
		self.t_max = config.T_max
		
		# discount factor on reward
		self.gamma = config.gamma
		# factor of value loss in the total loss function
		self.c = 0.5 # (ratio)
		# epislon greedy factor
		self.episilon = config.episilon
		# local env
		self.env = config.env
		# show the interface
    	self.env.render()
    	# if the game is done
    	self.done = False
    	# number of actions
    	self.n_actions = self.env.action_space.n
    	# feature size
		self.feature_size = CNN_Config().HIDDEN_UNITS_1
		# each agent shares own session
		self.session = self.get_session()
		
		# policy model
		with tf.variable_scope("local_model"):
			self.policyModel = LSTMPolicy(PolicyConfig())
			# global counter of the policy model
			self.global_steps = tf.Variable(1, name='global_step', trainable=False, dtype=tf.int32)
		# optimizer
    	self.optimizer = tf.train.RMSPropOptimizer(0.01)

    	# placeholders for rewards, values and action probs
    	self.reward = tf.placeholder(tf.float32, (1, None))
    	self.action_prob = tf.placeholder(tf.float32, (1, None))
    	self.value = tf.placeholder(tf.float32, (1, None))
    	self.max_grad_norm = config.max_grad_norm
    	self.train_op = self.optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)


    def get_session(self):
    	session = tf.get_default_session()
    	if session:
    		return session
    	else:
    		return tf.Session()

	def play(self):
		t = self.t
		self.done = False

		observations = []
		rewards = []
		values = []
		action_probs = []
		last_env = self.env.reset()

		while not (self.done or (t - self.t) >= self.local_num_steps or self.t >= self.t_max):
			t = t + 1
			observation = cnn_processing(last_env)
			action, action_prob = self.get_action_and_probs(self.session, observation)
			last_env, reward, done, _ = env.step(action)
			value = 0 if done else self.get_value(self.session, observation)
			self.done = done
			observations.append(observation)
			rewards.append(reward)
			values.append(value)
			action_probs.append(action_prob)

		# assign global step
		tf.assign_add(global_step, self.t - t)

		# update local counter
		self.t = t

		#sv = tf.train.Supervisor(logdir=FLAGS.save_path)
    	#with sv.managed_session() as session:
		#	self.rollout(session, observations)

		self.rollout(self.session, observations)


	def rollout(self, sesion, observations, rewards, values, action_probs):
		losses = []
		R = 0

		'''
		for each timestep
		R_t = r_t + γ*r_(t+1) + · · · + γ^(T−t+1)*r_(T−1) + γ^(T−t)*V(s_t)
		A_t = R_t - V(s_t)
		loss  = -sum_(t,T)(log(action_prob(t) * A_t + c(V(s_t) - R_t)^2)
		'''

		# calculate rewards, R_t from rewards & values


		self.update_gradients(reward, value, action_probs)


		run_policy_model(session, states, policy_loss[::-1], self.server.policyModel,
			eval_op = self.server.policyModel.train_op, verbose = True)

	# suppose input is in the size as (config.batch_size, None, config.input_size)
	def get_action_and_probs(self, session, input_):
		'''
		args: session, observation vector
		outputs: action, 
		'''
    	#inputs = tf.convert_to_tensor(inputs, dtype = tf.float32)
		# Note: outputs size: `[batch_size, max_time, cell.output_size]`
		softmax = tf.nn.sofmax(self.policyModel.action_logits)
		#sess.run(epsilon_greedy(pdf, self.episilon, self.n_actions), )
    	return session.run([tf.reduce_max(softmax), tf.argmax(softmax)],
    		feed_dict = {self.input = input_, self._initial_state = model.initial_state})

    def get_value(self, session, input_):
    	return session.run(tf.reduce_sum(self.value_pred),
    		feed_dict = {self.input = input_, self._initial_state = model.initial_state})

	#def assign_lr(self, session, lr_value):
	#    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

	# update all of training variables in policy_losses
	def update_gradients(self, rewards, values, action_probs):
		policy_loss = tf.reduce_sum(tf.log(action_probs) * (rewards - values))
		value_loss = tf.reduce_sum(tf.square(rewards - values))
		
		total_loss = - (policy_loss + self.c * value_loss)
		tvars = self.policyModel.tvars
		grads, _ = tf.clip_by_global_norm(tf.gradients(total_loss, [tvars]), self.max_grad_norm)
		# will update policy gradients & value gradients with respect to the sum of loss function
		self.train_op = self.optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)

	# run lstm policy model & value model in rollout
	# inputs = observations (list of state)
	def run_policy_model(self, session, input_, losses, model, eval_op = None, verbose = False):
		with session as sess:
			sess.run(tf.global_variables_initializer())
