import gym
import cPickle as pickle
from policyModels import LSTMPolicy


"""

and shared RMSProp was used for optimization. 


The model used by actor-critic agents had two set of outputs – a softmax output with one entry per action representing the
probability of selecting the action, and a single linear output representing the value function. All experiments
used a discount of = 0:99 and an RMSProp decay factor of  = 0:99.

"""


class Config(object):
  ######hyperparameters######
  H_policy = 20 #hidden layers: based on the a3c paper: 
  # H_value = 20 #hidden layers
  T_max = 20 #20 episodes
  t_max = 5 #update for every 5 actions
  I_max = 5 #update for every 5 actions
  # target newtowrk is updated every 40000 frames, cnn: 16 
  discount = 0.99 #discount factor
  policy_lr = 1e-4 #learning rate for policy estimator
  value_lr = 1e-4  #learning rate for value estimator
  epsilon = 0.9 #episilon greedy factor
  nActors = 1 # number of actors

  ######environment info#######
  env = gym.make("Pong-v0") 
  #env = gym.make("MsPacman-v0")
  nfeatures = #
  actions = env.action_space.n

  #resume = False

# based on the paper: policy & value estimators share the parameters and 
class ActorCriticServer():
    def __init__(self, namespace = "global"):
        self.T = self.Config.batch_size
        self.reward = 0
        # self.network = 

        self.policy = self.LSTMModel()
        self.actors = self.Config.nActors*(ActorCriticActor(self))
        self.T = 0
        self.theta_ = self.server.policy.theta
        self.theta_v_ = self.server.policy.theta_v


  def update_batch(self):
    # sequentially update actors from multipl

  def eval(self):

class ActorCriticActor():
  def __init__(self, server, namespace = "local"):
    self.t = 1 # counter
    self.env = self.Config.env #local environment
    self.env.render()
    
    self.R = 0.0 #reward
    self.Adv = 0.0 #advantage
    self.rollout = [] #rollout (do the update in minibatch)
    self.policy = self.server.policy
    self.s0 = self.policy.initial_state

  def reset(self):
    self.env.render()
    self.theta_ = tf.zeros()
    self.theta_v_ = tf.zeros()
    self.s0 = self.server.policy.initial_state
    self.t = 1
    self.R = 0.0
    #self.Adv = 0.0

  def play(self):
    self.reset()
    done = false
    while !done or t <= self.Config.t_max:
      a_t = self.policy.get_action()
      s_t, r, done, _ = env.step(a_t)

      v_t = self.policy.get_value(s_t)
      r += r*gamma**(t-1) if !done else v_t
      self.rollout.append(a_t, s_t, r) #do g

      self.t += 1
      self.server.T += 1

  def updateGradient(self): #update gradients in the rollout
    for (a_t, s_t, R, v_t) in self.rollout:
      v_t = self.policy.get_value(s_t)
      self.theta_ += tf.gradient(tf.log(self.policy.predict(a_t, s_t)))*(R - v_t)
      self.theta_v += tf.gradient((R - v_t)*(R - v_t))

  


