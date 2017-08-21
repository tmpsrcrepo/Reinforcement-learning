class Policy(object):
  def __init__(self, config): 
    self.num_steps = config.num_steps
    self.theta_v = []
    self.action_probs = []

  def get_action_probs(self):
  	return self.action_probs

  def get_weights(self):
  	return self.theta_v

  def update_model(self, features, labels):
  	return