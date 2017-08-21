import gym
import universe
import target_network

"""env = gym.make('flashgames.DuskDrive-v0')
env.configure(remotes=1) #automatically creates a local docker container
observation_n = env.reset()

while True:
	action_n = [[('KeyEvent', 'ArrowUp', True)] for ob in observation_n] # your agent here
	observation_n, reward_n, done_n, info = env.step(action_n)
	env.render()
"""


def train():
	with tf.Graph().as_default():
    	global_step = tf.contrib.framework.get_or_create_global_step()


      	# steps
      	"""
      	1. create a computation graph
      	2. repeat N times:
      		take an action, action (1. random(), under policy gradient + episilon )
      		get the envrionment (environment.step())

      	2. inference ()
      	3. 
      	"""

    	# Force input pipeline to CPU:0 to avoid operations sometimes ending up on
    	# GPU and resulting in a slow down.
    	with tf.device('/cpu:0'):
      		images, labels = target_network.input()

      	# build the graph that computes the logits predictions from the inference model.
      	logits = target_network.inference()

		# Calculate loss.
      	loss = target_network.loss(logits, labels)

      	# build the graph that trains the model with one batch of examples and updates the model parameters.
      	train_op = target_network.train(loss, global_step)





    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""

      def begin(self):
        self._step = -1
        self._start_time = time.time()

      def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs(loss)  # Asks for loss value.

      def after_run(self, run_context, run_values):
        if self._step % FLAGS.log_frequency == 0:
          current_time = time.time()
          duration = current_time - self._start_time
          self._start_time = current_time

          loss_value = run_values.results
          examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
          sec_per_batch = float(duration / FLAGS.log_frequency)

          format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print (format_str % (datetime.now(), self._step, loss_value,
                               examples_per_sec, sec_per_batch))

    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=FLAGS.train_dir,
        hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
               tf.train.NanTensorHook(loss),
               _LoggerHook()],
        config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement)) as mon_sess:
      while not mon_sess.should_stop():
        mon_sess.run(train_op)


def main(argv=None):  # pylint: disable=unused-argument
  #cifar10.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train()


if __name__ == '__main__':
  tf.app.run()
