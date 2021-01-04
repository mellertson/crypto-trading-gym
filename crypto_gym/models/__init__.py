from .nupic import *
from .tensforflow import *


__all__ = [
	'ModelBase',
	'TensorflowModel',
	'NupicModel',
]


class ModelBase(object):
	""" Base class used to implement a Q-learning model. """

	# HIGH: finish writing the Q-learning base class.

	def __init__(self, num_actions, replay_memory, *args, **kwargs):
		super().__init__(*args, **kwargs)

	@property
	def name(self):
		return f'{self.__class__.__name__}'

	def get_predicted_actions(self, observation):
		"""
		Calculate and return the estimated Q-values for the given states.

		The output of this function is an array of Q-value-arrays.
		There is a Q-value for each possible action in the game-environment.
		So the output is a 3-dim array of Open AI discrete shapes:
			[spaces.Discrete(x), spaces.Discrete(y), spaces.Discrete(z)]
		"""
		raise NotImplementedError('subclasses must implement get_q_values()')

	def optimize(self, min_epochs=1.0, max_epochs=10,
				 batch_size=128, loss_limit=0.015,
				 learning_rate=1e-3):
		"""
		Optimize the Neural Network by sampling states and Q-values
		from the replay-memory.

		The original DeepMind paper performed one optimization iteration
		after processing each new state of the game-environment. This is
		an un-natural way of doing optimization of Neural Networks.

		So instead we perform a full optimization run every time the
		Replay Memory is full (or it is filled to the desired fraction).
		This also gives more efficient use of a GPU for the optimization.

		The problem is that this may over-fit the Neural Network to whatever
		is in the replay-memory. So we use several tricks to try and adapt
		the number of optimization iterations.

		:param min_epochs:
			Minimum number of optimization epochs. One epoch corresponds
			to the replay-memory being used once. However, as the batches
			are sampled randomly and biased somewhat, we may not use the
			whole replay-memory. This number is just a convenient measure.

		:param max_epochs:
			Maximum number of optimization epochs.

		:param batch_size:
			Size of each random batch sampled from the replay-memory.

		:param loss_limit:
			Optimization continues until the average loss-value of the
			last 100 batches is below this value (or max_epochs is reached).

		:param learning_rate:
			Learning-rate to use for the optimizer.
		"""
		raise NotImplementedError('subclasses must implement optimize')

	def close(self):
		"""Close the TensorFlow session."""
		pass

	def get_count_states(self):
		"""
		Get the number of states that has been processed in the game-environment.
		This is not used by the TensorFlow graph. It is just a hack to save and
		reload the counter along with the checkpoint-file.
		"""
		raise NotImplementedError()

	def get_count_episodes(self):
		"""
		Get the number of episodes that has been processed in the game-environment.
		"""
		raise NotImplementedError()

	def increase_count_states(self):
		"""
		Increase the number of states that has been processed
		in the game-environment.
		"""
		raise NotImplementedError()

	def increase_count_episodes(self):
		"""
		Increase the number of episodes that has been processed
		in the game-environment.
		"""
		raise NotImplementedError()

	def save_checkpoint(count_states):
		"""
		Serialize the model to the Django database so it can be loaded later.
		"""
		print(f'TODO: implement {self.name}.save_checkpoint()!')



