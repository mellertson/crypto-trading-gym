import requests, json, copy
from datetime import datetime
import os



__all__ = [
	'ModelBase',
	'PREDICTOR_SERVER_BASE_URL',
	'set_predictor_server_base_url',
]


PREDICTOR_SERVER_BASE_URL = os.environ.get(
	'PREDICTOR_SERVER_BASE_URL',
	default='http://localhost:5000',
)


def set_predictor_server_base_url(url):
	PREDICTOR_SERVER_BASE_URL = url
	os.environ['PREDICTOR_SERVER_BASE_URL'] = url
	return PREDICTOR_SERVER_BASE_URL


class ModelBase(object):
	""" Base class used to implement a Q-learning model. """

	# HIGH: finish writing the q-learning ModelBase class.

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	@property
	def name(self):
		return f'{self.__class__.__name__}'

	def get_q_values(self, observation):
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

	def save_checkpoint(self, count_states):
		"""
		Serialize the model to the Django database so it can be loaded later.
		"""
		# TODO: implement .save_checkpoint() method
		pass


