from . import ModelBase


__all__ = [
	'NupicModel',
]


class NupicModel(ModelBase):
	"""
	Creates a Nupic Network for Reinforcement Learning (Q-Learning).

	This model depends on the `spread-predictor` package to be running inside
	a flas docker container and available to receive HTTP requests on
	port 5000.
	"""

	# HIGH: write the NupicModel as an interface to spread-predictor.
	# HIGH: write NupicModel.__init__() method.
	# HIGH: write NupicModel.run() method.

	def __init__(self, num_actions, replay_memory, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.count_states = 0
		self.count_episodes = 0

	def get_count_states(self):
		"""
		Get the number of states that has been processed in the game-environment.
		This is not used by the TensorFlow graph. It is just a hack to save and
		reload the counter along with the checkpoint-file.
		"""
		return self.count_states

	def get_count_episodes(self):
		"""
		Get the number of episodes that has been processed in the game-environment.
		"""
		return self.count_episodes

	def increase_count_states(self):
		"""
		Increase the number of states that has been processed
		in the game-environment.
		"""
		self.count_states += 1

	def increase_count_episodes(self):
		"""
		Increase the number of episodes that has been processed
		in the game-environment.
		"""
		self.count_episodes += 1

	def get_predicted_actions(self, observation):
		"""
		Get the estimated Q-values for the given states from the Nupic predictor.

		The output of this function is an array of Q-value-arrays.
		There is a Q-value for each possible action in the game-environment.
		So the output is a 3-dim array of Open AI discrete shapes:
			[spaces.Discrete(x), spaces.Discrete(y), spaces.Discrete(z)]
		"""
		raise NotImplementedError('connect this method to the Nupic predictor using HTTP POST request.')




