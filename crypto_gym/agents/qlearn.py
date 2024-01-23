from time import sleep
from datetime import datetime, timedelta, timezone
import multiprocessing as mp
from multiprocessing import queues
import copy
import numpy as np
from crypto_gym.models import nupic as models
from crypto_gym.envs import CryptoEnv


__all__ = [
	'ActionFlattener',
	'ReplayMemory',
	'EpsilonGreedy',
	'QLearningAgent',
]


class ActionFlattener(object):
	"""
	Flattens and restores a tuple of tuples remembering its structure.

	:ivar actions: The "flattened" actions.
	:type actions: tuple of float
	:ivar original_actions: The original, "unflattened" actions.
	:type original_actions: tuple of tuple
	"""

	def __init__(self, actions, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.lengths = self.get_lengths(actions)
		self.original_actions = actions
		self.actions = self.flatten(actions)

	@classmethod
	def get_lengths(cls, actions):
		lengths = []
		for subactions in actions:
			lengths.append(len(subactions))
		return lengths

	@classmethod
	def flatten(cls, actions):
		flattened_actions = []
		for subactions in actions:
			flattened_actions.extend(subactions)
		return flattened_actions


class ReplayMemory:
	""" The replay memory used specifically by the Nupic Q-Learning Agent. """

	def __init__(self, num_input_fields, size, num_actions, discount_factor=0.97):
		"""

		:param num_input_fields:
			Shape of the state-array.

		:param size:
			Capacity of the replay-memory. This is the number of states.

		:param num_actions:
			Number of possible actions in the game-environment.

		:param discount_factor:
			Discount-factor used for updating Q-values.
		"""

		# Array for the previous states of the game-environment.
		self.states = np.zeros(shape=[size, num_input_fields], dtype=np.float64)

		# Array for the Q-values corresponding to the states.
		self.q_values = np.zeros(shape=[size, num_actions], dtype=np.float64)

		# Array for the Q-values before being updated.
		# This is used to compare the Q-values before and after the update.
		self.q_values_old = np.zeros(shape=[size, num_actions], dtype=np.float64)

		# Action indices taken for each of the states in the memory.
		self.actions = np.zeros(shape=size, dtype=np.int)

		# Rewards observed for each of the states in the memory.
		self.rewards = np.zeros(shape=size, dtype=np.float64)

		# Whether the life had ended in each state of the game-environment.
		self.end_life = np.zeros(shape=size, dtype=np.bool)

		# Whether the episode had ended (aka. game over) in each state.
		self.end_episode = np.zeros(shape=size, dtype=np.bool)

		# Estimation errors for the Q-values. This is used to balance
		# the sampling of batches for training the Neural Network,
		# so we get a balanced combination of states with high and low
		# estimation errors for their Q-values.
		self.estimation_errors = np.zeros(shape=size, dtype=np.float)

		# Capacity of the replay-memory as the number of states.
		self.size = size

		# Discount-factor for calculating Q-values.
		self.discount_factor = discount_factor

		# Reset the number of used states in the replay-memory.
		self.num_used = 0

		# Threshold for splitting between low and high estimation errors.
		self.error_threshold = 0.1

	def get_count_states(self):
		return self.num_used

	def is_full(self):
		"""Return boolean whether the replay-memory is full."""
		return self.num_used == self.size

	def used_fraction(self):
		"""Return the fraction of the replay-memory that is used."""
		return self.num_used / self.size

	def reset(self):
		"""Reset the replay-memory so it is empty."""
		self.num_used = 0

	def add(self, state, q_values, action, reward, end_life, end_episode):
		"""
		Add an observed state from the game-environment, along with the
		estimated Q-values, action taken, observed reward, etc.

		:param state: Current state of the game-environment.
		:type state: tuple of float

		:param q_values: The estimated Q-values for the state.
		:type q_values: tuple of float

		:param action: The action taken by the agent in this state of the game.
		:type action: tuple of float

		:param reward: The reward that was observed from taking this action
			and moving to the next state.
		:type reward: float

		:param end_life: True if the agent has lost a life in this state.
		:type end_life: bool

		:param end_episode: True if the agent has lost all lives
			aka. "game over" aka. "end of episode".
		:type end_episode: bool
		"""

		if not self.is_full():
			# Index into the arrays for convenience.
			k = self.num_used

			# Increase the number of used elements in the replay-memory.
			self.num_used += 1

			# Store all the values in the replay-memory.
			self.states[k] = state
			self.q_values[k] = q_values
			self.actions[k] = action
			self.end_life[k] = end_life
			self.end_episode[k] = end_episode

			# Note that the reward is limited. This is done to stabilize
			# the training of the Neural Network.
			# Mike E Note: I don't think we should clip the reward, because
			# we are not playing an Atari game, and the original author's
			# code was optimized for an Atary game.
			# self.rewards[k] = np.clip(reward, -1.0, 1.0)
			pass

	def update_all_q_values(self):
		"""
		Update all Q-values in the replay-memory.

		When states and Q-values are added to the replay-memory, the
		Q-values have been estimated by the Neural Network. But we now
		have more data available that we can use to improve the estimated
		Q-values, because we now know which actions were taken and the
		observed rewards. We sweep backwards through the entire replay-memory
		to use the observed data to improve the estimated Q-values.
		"""

		# Copy old Q-values so we can print their statistics later.
		# Note that the contents of the arrays are copied.
		self.q_values_old[:] = self.q_values[:]

		# Process the replay-memory backwards and update the Q-values.
		# This loop could be implemented entirely in NumPy for higher speed,
		# but it is probably only a small fraction of the overall time usage,
		# and it is much easier to understand when implemented like this.
		for k in reversed(range(self.num_used - 1)):
			# Get the data for the k'th state in the replay-memory.
			action = self.actions[k]
			reward = self.rewards[k]
			end_life = self.end_life[k]
			end_episode = self.end_episode[k]

			# Calculate the Q-value for the action that was taken in this state.
			if end_life or end_episode:
				# If the agent lost a life or it was game over / end of episode,
				# then the value of taking the given action is just the reward
				# that was observed in this single step. This is because the
				# Q-value is defined as the discounted value of all future game
				# steps in a single life of the agent. When the life has ended,
				# there will be no future steps.
				action_value = reward
			else:
				# Otherwise the value of taking the action is the reward that
				# we have observed plus the discounted value of future rewards
				# from continuing the game. We use the estimated Q-values for
				# the following state and take the maximum, because we will
				# generally take the action that has the highest Q-value.
				action_value = reward + self.discount_factor * np.max(self.q_values[k + 1])

			# Error of the Q-value that was estimated using the Neural Network.
			self.estimation_errors[k] = abs(action_value - self.q_values[k, action])

			# Update the Q-value with the better estimate.
			self.q_values[k, action] = action_value

		self.print_statistics()

	def print_statistics(self):
		"""Print statistics for the contents of the replay-memory."""

		print("Replay-memory statistics:")

		# Print statistics for the Q-values before they were updated
		# in update_all_q_values().
		msg = "\tQ-values Before, Min: {0:5.2f}, Mean: {1:5.2f}, Max: {2:5.2f}"
		print(msg.format(np.min(self.q_values_old),
						 np.mean(self.q_values_old),
						 np.max(self.q_values_old)))

		# Print statistics for the Q-values after they were updated
		# in update_all_q_values().
		msg = "\tQ-values After,  Min: {0:5.2f}, Mean: {1:5.2f}, Max: {2:5.2f}"
		print(msg.format(np.min(self.q_values),
						 np.mean(self.q_values),
						 np.max(self.q_values)))

		# Print statistics for the difference in Q-values before and
		# after the update in update_all_q_values().
		q_dif = self.q_values - self.q_values_old
		msg = "\tQ-values Diff.,  Min: {0:5.2f}, Mean: {1:5.2f}, Max: {2:5.2f}"
		print(msg.format(np.min(q_dif),
						 np.mean(q_dif),
						 np.max(q_dif)))

		# Print statistics for the number of large estimation errors.
		# Don't use the estimation error for the last state in the memory,
		# because its Q-values have not been updated.
		err = self.estimation_errors[:-1]
		err_count = np.count_nonzero(err > self.error_threshold)
		msg = "\tNumber of large errors > {0}: {1} / {2} ({3:.1%})"
		print(msg.format(self.error_threshold, err_count,
						 self.num_used, err_count / self.num_used))

		# How much of the replay-memory is used by states with end_life.
		end_life_pct = np.count_nonzero(self.end_life) / self.num_used

		# How much of the replay-memory is used by states with end_episode.
		end_episode_pct = np.count_nonzero(self.end_episode) / self.num_used

		# How much of the replay-memory is used by states with non-zero reward.
		reward_nonzero_pct = np.count_nonzero(self.rewards) / self.num_used

		# Print those statistics.
		msg = "\tend_life: {0:.1%}, end_episode: {1:.1%}, reward non-zero: {2:.1%}"
		print(msg.format(end_life_pct, end_episode_pct, reward_nonzero_pct))


class LinearControlSignal:
	"""
	A control signal that changes linearly over time.

	This is used to change e.g. the learning-rate for the optimizer
	of the Neural Network, as well as other parameters.

	TensorFlow has functionality for doing this, but it uses the
	global_step counter inside the TensorFlow graph, while we
	want the control signals to use a state-counter for the
	game-environment. So it is easier to make this in Python.
	"""

	def __init__(self, start_value, end_value, num_iterations, repeat=False):
		"""
		Create a new object.

		:param start_value:
			Start-value for the control signal.

		:param end_value:
			End-value for the control signal.

		:param num_iterations:
			Number of iterations it takes to reach the end_value
			from the start_value.

		:param repeat:
			Boolean whether to reset the control signal back to the start_value
			after the end_value has been reached.
		"""

		# Store arguments in this object.
		self.start_value = start_value
		self.end_value = end_value
		self.num_iterations = num_iterations
		self.repeat = repeat

		# Calculate the linear coefficient.
		self._coefficient = (end_value - start_value) / num_iterations

	def get_value(self, iteration):
		"""Get the value of the control signal for the given iteration."""

		if self.repeat:
			iteration %= self.num_iterations

		if iteration < self.num_iterations:
			value = iteration * self._coefficient + self.start_value
		else:
			value = self.end_value

		return value


class EpsilonGreedy:
	"""
	The epsilon-greedy policy either takes a random action with
	probability epsilon, or it takes the action for the highest
	Q-value.

	If epsilon is 1.0 then the actions are always random.
	If epsilon is 0.0 then the actions are always argmax for the Q-values.

	Epsilon is typically decreased linearly from 1.0 to 0.1
	and this is also implemented in this class.

	During testing, epsilon is usually chosen lower, e.g. 0.05 or 0.01
	"""

	def __init__(self, num_actions,
				 epsilon_testing=0.05,
				 num_iterations=10,
				 start_value=1.0, end_value=0.1,
				 repeat=False):
		"""

		:param num_actions:
			Number of possible actions in the game-environment.

		:param epsilon_testing:
			Epsilon-value when testing.

		:param num_iterations:
			Number of training iterations required to linearly
			decrease epsilon from start_value to end_value.

		:param start_value:
			Starting value for linearly decreasing epsilon.

		:param end_value:
			Ending value for linearly decreasing epsilon.

		:param repeat:
			Boolean whether to repeat and restart the linear decrease
			when the end_value is reached, or only do it once and then
			output the end_value forever after.
		"""

		# Store parameters.
		self.num_actions = num_actions
		self.epsilon_testing = epsilon_testing

		# Create a control signal for linearly decreasing epsilon.
		self.epsilon_linear = LinearControlSignal(num_iterations=num_iterations,
												  start_value=start_value,
												  end_value=end_value,
												  repeat=repeat)

	def get_epsilon(self, iteration, training=True):
		"""
		Return the epsilon for the given iteration.
		If training==True then epsilon is linearly decreased,
		otherwise epsilon is a fixed number.
		"""

		if training:
			epsilon = self.epsilon_linear.get_value(iteration=iteration)
		else:
			epsilon = self.epsilon_testing

		return epsilon

	def get_action(self, q_values, iteration, training=True):
		"""
		Use the epsilon-greedy policy to select an action.

		:param q_values:
			These are the Q-values that are estimated by the Neural Network
			for the current state of the game-environment.

		:param iteration:
			This is an iteration counter. Here we use the number of states
			that has been processed in the game-environment.

		:param training:
			Boolean whether we are training or testing the
			Reinforcement Learning agent.

		:return:
			action (integer), epsilon (float)
		"""

		epsilon = self.get_epsilon(iteration=iteration, training=training)

		# With probability epsilon.
		if np.random.random() < epsilon:
			# Select a random action.
			action = np.random.randint(low=0, high=self.num_actions)
		else:
			# Otherwise select the action that has the highest Q-value.
			action = np.argmax(q_values)

		return action, epsilon


class QLearningAgent(object):
	"""
	Agent which uses Q-Learning with a neural network to trade crypto.

	:ivar model: Either a nupic or tensorflow model, which makes predictions.
	:type model: crypto_gym.models.NupicModel

	:ivar rp_mem_size: Cycles before optimizing the network using Q-Learning.
	:type rp_mem_size: int
	"""

	def __init__(self, env_name, exchange, base, quote, period_secs, ob_levels,
				 base_url, discount_factor=0.97, render=False, rp_mem_size=120):
		"""
		Constructor

		:param env_name:
		:param exchange:
		:param base:
		:param quote:
		:param period_secs:
		:param ob_levels:
		:param str base_url:
			The URL of the Algo-Backend Django REST API, to get market
			data from.
		:param discount_factor:
		:param render:
		:param rp_mem_size:
		"""
		self.env_name = env_name
		self.exchange = exchange
		self.base = base
		self.quote = quote
		self.period_secs = period_secs
		self.period_td = timedelta(seconds=period_secs)
		self.ob_levels = ob_levels
		self.base_url = base_url
		self.discount_factor = discount_factor
		self.render = render
		self.rp_mem_size = rp_mem_size
		self._print_buffer = ''
		# instantiate Open AI Gym environment
		# self.env = gym.make(env_name)
		self.env = CryptoEnv(
			self.exchange,
			self.base,
			self.quote,
			self.period_secs,
			self.ob_levels,
			self.base_url,
		)
		# The number of possible actions that the agent may take in every step.
		self.num_actions = sum(self.env.action_space.nvec)
		# Epsilon-greedy policy for selecting an action from the Q-values.
		# During training the epsilon is decreased linearly over the given
		# number of iterations. During testing the fixed epsilon is used.
		self.epsilon_greedies = []
		for action_names in self.env.action_names.values():
			self.epsilon_greedies.append(
				EpsilonGreedy(
					start_value=0.01,
					end_value=0.0,
					num_iterations=10,
					num_actions=len(action_names),
					epsilon_testing=0.001,
				)
			)

		# TRAINING ivars
		# The learning-rate for the optimizer decreases linearly.
		self.learning_rate_control = LinearControlSignal(
			start_value=1e-3,
			end_value=1e-5,
			num_iterations=1000,
		)
		# The loss-limit is used to abort the optimization whenever the
		# mean batch-loss falls below this limit.
		self.loss_limit_control = LinearControlSignal(
			start_value=0.1,
			end_value=0.015,
			num_iterations=1000,
		)
		# The maximum number of epochs to perform during optimization.
		# This is increased from 5 to 10 epochs, because it was found for
		# the Breakout-game that too many epochs could be harmful early
		# in the training, as it might cause over-fitting.
		# Later in the training we would occasionally get rare events
		# and would therefore have to optimize for more iterations
		# because the learning-rate had been decreased.
		self.max_epochs_control = LinearControlSignal(
			start_value=1.0,
			end_value=1.0,
			num_iterations=5e6,
		)
		# The fraction of the replay-memory to be used.
		# Early in the training, we want to optimize more frequently
		# so the Neural Network is trained faster and the Q-values
		# are learned and updated more often. Later in the training,
		# we need more samples in the replay-memory to have sufficient
		# diversity, otherwise the Neural Network will over-fit.
		self.replay_fraction = LinearControlSignal(
			start_value=0.1,
			end_value=1.0,
			num_iterations=1000,
		)
		# We only create the replay-memory when we are training the agent,
		# because it requires a lot of RAM.
		self.replay_memories = []
		for action_names in self.env.action_names.values():
			replay_memory = ReplayMemory(
				num_input_fields=len(self.env.get_input_field_names()),
				size=rp_mem_size,
				num_actions=len(action_names),
				discount_factor=self.discount_factor,
			)
			self.replay_memories.append(replay_memory)

		# instantiate the nupic model
		model_class_name = 'NupicModel'
		model_class = getattr(models, model_class_name, None)
		if model_class is None:
			raise ValueError(f'{model_class_name} class not found in {models}')
		self.model = model_class(
			self.exchange,
			self.base,
			self.quote,
			self.period_secs,
			self.env.get_input_field_names(),
			self.env.action_names,
		)
		# Log of the rewards obtained in each episode during calls to run()
		self.episode_rewards = []
		# multi-processing stuff for Celery
		self.command_queue = mp.Queue()
		self._is_running = False
		self._is_running_lock = mp.Lock()

	@property
	def name(self):
		return f'{self.__class__.__name__}'

	@property
	def is_running(self):
		_is_running = None
		with self._is_running_lock:
			_is_running = copy.deepcopy(self._is_running)
		return _is_running

	@is_running.setter
	def is_running(self, value):
		with self._is_running_lock:
			self._is_running = value
		return value

	def command_queue_processor(self):
		while self.is_running:
			try:
				command = self.command_queue.get(timeout=1.0)
				if command == 'stop':
					print(f'stop command recieved by: {self.name}')
					self.is_running = False
			except queues.Empty:
				pass

	def run(self, num_episodes=None):
		"""
		Run the game-environment and use the Neural Network to decide
		which actions to take in each step through Q-value estimates.

		:param num_episodes:
			Number of episodes to process in the game-environment.
			If None then continue forever. This is useful during training
			where you might want to stop the training using Ctrl-C instead.
		:type num_episodes: int
		:rtype: None
		"""
		current_episode = 0

		# Counter for the number of states we have processed.
		count_states = self.replay_memories[0].get_count_states()
		next_episode_time = datetime.now() - self.period_td

		while self.is_running:
			if num_episodes is not None:
				if current_episode > num_episodes:
					break
			observation = self.env.get_next_observation()
			if not self.env.is_observation_space_changed(observation):
				sleep(self.period_td.total_seconds())
				continue

			current_episode += 1

			# get q-values as tuple of tuple
			q_values = self.model.get_q_values(next_episode_time, observation)

			# Determine the action that the agent must take in the game-environment.
			# The epsilon is just used for printing further below.
			actions = []
			for epsilon_greedy in self.epsilon_greedies:
				i = self.epsilon_greedies.index(epsilon_greedy)
				action, epsilon = epsilon_greedy.get_action(
					q_values=q_values[i],
					iteration=count_states,
				)
				actions.append(action)
			actions = tuple(actions)
			assert(len(actions) == 3)

			# Take a step in the game-environment using the given action.
			observation, reward, end_episode, info = self.env.step(
				actions,
				observation=observation,
			)

			# TODO: calculate if the agent has "lost a life" in `end_life` var.
			end_life = False

			if self.render:
				self.env.render()

			if not self.env.last_action_was_executed:
				current_episode -= 1
				sleep(self.period_td.total_seconds())
				continue

			# Add the state to the replay-memories instances.
			for replay_memory in self.replay_memories:
				i = self.replay_memories.index(replay_memory)
				replay_memory.add(
					state=observation,
					q_values=q_values[i],
					action=actions[i],
					reward=reward,
					end_life=end_life,
					end_episode=end_episode,
				)

			# How much of the replay-memory should be used.
			use_fraction = self.replay_fraction.get_value(iteration=count_states)

			# When the replay-memory is sufficiently full.
			if self.replay_memories[0].is_full() \
				or self.replay_memories[0].used_fraction() > use_fraction:
				self.print_line('=')
				print(f'Training the network with Q-Learning!')
				self.print_line('=')

				# Update all Q-values in the replay-memory through a backwards-sweep.
				for replay_memory in self.replay_memories:
					replay_memory.update_all_q_values()

				# Get the control parameters for optimization of the Neural Network.
				# These are changed linearly depending on the state-counter.
				learning_rate = self.learning_rate_control.get_value(iteration=count_states)
				loss_limit = self.loss_limit_control.get_value(iteration=count_states)
				max_epochs = self.max_epochs_control.get_value(iteration=count_states)

				# Perform an optimization run on the Neural Network so as to
				# improve the estimates for the Q-values.
				self.model.optimize(
					self.replay_memories,
					next_episode_time,
				)

				# Save a checkpoint of the Neural Network so we can reload it.
				# COMPLETED: write NupicModel.save_checkpoint() method.
				self.model.save_checkpoint()

				# Reset the replay-memory. This throws away all the data we have
				# just gathered, so we will have to fill the replay-memory again.
				for replay_memory in self.replay_memories:
					replay_memory.reset()

			# re-init next episode time
			next_episode_time = next_episode_time + self.period_td
			print(f'Action: {self.env.last_action_order_type}, '
				  f'Price: {self.env.last_action_price}, '
				  f'Amount: {self.env.last_action_amount}')
			print(f'Sleeping {self.period_td.total_seconds()} seconds...')
			print(f'End of episode: {current_episode}')
			self.print_line('x')
			sleep(self.period_td.total_seconds())
			self.env.last_observation = self.env.get_next_observation()

	def print_line(self, x):
		line = f'{x}' * 100
		print(f'\n{line}\n')






