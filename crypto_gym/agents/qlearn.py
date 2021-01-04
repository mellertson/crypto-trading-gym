from time import sleep
from datetime import datetime, timedelta, timezone
import multiprocessing as mp
from multiprocessing import queues
import copy
from crypto_gym import models


__all__ = [
	'QLearningAgent',
]


class ReplayMemory:
	"""
	The replay-memory holds many previous states of the game-environment.
	This helps stabilize training of the Neural Network because the data
	is more diverse when sampled over thousands of different states.
	"""

	def __init__(self, state_shape, size, num_actions, discount_factor=0.97):
		"""

		:param state_shape:
			Shape of the state-array.

		:param size:
			Capacity of the replay-memory. This is the number of states.

		:param num_actions:
			Number of possible actions in the game-environment.

		:param discount_factor:
			Discount-factor used for updating Q-values.
		"""

		# Array for the previous states of the game-environment.
		self.states = np.zeros(shape=[size] + state_shape, dtype=np.uint8)

		# Array for the Q-values corresponding to the states.
		self.q_values = np.zeros(shape=[size, num_actions], dtype=np.float)

		# Array for the Q-values before being updated.
		# This is used to compare the Q-values before and after the update.
		self.q_values_old = np.zeros(shape=[size, num_actions], dtype=np.float)

		# Actions taken for each of the states in the memory.
		self.actions = np.zeros(shape=size, dtype=np.int)

		# Rewards observed for each of the states in the memory.
		self.rewards = np.zeros(shape=size, dtype=np.float)

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

		:param state:
			Current state of the game-environment.
			This is the output of the MotionTracer-class.

		:param q_values:
			The estimated Q-values for the state.

		:param action:
			The action taken by the agent in this state of the game.

		:param reward:
			The reward that was observed from taking this action
			and moving to the next state.

		:param end_life:
			Boolean whether the agent has lost a life in this state.

		:param end_episode:
			Boolean whether the agent has lost all lives aka. game over
			aka. end of episode.
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
			self.rewards[k] = np.clip(reward, -1.0, 1.0)

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

	def prepare_sampling_prob(self, batch_size=128):
		"""
		Prepare the probability distribution for random sampling of states
		and Q-values for use in training of the Neural Network.

		The probability distribution is just a simple binary split of the
		replay-memory based on the estimation errors of the Q-values.
		The idea is to create a batch of samples that are balanced somewhat
		evenly between Q-values that the Neural Network already knows how to
		estimate quite well because they have low estimation errors, and
		Q-values that are poorly estimated by the Neural Network because
		they have high estimation errors.

		The reason for this balancing of Q-values with high and low estimation
		errors, is that if we train the Neural Network mostly on data with
		high estimation errors, then it will tend to forget what it already
		knows and hence become over-fit so the training becomes unstable.
		"""

		# Get the errors between the Q-values that were estimated using
		# the Neural Network, and the Q-values that were updated with the
		# reward that was actually observed when an action was taken.
		err = self.estimation_errors[0:self.num_used]

		# Create an index of the estimation errors that are low.
		idx = err < self.error_threshold
		self.idx_err_lo = np.squeeze(np.where(idx))

		# Create an index of the estimation errors that are high.
		self.idx_err_hi = np.squeeze(np.where(np.logical_not(idx)))

		# Probability of sampling Q-values with high estimation errors.
		# This is either set to the fraction of the replay-memory that
		# has high estimation errors - or it is set to 0.5. So at least
		# half of the batch has high estimation errors.
		prob_err_hi = len(self.idx_err_hi) / self.num_used
		prob_err_hi = max(prob_err_hi, 0.5)

		# Number of samples in a batch that have high estimation errors.
		self.num_samples_err_hi = int(prob_err_hi * batch_size)

		# Number of samples in a batch that have low estimation errors.
		self.num_samples_err_lo = batch_size - self.num_samples_err_hi

	def random_batch(self):
		"""
		Get a random batch of states and Q-values from the replay-memory.
		You must call prepare_sampling_prob() before calling this function,
		which also sets the batch-size.

		The batch has been balanced so it contains states and Q-values
		that have both high and low estimation errors for the Q-values.
		This is done to both speed up and stabilize training of the
		Neural Network.
		"""

		# Random index of states and Q-values in the replay-memory.
		# These have LOW estimation errors for the Q-values.
		idx_lo = np.random.choice(self.idx_err_lo,
								  size=self.num_samples_err_lo,
								  replace=False)

		# Random index of states and Q-values in the replay-memory.
		# These have HIGH estimation errors for the Q-values.
		idx_hi = np.random.choice(self.idx_err_hi,
								  size=self.num_samples_err_hi,
								  replace=False)

		# Combine the indices.
		idx = np.concatenate((idx_lo, idx_hi))

		# Get the batches of states and Q-values.
		states_batch = self.states[idx]
		q_values_batch = self.q_values[idx]

		return states_batch, q_values_batch

	def all_batches(self, batch_size=128):
		"""
		Iterator for all the states and Q-values in the replay-memory.
		It returns the indices for the beginning and end, as well as
		a progress-counter between 0.0 and 1.0.

		This function is not currently being used except by the function
		estimate_all_q_values() below. These two functions are merely
		included to make it easier for you to experiment with the code
		by showing you an easy and efficient way to loop over all the
		data in the replay-memory.
		"""

		# Start index for the current batch.
		begin = 0

		# Repeat until all batches have been processed.
		while begin < self.num_used:
			# End index for the current batch.
			end = begin + batch_size

			# Ensure the batch does not exceed the used replay-memory.
			if end > self.num_used:
				end = self.num_used

			# Progress counter.
			progress = end / self.num_used

			# Yield the batch indices and completion-counter.
			yield begin, end, progress

			# Set the start-index for the next batch to the end of this batch.
			begin = end

	def estimate_all_q_values(self, model):
		"""
		Estimate all Q-values for the states in the replay-memory
		using the model / Neural Network.

		Note that this function is not currently being used. It is provided
		to make it easier for you to experiment with this code, by showing
		you an efficient way to iterate over all the states and Q-values.

		:param model:
			Instance of the NeuralNetwork-class.
		"""

		print("Re-calculating all Q-values in replay memory ...")

		# Process the entire replay-memory in batches.
		for begin, end, progress in self.all_batches():
			# Print progress.
			msg = "\tProgress: {0:.0%}"
			msg = msg.format(progress)
			print_progress(msg)

			# Get the states for the current batch.
			states = self.states[begin:end]

			# Calculate the Q-values using the Neural Network
			# and update the replay-memory.
			self.q_values[begin:end] = model.get_q_values(observation=states)

		# Newline.
		print()

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
				 num_iterations=1e6,
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
	"""

	def __init__(self, env_name, model_class_name, exchange, base, quote,
				 period_secs, ob_levels, window_size, base_url, max_episodes,
				 discount_factor=0.97, render=False):
		self.env_name = env_name
		self.model_type = model_class_name
		self.exchange = exchange
		self.base = base
		self.quote = quote
		self.period_secs = period_secs
		self.period_td = timedelta(seconds=period_secs)
		self.ob_levels = ob_levels
		self.window_size = window_size
		self.base_url = base_url
		self.max_episodes = max_episodes
		self.discount_factor = discount_factor
		self.render = render
		# instantiate Open AI Gym environment
		self.env = gym.make(env_name)
		# The number of possible actions that the agent may take in every step.
		self.num_actions = self.env.action_space.n
		# Epsilon-greedy policy for selecting an action from the Q-values.
		# During training the epsilon is decreased linearly over the given
		# number of iterations. During testing the fixed epsilon is used.
		self.epsilon_greedy = EpsilonGreedy(
			start_value=1.0,
			end_value=0.1,
			num_iterations=1e6,
			num_actions=self.num_actions,
			epsilon_testing=0.01,
		)

		# TRAINING ivars
		# The learning-rate for the optimizer decreases linearly.
		self.learning_rate_control = LinearControlSignal(
			start_value=1e-3,
			end_value=1e-5,
			num_iterations=5e6,
		)
		# The loss-limit is used to abort the optimization whenever the
		# mean batch-loss falls below this limit.
		self.loss_limit_control = LinearControlSignal(
			start_value=0.1,
			end_value=0.015,
			num_iterations=5e6,
		)
		# The maximum number of epochs to perform during optimization.
		# This is increased from 5 to 10 epochs, because it was found for
		# the Breakout-game that too many epochs could be harmful early
		# in the training, as it might cause over-fitting.
		# Later in the training we would occasionally get rare events
		# and would therefore have to optimize for more iterations
		# because the learning-rate had been decreased.
		self.max_epochs_control = LinearControlSignal(
			start_value=5.0,
			end_value=10.0,
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
			num_iterations=5e6,
		)
		# We only create the replay-memory when we are training the agent,
		# because it requires a lot of RAM.
		self.replay_memory = ReplayMemory(
			size=200000,
			num_actions=self.num_actions,
			discount_factor=self.discount_factor,
		)
		# instantiate the nupic/tensorflow model
		model_class = getattr(models, model_class_name, None)
		if model_class is None:
			raise ValueError(f'{model_class_name} class not found in {models}')
		self.model = model_class(
			num_actions=self.num_actions,
			replay_memory=self.replay_memory,
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
			copy.deepcopy(self._is_running)
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
		# Counter for the number of states we have processed.
		count_states = self.replay_memory.get_count_states()
		now = datetime.now()
		next_cycle_timestamp = datetime(
			now.year,
			now.month,
			now.day,
			now.hour,
			now.minute,
			now.second,
		)
		next_cycle_timestamp = next_cycle_timestamp - self.period_td
		observation = self.env.get_next_observation()

		while self.is_running:
			if datetime.now() < next_cycle_timestamp:
				pause = next_cycle_timestamp - datetime.now()
				sleep(pause.total_seconds())
			q_values = self.model.get_predicted_actions(next_cycle_timestamp, observation)
			next_cycle_timestamp = next_cycle_timestamp + self.period_td
			# Determine the action that the agent must take in the game-environment.
			# The epsilon is just used for printing further below.
			action, epsilon = self.epsilon_greedy.get_action(
				q_values=q_values,
				iteration=count_states,
			)

			# Take a step in the game-environment using the given action.
			observation, reward, end_episode, info = self.env.step(action=action)

			# TODO: calculate if the agent has "lost a life" in `end_life` var.
			end_life = False

			if self.render:
				self.env.render()

			# Add the state of the game-environment to the replay-memory.
			self.replay_memory.add(state=observation,
								   q_values=q_values,
								   action=action,
								   reward=reward,
								   end_life=end_life,
								   end_episode=end_episode)

			# How much of the replay-memory should be used.
			use_fraction = self.replay_fraction.get_value(iteration=count_states)

			# When the replay-memory is sufficiently full.
			if self.replay_memory.is_full() \
				or self.replay_memory.used_fraction() > use_fraction:

				# Update all Q-values in the replay-memory through a backwards-sweep.
				self.replay_memory.update_all_q_values()

				# Get the control parameters for optimization of the Neural Network.
				# These are changed linearly depending on the state-counter.
				learning_rate = self.learning_rate_control.get_value(iteration=count_states)
				loss_limit = self.loss_limit_control.get_value(iteration=count_states)
				max_epochs = self.max_epochs_control.get_value(iteration=count_states)

				# Perform an optimization run on the Neural Network so as to
				# improve the estimates for the Q-values.
				# This will sample random batches from the replay-memory.
				# HIGH: write NupicModel.optimize() method.
				self.model.optimize(

					learning_rate=learning_rate,
					loss_limit=loss_limit,
					max_epochs=max_epochs,
				)

				# Save a checkpoint of the Neural Network so we can reload it.
				# HIGH: write NupicModel.save_checkpoint() method.
				self.model.save_checkpoint(count_states)

				# Reset the replay-memory. This throws away all the data we have
				# just gathered, so we will have to fill the replay-memory again.
				self.replay_memory.reset()

