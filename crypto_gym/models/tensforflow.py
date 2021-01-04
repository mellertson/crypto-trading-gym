import tensorflow.compat.v1 as tf
import numpy as np
import gym
import PIL.Image
import sys
import os
import time
import csv
import argparse


tf.disable_v2_behavior()


# TODO: remove unecessary import statements (above).


__all__ = [
	'TensorflowModel',
]


class TensorflowModel:
	"""
	Creates a Tensorflow Network for Reinforcement Learning (Q-Learning).

	Functions are provided for estimating Q-values from states of the
	game-environment, and for optimizing the Neural Network so it becomes
	better at estimating the Q-values.
	"""

	def __init__(self, num_actions, replay_memory):
		"""
		:param num_actions:
			Number of discrete actions for the game-environment.

		:param replay_memory:
			Object-instance of the ReplayMemory-class.

		# HIGH: add network type parameter with choices: 'nupic' | 'tensorflow'
		"""

		# Replay-memory used for sampling random batches.
		self.replay_memory = replay_memory

		# Path for saving/restoring checkpoints.
		self.checkpoint_path = os.path.join(checkpoint_dir, "checkpoint")

		# Placeholder variable for inputting states into the Neural Network.
		# A state is a multi-dimensional array holding image-frames from
		# the game-environment.
		self.x = tf.placeholder(dtype=tf.float32, shape=[None] + state_shape, name='x')

		# Placeholder variable for inputting the learning-rate to the optimizer.
		self.learning_rate = tf.placeholder(dtype=tf.float32, shape=[])

		# Placeholder variable for inputting the target Q-values
		# that we want the Neural Network to be able to estimate.
		self.q_values_new = tf.placeholder(tf.float32,
										   shape=[None, num_actions],
										   name='q_values_new')

		# This is a hack that allows us to save/load the counter for
		# the number of states processed in the game-environment.
		# We will keep it as a variable in the TensorFlow-graph
		# even though it will not actually be used by TensorFlow.
		self.count_states = tf.Variable(initial_value=0,
										trainable=False, dtype=tf.int64,
										name='count_states')

		# Similarly, this is the counter for the number of episodes.
		self.count_episodes = tf.Variable(initial_value=0,
										  trainable=False, dtype=tf.int64,
										  name='count_episodes')

		# TensorFlow operation for increasing count_states.
		self.count_states_increase = tf.assign(self.count_states,
											   self.count_states + 1)

		# TensorFlow operation for increasing count_episodes.
		self.count_episodes_increase = tf.assign(self.count_episodes,
												 self.count_episodes + 1)

		# The Neural Network will be constructed in the following.
		# Note that the architecture of this Neural Network is very
		# different from that used in the original DeepMind papers,
		# which was something like this:
		# Input image:      84 x 84 x 4 (4 gray-scale images of 84 x 84 pixels).
		# Conv layer 1:     16 filters 8 x 8, stride 4, relu.
		# Conv layer 2:     32 filters 4 x 4, stride 2, relu.
		# Fully-conn. 1:    256 units, relu. (Sometimes 512 units).
		# Fully-conn. 2:    num-action units, linear.

		# The DeepMind architecture does a very aggressive downsampling of
		# the input images so they are about 10 x 10 pixels after the final
		# convolutional layer. I found that this resulted in significantly
		# distorted Q-values when using the training method further below.
		# The reason DeepMind could get it working was perhaps that they
		# used a very large replay memory (5x as big as here), and a single
		# optimization iteration was performed after each step of the game,
		# and some more tricks.

		# Initializer for the layers in the Neural Network.
		# If you change the architecture of the network, particularly
		# if you add or remove layers, then you may have to change
		# the stddev-parameter here. The initial weights must result
		# in the Neural Network outputting Q-values that are very close
		# to zero - but the network weights must not be too low either
		# because it will make it hard to train the network.
		# You can experiment with values between 1e-2 and 1e-3.
		init = tf.truncated_normal_initializer(mean=0.0, stddev=2e-2)

		# This builds the Neural Network using the tf.layers API,
		# which is very verbose and inelegant, but should work for everyone.

		# Padding used for the convolutional layers.
		padding = 'SAME'

		# Activation function for all convolutional and fully-connected
		# layers, except the last.
		activation = tf.nn.relu

		# Reference to the lastly added layer of the Neural Network.
		# This makes it easy to add or remove layers.
		net = self.x

		# First convolutional layer.
		net = tf.layers.conv2d(inputs=net, name='layer_conv1',
							   filters=16, kernel_size=3, strides=2,
							   padding=padding,
							   kernel_initializer=init, activation=activation)

		# Second convolutional layer.
		net = tf.layers.conv2d(inputs=net, name='layer_conv2',
							   filters=32, kernel_size=3, strides=2,
							   padding=padding,
							   kernel_initializer=init, activation=activation)

		# Third convolutional layer.
		net = tf.layers.conv2d(inputs=net, name='layer_conv3',
							   filters=64, kernel_size=3, strides=1,
							   padding=padding,
							   kernel_initializer=init, activation=activation)

		# Flatten output of the last convolutional layer so it can
		# be input to a fully-connected (aka. dense) layer.
		net = tf.layers.flatten(net)

		# First fully-connected (aka. dense) layer.
		net = tf.layers.dense(inputs=net, name='layer_fc1', units=1024,
							  kernel_initializer=init, activation=activation)

		# Second fully-connected layer.
		net = tf.layers.dense(inputs=net, name='layer_fc2', units=1024,
							  kernel_initializer=init, activation=activation)

		# Third fully-connected layer.
		net = tf.layers.dense(inputs=net, name='layer_fc3', units=1024,
							  kernel_initializer=init, activation=activation)

		# Fourth fully-connected layer.
		net = tf.layers.dense(inputs=net, name='layer_fc4', units=1024,
							  kernel_initializer=init, activation=activation)

		# Final fully-connected layer.
		net = tf.layers.dense(inputs=net, name='layer_fc_out', units=num_actions,
							  kernel_initializer=init, activation=None)

		# The output of the Neural Network is the estimated Q-values
		# for each possible action in the game-environment.
		self.q_values = net

		# TensorFlow has a built-in loss-function for doing regression:
		# self.loss = tf.nn.l2_loss(self.q_values - self.q_values_new)
		# But it uses tf.reduce_sum() rather than tf.reduce_mean()
		# which is used by PrettyTensor. This means the scale of the
		# gradient is different and hence the hyper-parameters
		# would have to be re-tuned, because they were tuned for
		# the original version of this tutorial using PrettyTensor.
		# So instead we calculate the L2-loss similarly to how it is
		# done in PrettyTensor.
		squared_error = tf.square(self.q_values - self.q_values_new)
		sum_squared_error = tf.reduce_sum(squared_error, axis=1)
		self.loss = tf.reduce_mean(sum_squared_error)

		# Optimizer used for minimizing the loss-function.
		# Note the learning-rate is a placeholder variable so we can
		# lower the learning-rate as optimization progresses.
		self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

		# Used for saving and loading checkpoints.
		self.saver = tf.train.Saver()

		# Create a new TensorFlow session so we can run the Neural Network.
		self.session = tf.Session()

		# Load the most recent checkpoint if it exists,
		# otherwise initialize all the variables in the TensorFlow graph.
		self.load_checkpoint()

	def close(self):
		"""Close the TensorFlow session."""
		self.session.close()

	def load_checkpoint(self):
		"""
		Load all variables of the TensorFlow graph from a checkpoint.
		If the checkpoint does not exist, then initialize all variables.
		"""

		try:
			print("Trying to restore last checkpoint ...")

			# Use TensorFlow to find the latest checkpoint - if any.
			last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=checkpoint_dir)

			# Try and load the data in the checkpoint.
			self.saver.restore(self.session, save_path=last_chk_path)

			# If we get to this point, the checkpoint was successfully loaded.
			print("Restored checkpoint from:", last_chk_path)
		except:
			# If the above failed for some reason, simply
			# initialize all the variables for the TensorFlow graph.
			print("Failed to restore checkpoint from:", checkpoint_dir)
			print("Initializing variables instead.")
			self.session.run(tf.global_variables_initializer())

	def save_checkpoint(self, current_iteration):
		"""Save all variables of the TensorFlow graph to a checkpoint."""

		self.saver.save(self.session,
						save_path=self.checkpoint_path,
						global_step=current_iteration)

		print("Saved checkpoint.")

	def get_q_values(self, states):
		"""
		Calculate and return the estimated Q-values for the given states.

		A single state contains two images (or channels): The most recent
		image-frame from the game-environment, and a motion-tracing image.
		See the MotionTracer-class for details.

		The input to this function is an array of such states which allows
		for batch-processing of the states. So the input is a 4-dim
		array with shape: [batch, height, width, state_channels].

		The output of this function is an array of Q-value-arrays.
		There is a Q-value for each possible action in the game-environment.
		So the output is a 2-dim array with shape: [batch, num_actions]
		"""

		# Create a feed-dict for inputting the states to the Neural Network.
		feed_dict = {self.x: states}

		# Use TensorFlow to calculate the estimated Q-values for these states.
		values = self.session.run(self.q_values, feed_dict=feed_dict)

		return values

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

		print("Optimizing Neural Network to better estimate Q-values ...")
		print("\tLearning-rate: {0:.1e}".format(learning_rate))
		print("\tLoss-limit: {0:.3f}".format(loss_limit))
		print("\tMax epochs: {0:.1f}".format(max_epochs))

		# Prepare the probability distribution for sampling the replay-memory.
		self.replay_memory.prepare_sampling_prob(batch_size=batch_size)

		# Number of optimization iterations corresponding to one epoch.
		iterations_per_epoch = self.replay_memory.num_used / batch_size

		# Minimum number of iterations to perform.
		min_iterations = int(iterations_per_epoch * min_epochs)

		# Maximum number of iterations to perform.
		max_iterations = int(iterations_per_epoch * max_epochs)

		# Buffer for storing the loss-values of the most recent batches.
		loss_history = np.zeros(100, dtype=float)

		for i in range(max_iterations):
			# Randomly sample a batch of states and target Q-values
			# from the replay-memory. These are the Q-values that we
			# want the Neural Network to be able to estimate.
			state_batch, q_values_batch = self.replay_memory.random_batch()

			# Create a feed-dict for inputting the data to the TensorFlow graph.
			# Note that the learning-rate is also in this feed-dict.
			feed_dict = {self.x: state_batch,
						 self.q_values_new: q_values_batch,
						 self.learning_rate: learning_rate}

			# Perform one optimization step and get the loss-value.
			loss_val, _ = self.session.run([self.loss, self.optimizer],
										   feed_dict=feed_dict)

			# Shift the loss-history and assign the new value.
			# This causes the loss-history to only hold the most recent values.
			loss_history = np.roll(loss_history, 1)
			loss_history[0] = loss_val

			# Calculate the average loss for the previous batches.
			loss_mean = np.mean(loss_history)

			# Print status.
			pct_epoch = i / iterations_per_epoch
			msg = "\tIteration: {0} ({1:.2f} epoch), Batch loss: {2:.4f}, Mean loss: {3:.4f}"
			msg = msg.format(i, pct_epoch, loss_val, loss_mean)
			print_progress(msg)

			# Stop the optimization if we have performed the required number
			# of iterations and the loss-value is sufficiently low.
			if i > min_iterations and loss_mean < loss_limit:
				break

		# Print newline.
		print()

	def get_weights_variable(self, layer_name):
		"""
		Return the variable inside the TensorFlow graph for the weights
		in the layer with the given name.

		Note that the actual values of the variables are not returned,
		you must use the function get_variable_value() for that.
		"""

		# The tf.layers API uses this name for the weights in a conv-layer.
		variable_name = 'kernel'

		with tf.variable_scope(layer_name, reuse=True):
			variable = tf.get_variable(variable_name)

		return variable

	def get_variable_value(self, variable):
		"""Return the value of a variable inside the TensorFlow graph."""

		weights = self.session.run(variable)

		return weights

	def get_layer_tensor(self, layer_name):
		"""
		Return the tensor for the output of a layer.
		Note that this does not return the actual values,
		but instead returns a reference to the tensor
		inside the TensorFlow graph. Use get_tensor_value()
		to get the actual contents of the tensor.
		"""

		# The name of the last operation of a layer,
		# assuming it uses Relu as the activation-function.
		tensor_name = layer_name + "/Relu:0"

		# Get the tensor with this name.
		tensor = tf.get_default_graph().get_tensor_by_name(tensor_name)

		return tensor

	def get_tensor_value(self, tensor, state):
		"""Get the value of a tensor in the Neural Network."""

		# Create a feed-dict for inputting the state to the Neural Network.
		feed_dict = {self.x: [state]}

		# Run the TensorFlow session to calculate the value of the tensor.
		output = self.session.run(tensor, feed_dict=feed_dict)

		return output

	def get_count_states(self):
		"""
		Get the number of states that has been processed in the game-environment.
		This is not used by the TensorFlow graph. It is just a hack to save and
		reload the counter along with the checkpoint-file.
		"""
		return self.session.run(self.count_states)

	def get_count_episodes(self):
		"""
		Get the number of episodes that has been processed in the game-environment.
		"""
		return self.session.run(self.count_episodes)

	def increase_count_states(self):
		"""
		Increase the number of states that has been processed
		in the game-environment.
		"""
		return self.session.run(self.count_states_increase)

	def increase_count_episodes(self):
		"""
		Increase the number of episodes that has been processed
		in the game-environment.
		"""
		return self.session.run(self.count_episodes_increase)

