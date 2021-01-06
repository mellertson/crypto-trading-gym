import unittest
from . import *
from ..agents.qlearn import *


class ActionFlattener_base_class(unittest.TestCase):
	""" Base class to test the ActionFlattener class. """
	class_name = 'ActionFlattener'

	def setUp(self):
		super().setUp()
		# the lengths of each of the inner tuples of actions.
		self.lengths = [5, 3, 3]
		# create action structure like the one used in QLearningAgent v1.0.
		self.actions = ((0,1,2,3,4), (0,1,2), (0,1,2))
		# the instance under test.
		self.instance = ActionFlattener(self.actions)


class ActionFlattener__init__method(ActionFlattener_base_class):

	def test_actions_ivar_should_be_flattened(self):
		# setup
		expected = [0, 1, 2, 3, 4, 0, 1, 2, 0, 1, 2]

		# verify: length of ActionFlattener.actions
		self.assertEqual(
			len(expected),
			len(self.instance.actions),
			msg=heading(f'length of {self.class_name} should be {len(expected)}')
		)

		# verify: structure of actions ivar
		for i in range(len(expected)):
			self.assertEqual(
				expected[i],
				self.instance.actions[i],
				msg=heading(f'Expected: {expected}\b\bBut, got: {self.instance.actions}')
			)

	def test_actions_arg_should_be_stored_in_original_actions_ivar(self):
		self.assertEqual(self.actions, self.instance.original_actions)



if __name__ == '__main__':
	unittest.main()
