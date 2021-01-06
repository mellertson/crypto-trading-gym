import unittest
from . import *
from ..agents.qlearn import *


class Test_QLearnAgent_class(unittest.TestCase):
	""" Test the QLearnAgent class. """

	def setUp(self):
		super().setUp()
		self.env_name = 'QLearningAgent'
		self.exchange = 'bitmex'
		self.base = 'BTC'
		self.quote = 'USD'
		self.period_secs = 15
		self.ob_levels = 3
		self.base_url = 'http://0.0.0.0:8000'
		self.agent = None

	def test__init__(self):
		self.agent = QLearningAgent(
			self.env_name,
			self.exchange,
			self.base,
			self.quote,
			self.period_secs,
			self.ob_levels,
			self.base_url,
		)

	def test_run_the_agent_for_one_episode(self):
		# setup
		self.agent.is_running = True

		# test
		# self.agent.run(num_episodes=1)

		# verify
		# TODO: verify something here
		pass


if __name__ == '__main__':
	unittest.main()

