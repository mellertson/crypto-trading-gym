import unittest
from . import *
from ..agents.qlearn import *


class Test_QLearnAgent_class(unittest.TestCase):
	""" Test the QLearnAgent class. """

	@classmethod
	def setUpClass(cls):
		cls.env_name = 'QLearningAgent'
		cls.exchange = 'bitmex'
		cls.base = 'BTC'
		cls.quote = 'USD'
		cls.period_secs = 1
		cls.ob_levels = 3
		cls.base_url = 'http://0.0.0.0:8000'
		cls.agent = QLearningAgent(
			cls.env_name,
			cls.exchange,
			cls.base,
			cls.quote,
			cls.period_secs,
			cls.ob_levels,
			cls.base_url,
			rp_mem_size=120,
		)
		self.num_episodes = 60 * 24

	def setUp(self):
		super().setUp()

	def test_agent_should_have_5_primary_networs(self):
		expected = len(self.agent.env._primary_actions)
		self.assertEqual(5, expected)
		self.assertEqual(
			expected,
			len(self.agent.model.networks['primary']),
			msg=heading(
				f'\nNupic model should have {expected} primary networks, '
				f'one for each primary action.\n')
		)

	def test_agent_should_have_3_amount_networks(self):
		expected = len(self.agent.env._amount_actions)
		self.assertEqual(3, expected)
		self.assertEqual(
			expected,
			len(self.agent.model.networks['amount']),
			msg=heading(
				f'\nNupic model should have {expected} amount networks, '
				f'one for each amnount action.\n')
		)

	def test_agent_should_have_3_price_networks(self):
		expected = len(self.agent.env._price_actions)
		self.assertEqual(3, expected)
		self.assertEqual(
			expected,
			len(self.agent.model.networks['price']),
			msg=heading(
				f'\nNupic model should have {expected} price networks, '
				f'one for each amnount action.\n')
		)

	def test_run_the_agent_for_one_episode(self):
		# setup
		self.agent.is_running = True

		# test
		self.agent.run(num_episodes=60*60)

		# verify
		# TODO: verify something here
		pass


if __name__ == '__main__':
	unittest.main()










