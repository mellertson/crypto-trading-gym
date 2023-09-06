import unittest
import pandas as pd
from datetime import datetime, timedelta
from . import *
from ..agents.qlearn import *
from ..envs.crypto_env import CryptoEnv
from ..models import set_predictor_server_base_url


class Test_QLearnAgent_class(unittest.TestCase):
	""" Test the QLearnAgent class. """

	@classmethod
	def setUpClass(cls):
		cls.env_name = 'QLearningAgent'
		cls.exchange = 'bitmex'
		cls.base = 'BTC'
		cls.quote = 'USD'
		cls.period_secs = 2
		cls.ob_levels = 3
		# Nupic Predictor REST API URL
		cls.nupic_predictor_url = set_predictor_server_base_url(
			'http://localhost:5000',
		)

		# Django REST API URL
		cls.base_url = 'http://0.0.0.0:8000'
		cls.agent = QLearningAgent(
			cls.env_name,
			cls.exchange,
			cls.base,
			cls.quote,
			cls.period_secs,
			cls.ob_levels,
			cls.base_url,
			rp_mem_size=500,
		)
		cls.num_episodes = 60 * 24

	def setUp(self):
		super().setUp()

	def test_agent_should_have_5_primary_networks(self):
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
		self.agent.run(num_episodes=None)

		# verify
		# TODO: verify something here
		pass


class Test_CrytoEnvironment_class(unittest.TestCase):

	def setUp(self):
		super().setUp()
		self.env_name = 'QLearningAgent'
		self.exchange = 'bitmex'
		self.base = 'BTC'
		self.quote = 'USD'
		self.period_secs = 60 * 30
		self.ob_levels = 3
		self.base_url = 'http://0.0.0.0:8000'
		self.env = CryptoEnv(
			self.exchange,
			self.base,
			self.quote,
			self.period_secs,
			self.ob_levels,
			self.base_url,
		)
		self.env.last_step_dt = datetime.now() - self.env.period_td

	def test_fetch_trade_data(self):
		""" Test GET trade data from Django REST API. """
		# test
		trade_df = self.env.fetch_trade_data()

		# verify
		self.assertIsInstance(trade_df, pd.DataFrame)

	def test_fetch_account_data(self):
		""" Test GET account balance data from Django REST API. """
		# test
		account_bal_df = self.env.fetch_account_balance_data()

		# verify
		self.assertIsInstance(account_bal_df, pd.DataFrame)

	def test_fetch_order_book_data(self):
		""" Test GET order_book data from Django REST API. """
		# test
		order_book_df = self.env.fetch_order_book_data()

		# verify
		self.assertIsInstance(order_book_df, pd.DataFrame)


if __name__ == '__main__':
	unittest.main()









