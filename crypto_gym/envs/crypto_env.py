import numpy as np
import pandas as pd
import requests
import gym
import json
from datetime import datetime, timezone, timedelta
from gym import spaces
from gym.utils import seeding
from .trading_env import TradingEnv

# MEDIUM: create step() method
# MEDIUM: create reset() method
# MEDIUM: create render() method
# MEDIUM: create close() method


class CryptoEnv(gym.Env):
	""" An Open AI Gym environment to trade crypto-currency on an exchange. """

	def __init__(self, exchange, base, quote, period_secs, ob_levels,
				 window_size, base_url, *args, **kwargs):
		# ivars
		self.exchange = exchange
		self.base = base
		self.quote = quote
		self.period_secs = period_secs
		self.period_td = timedelta(seconds=period_secs)
		self.ob_levels = ob_levels
		self.window_size = window_size
		self.base_url = base_url #: e.g. 'http://localhost:8000'
		self.last_step_dt = None
		self.df = None
		# define the action space
		self._primary_actions = [
			'HODL',
			'market_sell',
			'market_buy',
			'limit_sell',
			'limit_buy',
			'liquidate',
		]
		self._amount_actions = [
			'amount_level_1',
			'amount_level_2',
			'amount_level_3',
		]
		self._price_actions = [
			'price_level_1',
			'price_level_2',
			'price_level_3',
		]
		self.action_space = spaces.MultiDiscrete([
			len(self._primary_actions),
			len(self._amount_actions),
			len(self._price_actions),
		])
		# define the observation space
		self._order_book_length = ob_levels * 4 #: buy & sell price & amount per ob level
		self._trade_length = 4 #: buy & sell price & amount
		self._account_bal_length = 3 #: total, used, & free balances
		self.shape = (
			self._order_book_length +
			self._trade_length +
			self._account_bal_length)
		self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.shape)

	def seed(self, seed=None):
		return seeding.np_random(seed)

	def reset(self):
		""" Initialize state by getting data via HTTP GET request to Django. """
		order_book = self.get_order_book_data()
		trades = self.get_trade_data()
		account_balance = self.get_account_balance_data()
		self.df = pd.concat([order_book, trades, account_balance], axis=1)

	def get_order_book_data(self):
		""" Get order book data via HTTP GET request.

		:rtype: pandas.DataFrame
		"""
		r = requests.get(
			f'{self.base_url}/api/market_data/order_book/'
			f'{self.exchange}/'
			f'{self.base}/{self.quote}/{self.ob_levels}/')
		order_book = json.loads(r.content.decode('utf-8'))
		order_book = pd.DataFrame(data=order_book)
		return order_book

	def get_trade_data(self):
		"""
		Get trade data via HTTP GET request.

		:rtype: pandas.DataFrame
		"""
		self.last_step_dt = datetime.now(tz=timezone.utc)
		end_date = self.last_step_dt + self.period_td
		r = requests.get(
			f'{self.base_url}/api/market_data/trade/'
			f'{self.exchange}/'
			f'{self.base}/{self.quote}/'
			f'{self.last_step_dt.isoformat()}/{end_date.isoformat()}/'
		)
		trades = json.loads(r.content.decode('utf-8'))
		trades = pd.DataFrame(data=trades)
		return trades

	def get_account_balance_data(self):
		"""
		Get account balance data via HTTP GET request.

		:rtype: pandas.DataFrame
		"""
		self.last_step_dt = datetime.now(tz=timezone.utc)
		end_date = self.last_step_dt + self.period_td
		r = requests.get(
			f'{self.base_url}/api/account/balance/'
			f'{self.exchange}/'
			f'{self.base}/{self.quote}/'
			f'{self.last_step_dt.isoformat()}/{end_date.isoformat()}/'
		)
		account_balance = json.loads(r.content.decode('utf-8'))
		account_balance = pd.DataFrame(data=account_balance)
		return account_balance





