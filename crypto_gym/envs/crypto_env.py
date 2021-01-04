import numpy as np
import pandas as pd
import requests
import gym
import json, prettyprint, copy
import collections
from datetime import datetime, timezone, timedelta
from gym import spaces
from gym.utils import seeding
from .trading_env import TradingEnv


class CryptoEnv(gym.Env):
	""" An Open AI Gym environment to trade crypto-currency on an exchange. """
	metadata = {'render.modes': ['ascii']}

	def __init__(self, exchange, base, quote, period_secs, ob_levels,
				 window_size, base_url, max_episodes, *args, **kwargs):
		# ivars
		self.exchange = exchange
		self.base = base
		self.quote = quote
		self.period_secs = period_secs
		self.period_td = timedelta(seconds=period_secs)
		self.ob_levels = ob_levels
		self.window_size = window_size
		self.base_url = base_url #: e.g. 'http://localhost:8000'
		self.max_episodes = max_episodes
		self.current_episode = 0
		self.last_step_dt = None
		self.observation = None
		# define the action space
		self._primary_actions = [
			'HODL',
			'market_sell',
			'market_buy',
			'limit_sell',
			'limit_buy',
			# TODO: add liquidate to Django REST API for 'liquidate' action.
			# 'liquidate',
		]
		self._amount_actions = collections.OrderedDict()
		self._amount_actions['amount_level_1'] = 0.001 #: meaning 0.1% of account.free_balance.
		self._amount_actions['amount_level_2'] = 0.002 #: meaning 0.2% of account.free_balance.
		self._amount_actions['amount_level_3'] = 0.003 #: meaning 0.3% of account.free_balance.
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
		self.action_names = self.build_action_names()
		# define the observation space
		self._order_book_length = ob_levels * 4 #: buy & sell price & amount per ob level
		self._trade_length = 4 #: buy & sell price & amount
		self._account_bal_length = 3 #: total, used, & free balances
		self.shape = (
			self._order_book_length +
			self._trade_length +
			self._account_bal_length)
		self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.shape)
		self.order_book_df = None
		self.trade_df = None
		self.account_bal_df = None
		self.last_total_balance = None
		self.orders = []

	def build_action_names(self):
		primary_action_names = copy.deepcopy(self._primary_actions)
		amount_action_names = []
		for amount_action_name in self._amount_actions.keys():
			amount_action_names.append(amount_action_name)
		price_action_names = []
		for price_action_name in self._price_actions:
			price_action_names.append(price_action_name)
		action_names = {
			'primary': primary_action_names,
			'amount': amount_action_names,
			'price': price_action_names,
		}
		return action_names

	def seed(self, seed=None):
		return seeding.np_random(seed)

	def fetch_order_book_data(self):
		""" Get order book data via HTTP GET request and cache locally.

		:rtype: pandas.DataFrame
		"""
		r = requests.get(
			f'{self.base_url}/api/market_data/order_book/'
			f'{self.exchange}/'
			f'{self.base}/{self.quote}/{self.ob_levels}/')
		order_book = json.loads(r.content.decode('utf-8'))
		self.order_book_df = pd.DataFrame(data=order_book)
		return self.order_book_df

	def fetch_trade_data(self):
		"""
		Get trade data via HTTP GET request and cache locally.

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
		self.trade_df = pd.DataFrame(data=trades)
		return self.trade_df

	def fetch_account_balance_data(self):
		"""
		Get account balance data via HTTP GET request and cache locally.

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
		self.account_bal_df = pd.DataFrame(data=account_balance)
		return self.account_bal_df

	def get_next_observation(self):
		"""
		Get the next observation from the REST API server and cache locally.

		:rtype: tuple of float
		"""
		# update the last total balance so the reward is calculated correctly.
		self.last_total_balance = self.account_balance_total
		# get all "data sources" from the REST API server and cache locally.
		order_book = self.fetch_order_book_data()
		trades = self.fetch_trade_data()
		account_balance = self.fetch_account_balance_data()
		# concatenate dataframes together
		observation = pd.concat([order_book, trades, account_balance], axis=1)
		# slice out 1st row & convert to float data types
		observation = pd.to_numeric(observation.iloc[0])
		assert (observation.space == self.observation_space.space)
		return observation

	def reset(self):
		""" Initialize state by getting data via HTTP GET request to Django. """
		self.current_episode = 0
		self.observation = self.get_next_observation()
		return self.observation

	def translate_primary_action(self, action):
		"""
		Translate a primary action into an order type and side.

		:param action:
		:type action: int
		:returns: Formatted like this: [Order.type, Order.side]
		:rtype: list of str
		"""
		action_name = self._price_actions.index(action)
		if action_name == 'market_sell':
			return ['market', 'sell']
		elif action_name == 'market_buy':
			return ['market', 'buy']
		elif action_name == 'limit_sell':
			return ['limit', 'sell']
		elif action_name == 'limit_buy':
			return ['limit', 'buy']
		else:
			return [None, None]

	def translate_amount_action(self, action):
		"""
		Translate an amount action index into an order amount.

		:param action:
		:type action: int
		:return: The order amount to use when placing the order on the exchange.
		:rtype: float
		"""
		amount_pct = self._amount_actions[action]
		amount = self.account_balance_free * amount_pct
		return amount

	def translate_price_action(self, action, side):
		"""
		Translate a price action index into an order price.

		:param action:
		:type action: int
		:param side: Either 'buy' or 'sell'.
		:type side: str
		:return: The order price to use when placing the order on the exchange.
		:rtype: float
		"""
		# increment because action is zero indexed.
		price_level = action + 1
		ob_side = 'ask' if side == 'sell' else 'bid'
		col_name = f'order_book_{ob_side}_price_lvl_{price_level}'
		price = float(self.order_book_df[col_name][0])
		return price

	@property
	def account_balance_free(self):
		if self.account_bal_df is None:
			return 0.0
		elif 'free_balance' not in self.account_bal_df.columns:
			return 0.0
		else:
			return float(self.account_bal_df['free_balance'][0])

	@property
	def account_balance_total(self):
		if self.account_bal_df is None:
			return 0.0
		elif 'total_balance' not in self.account_bal_df.columns:
			return 0.0
		else:
			return float(self.account_bal_df['total_balance'][0])

	def build_order_url_and_payload(self, type, side, price, amount):
		"""
		Build the URL and payload to place an order to the Django REST API.

		:param type: Either: 'market' | 'limit'
		:type type: str
		:param side: Either: 'sell' | 'buy'
		:type side: str
		:param price:
		:type price: float
		:param amount:
		:type amount: float
		:return: The URL to POST the payload to.
		:rtype: tuple
		"""
		url = f'{self.base_url}/api/order/'
		payload = {
			'exchange_id': self.exchange,
			'market': f'{self.base}/{self.quote}',
			'type': type,
			'side': side,
			'price': price,
			'amount': amount,
		}
		return url, payload

	@property
	def done(self):
		"""
		Tests if the game has been "won" or "lost".

		:returns: True - if the agent is able to place new trades.
			False - if the account balance is zero AND there are no open orders.
		:rtype: bool
		"""
		return False

	@property
	def reward(self):
		"""
		Calculate the current "reward" value.

		:rtype: float
		"""
		return self.last_total_balance - self.account_balance_total

	def execute_action(self, action):
		"""
		Execute the given action.

		:param action:
		:return:
		"""
		type, side = self.translate_primary_action(action[0])
		if type is not None:
			amount = self.translate_amount_action(action[1])
			price = self.translate_price_action(action[3])
			url, payload = self.build_place_order_url()
			r = requests.post(url, json=payload)
			order = json.loads(r.content.decode('utf-8'))
			self.orders.append(order)

	def step(self, action):
		"""
		Execute action in the environment and return the next state.

		Accepts an action and returns a tuple (observation, reward, done, info).

		:param action: The action to execute in this environment. The action
			for this environment are:
			(primary_action, acmount_action, price_action)
		:type action: list of int

		Returns:
			observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further
            	step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful
            	for debugging, and sometimes learning)
		"""
		self.execute_action(action)
		observation = self.get_next_observation()
		reward = self.reward
		done = self.done
		info = self.current_info
		return observation, reward, done, info

	@property
	def current_info(self):
		return {
			'current_episode': self.current_episode,
			'max_episodes': self.max_episodes,
			'order_book': self.order_book_df,
			'trade': self.trade_df,
			'account_balance': self.account_bal_df,
			'base_url': self.base_url,
			'exchange': self.exchange,
			'base': self.base,
			'quote': self.quote,
			'period_secs': self.period_secs,
			'period_td': self.period_td,
			'ob_levels': self.ob_levels,
			'window_size': self.window_size,
			'orders': self.orders,
			'reward': f'{self.reward:,.10f}',
		}

	def render(self, mode='ascii'):
		if mode == 'ascii':
			rendered = prettyprint.pformat(self.current_info, indent=4, width=120)
			return rendered
		else:
			super().render(mode=mode)

