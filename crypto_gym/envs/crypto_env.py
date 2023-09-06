import numpy as np
import pandas as pd
import requests
import gym
import json, prettyprint, copy
import collections, math, time
from datetime import datetime, timezone, timedelta
from gym import spaces
from gym.utils import seeding
from .trading_env import TradingEnv


def round_half_up(n, decimals=0):
	"""
	Round up to the nearest decimal place.

	:param n:
	:type n: float
	:param decimals:
	:type decimals: int
	:rtype: float
	"""
	multiplier = 10 ** decimals
	y = math.floor(n * multiplier + 0.5) / multiplier
	return y

def round_half_down(n, decimals=0):
	"""
	Round down to the nearest decimal place.

	:param n:
	:type n: float
	:param decimals:
	:type decimals: int
	:rtype: float
	"""
	multiplier = 10 ** decimals
	y = math.ceil(n * multiplier - 0.5) / multiplier
	return y


class CryptoEnv(gym.Env):
	"""
	An Open AI Gym environment to trade crypto-currency on an exchange.

	:ivar base_url: The URL to the Django REST API server, which is used to
		get order book and trade data.  And also to place trades with the
		upstream exchange.
		For example: 'http://localhost:8000'
	:type base_url: str
	"""
	metadata = {'render.modes': ['ascii']}

	def __init__(self, exchange, base, quote, period_secs, ob_levels, base_url,
				 *args, **kwargs):
		# ivars
		self.exchange = exchange
		self.base = base
		self.quote = quote
		self.period_secs = period_secs
		self.period_td = timedelta(seconds=period_secs)
		self.ob_levels = ob_levels
		self.base_url = base_url #: e.g. 'http://localhost:8000'
		self.current_episode = 0
		self.last_step_dt = datetime.now()
		self.observation = None
		self.exchange_rate = 0.0
		# define the action space
		self._primary_actions = [
			'hodl',
			'market_sell',
			'market_buy',
			'limit_sell',
			'limit_buy',
			# TODO: add liquidate to Django REST API for 'liquidate' action.
			'liquidate',
		]
		self._amount_actions = collections.OrderedDict()
		self._amount_actions['amount_level_1'] = 0.002 #: meaning 0.5% of account.free_balance.
		self._amount_actions['amount_level_2'] = 0.004 #: meaning 0.7% of account.free_balance.
		self._amount_actions['amount_level_3'] = 0.006 #: meaning 0.9% of account.free_balance.
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
		self._order_book_length = ob_levels * 4 #: buy & sell prices and buy & sell amounts per ob level
		self._trade_length = 4 #: buy & sell prices and buy & sell amounts
		self._account_bal_length = 3 #: total, used, & free balances
		self._position_bal_length = 6 #: total, used, & free balances
		self.shape = tuple([len(self.get_input_field_names())])
		self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.shape)
		self.order_book_df = None
		self.trade_df = None
		self.account_bal_df = None
		self.position_bal_df = None
		self.last_total_balance = None
		self.orders = []
		self.last_observation = []
		self.last_action_order_type = None
		self.last_action_price = None
		self.last_action_amount = None
		self.last_action_was_executed = True

	def len_actionable_game_space(self):
		"""
		Gets the length of observation space which the agent should use to
		make decisions.

		For Bitmex Tesnet the "actionable game space" includes the order book,
		and trade history.  But, does not include account balance information,
		nor position information.  Account balance and position information is
		not included in the "actionable game space" to avoid agent induced
		feedback loops, e.g. when the agent places a market order.

		:rtype: int
		"""
		return len(self.get_input_field_names()) \
			   - len(self.get_account_field_names()) \
			   - len(self.get_position_field_names())

	def get_order_book_field_names(self):
		"""
		Get order book field names

		:rtype: dict
		"""
		order_book_data = {}
		for i in range(self.ob_levels):
			order_book_data[f'order_book_ask_price_lvl_{i}'] = ['101.0']
			order_book_data[f'order_book_ask_amount_lvl_{i}'] = ['0.101']
			order_book_data[f'order_book_bid_price_lvl_{i}'] = ['99.0']
			order_book_data[f'order_book_bid_amount_lvl_{i}'] = ['0.99']
		return order_book_data

	def get_trade_field_names(self):
		"""
		Get trade field names

		:rtype: dict
		"""
		return {
			'trade_sell_price': '20000.0000000000',
			'trade_sell_amount': '100.0000000000',
			'trade_buy_price': '21000.0000000000',
			'trade_buy_amount': '200.0000000000',
		}

	def get_position_field_names(self):
		"""
		Get position field names

		:rtype: dict
		"""
		return {
			'position_status': '0',
			'position_side': '0',
			'position_entry_price': '0',
			'position_break_even_price': '0',
			'position_liquidation_price': '0',
			'position_current_amount': '0',
		}

	def get_account_field_names(self):
		"""
		Get account field names

		:rtype: dict
		"""
		return {
			'total_balance': '<str:account.total_balance>',
			'used_balance': '<str:account.used_balance>',
			'free_balance': '<str:account.free_balance>',
		}

	def get_input_field_names(self):
		"""
		Return a list of field names from the data source.

		# MEDIUM: build input field names dynamically from the Django JSON data.

		:rtype: list of str
		"""
		fields = list(self.get_order_book_field_names().keys())
		fields.extend(self.get_trade_field_names().keys())
		fields.extend(self.get_account_field_names().keys())
		fields.extend(self.get_position_field_names().keys())
		# HIGH: add 2 fields for Twitter.user & sentiment_score
		return fields

	def build_action_names(self):
		primary_action_names = copy.deepcopy(self._primary_actions)
		amount_action_names = []
		for amount_action_name in self._amount_actions.keys():
			amount_action_names.append(amount_action_name)
		price_action_names = []
		for price_action_name in self._price_actions:
			price_action_names.append(price_action_name)
		action_names = collections.OrderedDict()
		action_names['primary'] = primary_action_names
		action_names['amount'] = amount_action_names
		action_names['price'] = price_action_names
		return action_names

	def seed(self, seed=None):
		return seeding.np_random(seed)

	def fetch_order_book_data(self):
		""" Get order book data via HTTP GET request and cache locally.

		:rtype: pandas.DataFrame
		"""
		end_date = self.last_step_dt + self.period_td
		url = f'{self.base_url}/api/market_data/order_book/' \
			f'{self.exchange}/' \
			f'{self.base}/{self.quote}/{self.ob_levels}/'
		r = requests.get(url)
		while r.status_code != 200:
			print(f'ERROR: {r.status_code} by GET: {url}, sleeping 1 second...')
			time.sleep(1)
			r = requests.get(url)
		order_book = json.loads(r.content.decode('utf-8'))
		self.order_book_df = pd.DataFrame(
			data=order_book,
			index=[end_date],
		)
		# print(f'order_book = {order_book}')
		self.exchange_rate = float(order_book['order_book_ask_price_lvl_1'][0])
		return self.order_book_df

	def fetch_trade_data(self):
		"""
		Get trade data via HTTP GET request and cache locally.

		:rtype: pandas.DataFrame
		"""
		end_date = self.last_step_dt + self.period_td
		url = f'{self.base_url}/api/market_data/trade/' \
			f'{self.exchange}/' \
			f'{self.base}/{self.quote}/' \
			f'{self.last_step_dt.isoformat()}/{end_date.isoformat()}/'
		r = requests.get(url)
		while r.status_code != 200:
			print(f'ERROR: {r.status_code} by GET from: {url} sleeping 1 second...')
			time.sleep(1)
			r = requests.get(url)
		trades = json.loads(r.content.decode('utf-8'))
		self.trade_df = pd.DataFrame(
			data=trades,
			index=[end_date],
		)
		return self.trade_df

	def fetch_account_balance_data(self):
		"""
		Get account balance data via HTTP GET request and cache locally.

		:rtype: pandas.DataFrame
		"""
		end_date = self.last_step_dt + self.period_td
		url = f'{self.base_url}/api/account/balance/' \
			f'{self.exchange}/' \
			f'{self.base}/'
		r = requests.get(url)
		while r.status_code != 200:
			print(f'ERROR: {r.status_code} by GET from: {url} sleeping 1 second...')
			time.sleep(1)
			r = requests.get(url)
		account_balance = json.loads(r.content.decode('utf-8'))
		self.account_bal_df = pd.DataFrame(
			data=account_balance,
			index=[end_date],
		)
		return self.account_bal_df

	def fetch_position_balance_data(self):
		"""
		Get position balance data via HTTP GET request and cache locally.

		:rtype: pandas.DataFrame
		"""
		end_date = self.last_step_dt + self.period_td
		url = f'{self.base_url}/api/position/algo_balance/' \
			f'{self.exchange}/' \
			f'{self.base}/' \
			f'{self.quote}/'
		r = requests.get(url)
		while r.status_code != 200:
			print(f'ERROR: {r.status_code} by GET from: {url} sleeping 1 second...')
			time.sleep(1)
			r = requests.get(url)
		position_balance = json.loads(r.content.decode('utf-8'))
		self.position_bal_df = pd.DataFrame(
			data=position_balance,
			index=[end_date],
		)
		return self.position_bal_df

	def get_next_observation(self):
		"""
		Get the next observation from the REST API server and cache locally.

		:rtype: tuple of float
		"""
		# update the last total balance so the reward is calculated correctly.
		self.last_total_balance = self.account_balance_total

		# get all "data sources" from the REST API server and cache locally.
		order_book = self.fetch_order_book_data()
		trade = self.fetch_trade_data()
		account_balance = self.fetch_account_balance_data()
		position_balance = self.fetch_position_balance_data()
		# HIGH: get Twitter sentiment data and include it as a data source.

		# concatenate dataframes together
		observation = list(order_book.iloc[0])
		observation.extend(list(trade.iloc[0]))
		observation.extend(list(account_balance.iloc[0]))
		observation.extend(list(position_balance.iloc[0]))
		# print(f'shape of observation = {np.shape(observation)}')
		# print(f'shape of self.observation_space.shape = {self.observation_space.shape}')
		assert (np.shape(observation) == self.observation_space.shape)

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
		action_name = self._primary_actions[action]
		if action_name == 'market_sell':
			return ['market', 'sell']
		elif action_name == 'market_buy':
			return ['market', 'buy']
		elif action_name == 'limit_sell':
			return ['limit', 'sell']
		elif action_name == 'limit_buy':
			return ['limit', 'buy']
		elif action_name == 'liquidate':
			return ['liquidate', None]
		else:
			return [None, None]

	def translate_amount_action(self, action_index):
		"""
		Translate an amount action index into an order amount.

		:param action_index:
		:type action_index: int
		:return: The order amount in USD to place on the exchange.
		:rtype: int
		"""
		action_key = f'amount_level_{action_index + 1}'
		amount_pct = self._amount_actions[action_key]
		amount = int(self.account_balance_free * amount_pct * self.exchange_rate)
		return amount

	def translate_price_action(self, action, side):
		"""
		Translate a price action index into an order price.

		:param action:
		:type action: int
		:param side: Either 'buy' or 'sell'.
		:type side: str
		:return: The order price to use when placing the order on the exchange.
		:rtype: int
		"""
		# increment because action is zero indexed.
		price_level = action
		ob_side = 'ask' if side == 'sell' else 'bid'
		col_name = f'order_book_{ob_side}_price_lvl_{price_level}'
		if col_name in self.order_book_df:
			price = float(self.order_book_df[col_name][0])
		else:
			price = 0
		# frac = int(f'{price:.1f}'[-1])
		# if frac >= 5:
		# 	frac = 5
		# else:
		# 	frac = 0
		# price = float(f'{int(price)}.{frac}')
		return int(price)

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
		amount = amount / 1000.0
		amount = round_half_down(amount, decimals=1)
		amount = int(amount * 1000.0)
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
		return self.account_balance_total - self.last_total_balance

	def execute_action_close_open_position(self):
		"""
		Send PUT request to close the open position.

		:rtype: None
		"""
		url = f'{self.base_url}/api/position/close/' \
			f'{self.exchange}/' \
			f'{self.base}/' \
			f'{self.quote}/'
		r = requests.put(url)
		if r.status_code != 200:
			print(f'ERROR: {r.status_code} during PUT to: {url} sleeping 1 second...')
			time.sleep(1.0)

	def execute_action(self, actions):
		"""
		Execute the given actions.

		:param actions: The actions to execute in this environment. The actions
			for this environment are:
			(primary_action, amount_action, price_action)
		:type actions: list of int

		:returns:
			True - if the last action was executed.
			False - if the last action was not executed.
		:rtype: bool
		"""
		order_type, side = self.translate_primary_action(actions[0])
		if self.last_action_order_type == order_type:
			return False
		if order_type == 'liquidate':
			# close the open position.
			self.execute_action_close_open_position()
			self.last_action_order_type = order_type
			return True
		elif order_type is None:
			return False

		amount = self.translate_amount_action(actions[1])
		price = self.translate_price_action(actions[2], side)
		if self.last_action_amount == amount:
			return False
		if self.last_action_price == price:
			return False
		self.last_action_order_type = order_type
		self.last_action_amount = amount
		self.last_action_price = price
		url, payload = self.build_order_url_and_payload(
			order_type,
			side,
			price,
			amount
		)
		if price > 0 and amount > 0:
			r = requests.post(url, json=payload)
			if r.status_code != 201:
				print(f'ERROR: {r.status_code} recieved by GET from: {url}')
			else:
				order = json.loads(r.content.decode('utf-8'))
				print(f'Order Placed: {order}')
				self.orders.append(order)
			return True
		else:
			return False

	def is_observation_space_changed(self, obs):
		"""
		Test if the current observation is different than the last one

		:param obs: The current observation
		:type obs: tuple | list
		:rtype: bool
		"""
		if self.last_observation == obs:
			return False
		else:
			self.last_observation = obs
			return True

	def step(self, actions, observation=None):
		"""
		Execute action in the environment and return the next state.

		Accepts an action and returns a tuple (observation, reward, done, info).

		:param actions: The actions to execute in this environment. The actions
			for this environment are:
			(primary_action, acmount_action, price_action)
		:type actions: list of int
		:param observation: If observation is not None a new observation
			will be gotten and returned.
			If observation is None, this observation kwarg will be returned.
		:type observation: list | tuple | None

		Returns:
			observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further
            	step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful
            	for debugging, and sometimes learning)
		"""
		self.last_action_was_executed = self.execute_action(actions)
		if observation is None:
			observation = self.get_next_observation()
		reward = self.reward
		done = self.done
		info = self.current_info
		return observation, reward, done, info

	@property
	def current_info(self):
		return {
			'current_episode': self.current_episode,
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
			'orders': self.orders,
			'reward': f'{self.reward:,.10f}',
		}

	def render(self, mode='ascii'):
		if mode == 'ascii':
			rendered = prettyprint.pformat(self.current_info, indent=4, width=120)
			return rendered
		else:
			super().render(mode=mode)





