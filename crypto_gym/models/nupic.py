from random import uniform
import numpy as np
import copy, os, requests, yaml, json
import collections
from datetime import datetime, timedelta, timedelta
from crypto_gym.models import *


__all__ = [
	'NUPIC_MODELS_DIR',
	'NupicNetwork',
	'NupicModel',
]


NUPIC_MODELS_DIR = os.path.join(
	os.path.dirname(__file__),
	'nupic_model_files',
)


class NupicNetwork(object):
	"""
	Interfaces with the prediction server via HTTP messages.

	Multiple input fields can be specified, but ONLY ONE output field
	can be predicted per nupic network.
	"""

	def __init__(self, exchange, base, quote, input_fields, predicted_field, timeframe, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.exchange = exchange
		self.base = base
		self.quote = quote
		self.input_fields = input_fields
		self.predicted_field = predicted_field
		self.timeframe = timeframe
		self.predictor_id = None
		self.last_prediction_msg = {}
		self.last_prediction = None
		self.model_was_optimized = False
		self.model_template_filename = os.path.join(
			NUPIC_MODELS_DIR,
			'nupic-model-v1.yaml',
		)
		self.model = None
		self.build_model_from_tempalte()
		self.send_message_new_predictor()
		self.send_message_start_predictor()

	def __str__(self):
		return f'{self.__class__.__name__}({self.predictor_id})'

	def __del__(self):
		self.send_message_stop_predictor()

	def load_model_from_template(self):
		with open(self.model_template_filename, 'r') as f:
			model_template = yaml.load(f)
		return model_template

	def build_model_from_tempalte(self):
		model = self.load_model_from_template()

		# build and add encoders for the data source (inputs) into the model
		encoders = {}
		for field_name in self.input_fields:
			encoder = self._build_random_distributed_scalar_encoder(field_name)
			encoders[field_name] = encoder
		# build and add encoder for the predicted field (output as input)
		encoder = self._build_category_encoder(self.predicted_field)
		encoders[self.predicted_field] = encoder
		# build and add encoders for the datetime field (timestamp inputs)
		timestamp_encoders = self._build_timestamp_encoders('timestamp')
		encoders.update(timestamp_encoders)
		model['modelParams']['sensorParams']['encoders'] = encoders

		# build and add the classifier (output) into the model
		classifiers = {}
		classifier = self._build_sdr_classifier_region(self.predicted_field)
		classifiers[self.predicted_field] = classifier
		model['modelParams']['classifiers'] = classifiers
		self.model = model

	@property
	def url_new_predictor(self):
		"""
		Return a fully-qualified URL to create a new predictor via POST request.
		:rtype: str
		"""
		url = f'{PREDICTOR_SERVER_BASE_URL}/new/predictor/'
		return url

	@property
	def url_start_predictor(self):
		return f'{PREDICTOR_SERVER_BASE_URL}/start/predictor/{self.predictor_id}/'

	@property
	def url_predict_with_learning_off(self):
		return f'{PREDICTOR_SERVER_BASE_URL}/predict/{self.predictor_id}/false/'

	@property
	def url_predict_with_learning_on(self):
		return f'{PREDICTOR_SERVER_BASE_URL}/predict/{self.predictor_id}/true/'

	@property
	def url_save_predictor(self):
		return f'{PREDICTOR_SERVER_BASE_URL}/save/predictor/{self.predictor_id}/'

	def url_stop_predictor(self):
		return f'{PREDICTOR_SERVER_BASE_URL}/stop/predictor/{self.predictor_id}/'

	@classmethod
	def line_1(cls, x, y, z):
		line = [x]
		line.extend(y)
		line.append(z)
		line = ', '.join(line)
		return line

	@classmethod
	def line_2(cls, x, y, z=None):
		line = ''.join([x] + [', float'] * (len(y) + 1))
		return line

	@classmethod
	def line_3(cls, x, y, z=None):
		return ''.join([x] + [','] * (len(y) + 1))

	@classmethod
	def _build_random_distributed_scalar_encoder(cls, field_name, resolution=0.88, seed=1,):
		encoder = {
			'type': 'RandomDistributedScalarEncoder',
			'fieldname': field_name,
			'name': field_name,
			'resolution': resolution,
			'seed': seed,
		}
		return encoder

	@classmethod
	def _build_category_encoder(cls, field_name, w=21, category_list='1,2,3,4,5',):
		encoder = {
			'type': 'CategoryEncoder',
			'fieldname': field_name,
			'name': field_name,
			'w': w,
			'categoryList': category_list,
		}
		return encoder

	@classmethod
	def _build_timestamp_encoders(cls, field_name):
		encoders = {
			'time_of_day': {
				'type': 'DateEncoder',
				'fieldname': field_name,
				'name': 'time_of_day',
				'timeOfDay': [21, 1],
			},
			'weekend': {
				'type': 'DateEncoder',
				'fieldname': field_name,
				'name': 'weekend',
				'weekend': 21
			},
			'season': {
				'type': 'DateEncoder',
				'fieldname': field_name,
				'name': 'season',
				'season': 21,
			},
		}
		return encoders

	@classmethod
	def _build_sdr_classifier_region(cls, predicted_field, max_category_count=1000, steps='1', alpha=0.1, verbosity=1):
		classifier = {
			'regionType': 'SDRClassifierRegion',
			'verbosity': verbosity,
			'alpha': alpha,
			'steps': steps,
			'maxCategoryCount': max_category_count,
			'implementation': 'cpp',
		}
		return classifier

	def _construct_predict_payload(self, timestamp, observation, action):
		"""

		:type timestamp: str | datetime.datetime
		:param observation: The input data.
			For example: (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
		:type observation: tuple of float
		:param action: A specific action, we want the network to predict.
		:type action: float
		:rtype: str
		"""

		if isinstance(timestamp, datetime):
			cycle_timestamp = datetime(
				timestamp.year,
				timestamp.month,
				timestamp.day,
				timestamp.hour,
				timestamp.minute,
				timestamp.second,
			)
			line = cycle_timestamp.isoformat()
		else:
			line = str(timestamp)
		for x in observation:
			line += ', ' + str(x)
		line += ', ' + str(action)
		return line

	def send_message_new_predictor(self):
		payload = {
			'model': self.model,
			'exchange': self.exchange.lower(),
			'market': f'{self.base.lower()}{self.quote.lower()}',
			'predicted_field': self.predicted_field.lower(),
			'timeframe': self.timeframe.lower(),
		}
		print(f'payload = {payload}')
		r = requests.post(
			self.url_new_predictor,
			data=json.dumps(payload),
			headers={'Content-type': 'application/json', 'Accept': 'text/plain'},
		)
		if r.status_code != 201:
			raise RuntimeError(
				f'POST to {self.url_new_predictor} '
				f'returned: {r.status_code}\n\n{r.text}')
		else:
			self.predictor_id = json.loads(r.text)['predictor']['id']
			print(f'Prediction server created: {self}')

	def send_message_start_predictor(self):
		payload = [
			self.line_1('timestamp', self.input_fields, self.predicted_field),
			self.line_2('datetime', self.input_fields, self.predicted_field),
			self.line_3('T', self.input_fields, self.predicted_field),
		]
		r = requests.post(
			self.url_start_predictor,
			data=json.dumps(payload),
			headers={'Content-type': 'application/json', 'Accept': 'text/plain'},
		)
		if r.status_code != 200:
			raise RuntimeError(
				f'POST to {self.url_new_predictor} '
				f'returned: {r.status_code}\n\n{r.text}')
		else:
			print(f'Prediction server started: {self}')

	def send_message_predict_with_learning_off(self, timestamp, observation, action):
		"""
		Make a prediction with learning turned off.

		:param timestamp:
		:param observation: Input data as a "flat" tuple of floats.
		:type observation: tuple of float
		:param action: A specific action, we want the network to predict.
		:type action: float
		:return: A single predicted value.  But, will return a random number
			until a prediction is made with learning on.
		:rtype: float
		"""
		if not self.model_was_optimized:
			return uniform(0.0, 1.0)
		payload = self._construct_predict_payload(timestamp, observation, action)
		r = requests.post(
			self.url_predict_with_learning_off,
			data=json.dumps(payload),
			headers={'Content-type': 'application/json', 'Accept': 'text/plain'},
		)
		if r.status_code != 200:
			raise RuntimeError(
				f'POST to {self.url_predict_with_learning_off} '
				f'returned: {r.status_code}\n\n{r.text}')
		else:
			response = json.loads(r.content.decode('utf-8'))
			print(f'Prediction received: {response}')
			self.last_prediction_msg = response['message'][0]
			self.last_prediction = self.last_prediction_msg['prediction']
			return self.last_prediction

	def send_message_predict_with_learning_on(self, timestamp, observation, action):
		"""
		Make a prediction with learning turned on.

		:param timestamp:
		:param observation: Input data as a "flat" tuple of floats.
		:type observation: tuple of float
		:param action: A specific action, we want the network to predict.
		:type action: float
		:return: A single predicted value.
		:rtype: float
		"""
		if not self.model_was_optimized:
			self.model_was_optimized = True
		payload = self._construct_predict_payload(timestamp, observation, action)
		r = requests.post(
			self.url_predict_with_learning_on,
			data=json.dumps(payload),
			headers={'Content-type': 'application/json', 'Accept': 'text/plain'},
		)
		if r.status_code != 200:
			raise RuntimeError(
				f'POST to {self.url_predict_with_learning_on} '
				f'returned: {r.status_code}\n\n{r.text}')
		else:
			response = json.loads(r.content.decode('utf-8'))
			print(f'Training prediction received: {response}')
			self.last_prediction_msg = response['message'][0]
			self.last_prediction = self.last_prediction_msg['prediction']
			return self.last_prediction

	def send_message_save_predictor(self):
		r = requests.post(self.url_save_predictor)
		if r.status_code != 200:
			raise RuntimeError(
				f'POST to {self.url_save_predictor} '
				f'returned: {r.status_code}\n\n{r.text}')
		else:
			print(f'{self} saved on prediction server.')

	def send_message_stop_predictor(self):
		r = requests.post(self.url_stop_predictor)
		if r.status_code != 200:
			raise RuntimeError(
				f'POST to {self.url_start_predictor} '
				f'returned: {r.status_code}\n\n{r.text}')
		else:
			print(f'{self} stopped on prediction server.')

	@property
	def last_predicted_value(self):
		if len(self.last_prediction_msg) > 0:
			return float(self.last_prediction_msg['prediction'])


class NupicModel(ModelBase):
	"""
	Creates a Nupic Network for Reinforcement Learning (Q-Learning).

	This model depends on Docker service `bamm-nupic-predictor` and it must
	be running inside a Flask docker container and available to receive HTTP
	requests on	port 5000.

	Typical nupic models will contain multiple nupic networks. For each predicted
	output one nupic network is required.  For example, to
	predict the following inputs and predicted outputs a nupic model will
	contain 11 nupic networks.  All inputs will be fed into each of the 11
	nupic networks.
		Market Data Inputs (13 in total for this example):
			order_book_level_1_price
			order_book_level_1_amount
			order_book_level_2_price
			order_book_level_2_amount
			order_book_level_3_price
			order_book_level_3_amount
			trade_sell_price
			trade_sell_amount
			trade_buy_price
			trade_buy_amount
			account_total_balance
			account_used_balance
			account_free_balance
		Primary Predicted Outputs (5 in total for this example):
			reward_of_HODL
			reward_of_market_sell
			reward_of_market_buy
			reward_of_limit_sell
			reward_of_limit_buy
		Secondary Predicted Outputs (6 in total for this example):
			reward_of_amount_level_1
			reward_of_amount_level_2
			reward_of_amount_level_3
			reward_of_price_level_1
			reward_of_price_level_2
			reward_of_price_level_3

	In the previous example, 11 nupic networks will be constructed in a single
	nupic model instance.  The 11 nupic networks will be grouped heirarchically
	into 5 primary networks and 6 secondary networks.  All 5 outputs of the
	primary networks will be fed into the secondary networks along with the
	market data inputs.

	This means that while the 5 primary networks will be fed only the market
	data inputs, the 6 secondary networks will be fed the market data
	inputs plus the 5 predicted outputs of the primary networks.  So, the
	secondary networks will each be fed the following:
		- 13 market data inputs
		- 5 primary network predicted outputs
	"""

	def __init__(self, exchange, base, quote, period_secs, input_field_names, action_names,
				 *args, **kwargs):
		"""
		Initialize a nupic predictor via HTTP requests to the `spread-predictor`
		running inside a Flask docker container.

		:param input_field_names: For example
			[
				'order_book_level_1_price',
				'order_book_level_1_amount',
				'trade_sell_price',
				'trade_sell_amount',
				...,
				'account_used_balance',
				'account_free_balance',
			]
		:param action_names: For example
			{
				'primary': ['hodl', 'market_sell', 'market_buy', ...],
				'amount': ['amount_level_1', 'amount_level_2', ...],
				'price': ['price_level_1', 'price_level_2', ...],
			}
		:type action_names: dict
		:param args:
		:param kwargs:
		"""
		super().__init__(*args, **kwargs)
		# ivars
		self.exchange = exchange
		self.base = base
		self.quote = quote
		self.period_secs = period_secs
		self.timeframe = f'{period_secs}s'
		self.primary_input_fields = copy.deepcopy(input_field_names)
		self.secondary_input_fields = copy.deepcopy(input_field_names)
		self.secondary_input_fields.append('selected_primary_action')

		# local vars used for initialization.
		primary_networks = []
		amount_networks = []
		price_networks = []

		# Primary output networks.
		# instantiate "buy/sell/HODL" networks for "primary" actions.
		for primary_action_name in action_names['primary']:
			nupic_network = NupicNetwork(
				exchange=exchange,
				base=base,
				quote=quote,
				input_fields=input_field_names,
				predicted_field=primary_action_name,
				timeframe=self.timeframe,
			)
			primary_networks.append(nupic_network)

		# Secondary output networks.
		# instantiate "amount" and "price" networks for "secondary" actions.
		for amount_action_name in action_names['amount']:
			nupic_network = NupicNetwork(
				exchange=exchange,
				base=base,
				quote=quote,
				input_fields=self.secondary_input_fields,
				predicted_field=amount_action_name,
				timeframe=self.timeframe,
			)
			amount_networks.append(nupic_network)

		# Input networks.
		# instantiate "price" networks.
		for price_action_name in action_names['price']:
			nupic_network = NupicNetwork(
				exchange=exchange,
				base=base,
				quote=quote,
				input_fields=self.secondary_input_fields,
				predicted_field=price_action_name,
				timeframe=self.timeframe,
			)
			price_networks.append(nupic_network)
		self.networks = collections.OrderedDict(
			{
				'primary': primary_networks, #: 5 primary output networks
				'amount': amount_networks, #: 6 secondary output networks
				'price': price_networks, #: 13 price data input networks
			}
		)

	def get_selected_action(self, q_values):
		"""
		Get the "selected action" by this model from given `q_values`.

		:param q_values:
		:type q_values: tuple of float

		:returns: The action returned assumes the possible actions are 0-indexed.
			For example, given the following q_values:
				(0.0, 0.9, 0.1, 0.1, 0.1)
			And assuming the following actions:
				['HODL', 'market_sell', 'market_buy', 'limit_sell', 'limit_buy']
			The selected action would be 'market_sell' or the integer 1.
		:rtype: int
		"""
		selected_action = q_values[0]
		index = 0
		for q_value in q_values:
			if q_value > selected_action:
				selected_action = q_value
				index = q_values.index(q_value)
		return selected_action

	def get_q_values(self, timestamp, observation):
		"""
		Get predicted Q-values for a given observation from this Nupic model.

		:param timestamp:
		:type timestamp: datetime.datetime
		:param observation:
			A flat list of observations.  Meaning a list of floats.
		:type observation: list of float

		:returns: q_values (aka the rewards predicted by this model).
	 		The output of this function is an array of Q-value-arrays.
			There is a Q-value for each possible action in the game-environment.
			So the output is a 3-dim array of Open AI discrete shapes:
				[spaces.Discrete(x), spaces.Discrete(y), spaces.Discrete(z)]

		 	To be more clear, a more concrete example follows.  Note, for sake
		 	of the following example all q-values are given as 0.5.
			For example: ((0.5,0.5,0.5,0.5,0.5), (0.5,0.5,0.5), (0.5,0.5,0.5))
		:rtype: tuple of tuple
		"""
		primary_q_values = []
		amount_q_values = []
		price_q_values = []
		secondary_observations = copy.deepcopy(observation)

		# predict primary actions
		for primary_network in self.networks['primary']:
			prediction = primary_network.send_message_predict_with_learning_off(
				timestamp,
				observation,
				1, # just use zero while network is not learning.
			)
			primary_q_values.append(prediction)

		# add the "selected primary action" to the secondary observations.
		selected_primary_action = self.get_selected_action(primary_q_values)
		secondary_observations.append(selected_primary_action)

		# predict "amount" actions
		for amount_network in self.networks['amount']:
			prediction = amount_network.send_message_predict_with_learning_off(
				timestamp,
				secondary_observations,
				1, # just use zero while network is not learning.
			)
			amount_q_values.append(prediction)

		# predict "price" actions
		for price_network in self.networks['price']:
			prediction = price_network.send_message_predict_with_learning_off(
				timestamp,
				secondary_observations,
				1, # just use zero while network is not learning.
			)
			price_q_values.append(prediction)
		primary_q_values = tuple(primary_q_values)
		amount_q_values = tuple(amount_q_values)
		price_q_values = tuple(price_q_values)
		# self.print_line('-')
		# print(f'Primary Observations: {observation}')
		# print(f'Secondary Observations: {secondary_observations}')
		# print(f'Q-Values: {(primary_q_values, amount_q_values, price_q_values)}')
		# self.print_line('-')
		return (primary_q_values, amount_q_values, price_q_values)

	def optimize(self, replay_memories, timestamp):
		"""
		Optimize the Nupic networks by feeding in replay memories and
		observations into each network with learning turned on.

		------------------------------------------------------------------------
		NOTE: replay_memory.q_values should contain the adjusted q-values,
		which means the q-values the nupic model should have predicted.  In
		other words, the q-values contain the "desired prediction", so just
		feed them into the nupic model with learning turned on.
		------------------------------------------------------------------------

		:param replay_memories:
		:type replay_memories: list of crypto_gym.agents.ReplayMemory

		:param timestamp:
		:type timestamp: datetime.datetime
		"""
		# initialize the actions selected by the "primary" networks, so they
		#  can be fed into the "amount" and "price" networks.
		selected_primary_actions = []
		replay_memory = replay_memories[0]
		for i in range(replay_memory.num_used):
			selected_primary_actions.append(replay_memory.actions[i])

		# train the "primary", "amount", and "price" networks.
		for network_type, networks in self.networks.items():
			rp_index = list(self.networks.keys()).index(network_type)
			replay_memory = replay_memories[rp_index]
			for network in networks:
				q_value_index = networks.index(network)
				for i in range(replay_memory.num_used):
					# extract observation and q-values from replay memory.
					observation = replay_memory.states[i]
					q_values = replay_memory.q_values[i]

					# select the "desired q-value", which corresponds to the
					# index of the network, because the networks and q-values
					# should have the same structure and be in the same order.
					# See NINJA-115 for more information.
					desired_q_value = q_values[q_value_index]
					if network_type == 'primary':
						# train the "primary" network.
						network.send_message_predict_with_learning_on(
							timestamp,
							observation,
							desired_q_value,
						)
					else:
						# add the "selected primary action" to the secondary observations.
						secondary_observations = copy.deepcopy(observation)
						secondary_observations = np.append(
							secondary_observations,
							[selected_primary_actions[i]],
						)

						# train the "secondary" network.
						network.send_message_predict_with_learning_on(
							timestamp,
							secondary_observations,
							desired_q_value,
						)

	@property
	def name(self):
		return f'{self.__class__.__name__}'

	def print_line(self, x):
		line = f'{x}' * 100
		print(f'\n{line}\n')

	def save_checkpoint(self):
		"""
		Serialize the model to the Django database so it can be loaded later.
		"""
		# save the "primary", "amount", and "price" networks.
		for network_type, networks in self.networks.items():
			for network in networks:
				# save the Nupic network.
				network.send_message_save_predictor()















