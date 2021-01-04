from . import ModelBase
import copy


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

	def __init__(self, exchange, base, quote, input_fields, predicted_field,
				 timeframe, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.exchange = exchange
		self.base = base
		self.quote = quote
		self.input_fields = input_fields
		self.predicted_field = predicted_field
		self.timeframe = timeframe
		self.predictor_id = None
		self.last_prediction_msg = {}
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

		# build and add encoders (inputs) into the model
		encoders = []
		for field_name in self.input_fields:
			encoder = self._build_random_distributed_scalar_encoder(field_name)
			encoders.append(encoder)
		timestamp_encoders = self._build_timestamp_encoders('timestamp')
		encoders.extend(timestamp_encoders)
		model['modelParams']['sensorParams']['encoders'] = encoders

		# build and add the classifier (output) into the model
		classifiers = []
		classifier = self._build_sdr_classifier_region(self.predicted_field)
		classifiers.append(classifier)
		model['modelParams']['classifiers'] = classifiers
		self.model = model

	@property
	def url_new_predictor(self):
		"""
		Return a fully-qualified URL to create a new predictor via POST request.
		:rtype: str
		"""
		url = f'{PREDICTOR_SERVER_BASE_URL}/new/predictor/'

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
	def line_2(cls, x, y):
		line = ''.join([x] + [', float'] * (len(y) + 1))
		return line

	@classmethod
	def line_3(cls, x, y, z=None):
		return ''.join([x] + [','] * (len(y) + 1))

	@classmethod
	def _build_random_distributed_scalar_encoder(
		cls, field_name, resolution=0.88, seed=1,
	):
		encoder = {
			field_name: {
				'type': 'RandomDistributedScalarEncoder',
				'fieldname': field_name,
				'name': field_name,
				'resolution': resolution,
				'seed': seed,
			}
		}
		return encoder

	@classmethod
	def _build_category_encoder(
		cls, field_name, w=21, category_list='1,2,3,4,5',
	):
		encoder = {
			field_name: {
				'type': 'CategoryEncoder',
				'fieldname': field_name,
				'name': field_name,
				'w': w,
				'category_list': category_list,
			}
		}
		return encoder

	@classmethod
	def _build_timestamp_encoders(cls, field_name):
		encoders = [
			{
				'time_of_day': {
					'type': 'DateEncoder',
					'fieldname': field_name,
					'name': 'time_of_day',
					'timeOfDay': [21, 1],
				}
			},
			{
				'weekend': {
					'type': 'DateEncoder',
					'fieldname': field_name,
					'name': 'weekend',
					'weekend': 21
				}
			},
			{
				'season': {
					'type': 'DateEncoder',
					'fieldname': field_name,
					'name': 'season',
					'season': 21,
				}
			}
		]
		return encoders

	@classmethod
	def _build_sdr_classifier_region(
		cls, predicted_field, max_category_count=1000,
		steps='1', alpha=0.1, verbosity=0
	):
		classifier = {
			predicted_field: {
				'regionType': 'SDRClassifierRegion',
				'verbosity': verbosity,
				'alpha': alpha,
				'steps': steps,
				'maxCategoryCount': max_category_count,
				'implementation': 'cpp',
			}
		}
		return classifier

	def _construct_predict_payload(self, timestamp, observation, action):
		"""

		:type timestamp: str | datetime.datetime
		:type observation: tuple
		:type action: tuple
		:rtype: str
		"""
		if isinstance(timestamp, datetime):
			line = timestamp.isoformat()
		else:
			line = str(timestamp)
		for x in observation:
			line += ', ' + str(x)
		for x in action:
			line += ', ' + str(x)
		return line

	def send_message_new_predictor(self):
		payload = {
			'model': yaml_model,
			'exchange': self.exchange.lower(),
			'market': f'{self.base.lower()}{self.quote.lower()}',
			'predicted_field': self.predicted_field.lower(),
			'timeframe': self.timeframe.lower(),
		}
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
			'http://localhost:5000/start/predictor/{}/'.format(self.predictor_id),
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
			self.last_prediction_msg = json.loads(r.text.decode('utf-8'))['message'][0]
			print(f'Prediction received: {self}')
			return self.last_prediction_msg

	def send_message_predict_with_learning_on(self, timestamp, observation, action):
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
			self.last_prediction_msg = json.loads(r.text.decode('utf-8'))['message'][0]
			print(f'Training prediction received: {self}')
			return self.last_prediction_msg

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


class NetworkLink(object):
	""" Connects the output of a nupic network to the input of a nupic network. """

	def __init__(self, output_network, input_network, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.output_network = output_network
		self.input_network = input_network


class NupicModel(ModelBase):
	"""
	Creates a Nupic Network for Reinforcement Learning (Q-Learning).

	This model depends on the `spread-predictor` and it must be running inside
	a Flask docker container and available to receive HTTP requests on
	port 5000.

	Typical nupic models will contain multiple nupic networks. For each predited
	output one additional nupic netork is required.  For example, to
	predict the following inputs and predicted outputs a nupic model will
	contain 11 nupic networks.  All inputs will be fed into each of the 11
	nupic networks.
		Market Data Inputs:
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
		Primary Predicted Outputs:
			reward_of_HODL
			reward_of_market_sell
			reward_of_market_buy
			reward_of_limit_sell
			reward_of_limit_buy
		Secondary Predicted Outputs:
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

	# HIGH: write NupicModel.run() method.

	def __init__(self, exchange, base, quote, timeframe,
				 input_field_names, action_names, *args, **kwargs):
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
		self.timeframe = timeframe
		self.primary_input_fields = copy.deepcopy(input_field_names)
		self.secondary_input_fields = copy.deepcopy(input_field_names)
		self.secondary_input_fields.extend(action_names['primary'])

		# local vars used for initialization.
		primary_networks = []
		amount_networks = []
		price_networks = []

		# instantiate primary networks for "primary" actions.
		for primary_action_name in action_names['primary']:
			nupic_network = NupicNetwork(
				exchange=exchange,
				base=base,
				quote=quote,
				input_fields=input_field_names,
				predicted_field=primary_action_name,
				timeframe=timeframe,
			)
		primary_networks.append(nupic_network)

		# instantiate "amount" networks.
		for amount_action_name in action_names['amount']:
			nupic_network = NupicNetwork(
				exchange=exchange,
				base=base,
				quote=quote,
				input_fields=self.secondary_input_fields,
				predicted_field=amount_action_name,
				timeframe=timeframe,
			)
			amount_networks.append(nupic_network)

		# instantiate "price" networks.
		for price_action_name in action_names['price']:
			nupic_network = NupicNetwork(
				exchange=exchange,
				base=base,
				quote=quote,
				input_fields=self.secondary_input_fields,
				predicted_field=price_action_name,
				timeframe=timeframe,
			)
			price_networks.append(nupic_network)
		self.networks = {
			'primary': primary_networks,
			'amount': amount_networks,
			'price': price_networks,
		}

	def get_predicted_actions(self, timestamp, observation):
		"""
		Get the estimated Q-values for the given states from the Nupic predictor.

		The output of this function is an array of Q-value-arrays.
		There is a Q-value for each possible action in the game-environment.
		So the output is a 3-dim array of Open AI discrete shapes:
			[spaces.Discrete(x), spaces.Discrete(y), spaces.Discrete(z)]

		:param timestamp:
		:type timestamp: datetime.datetime
		:param observation:
		:type observation: list of float
		"""
		primary_q_values = ()
		amount_q_values = ()
		price_q_values = ()

		# predict primary actions
		for primary_network in self.networks['primary']:
			prediction = primary_network.send_message_predict_with_learning_off(
				timestamp,
				observation,
				0, # just use zero while network is not learning.
			)
			primary_q_values.append(prediction)

		# build observations for "amount" and "price" networks.
		secondary_observations = (o for o in observation)
		for q_value in primary_q_values:
			secondary_observations.append(q_value)

		# predict "amount" actions
		for amount_network in self.networks['amount']:
			prediction = amount_network.send_message_predict_with_learning_off(
				timestamp,
				observation,
				0, # just use zero while network is not learning.
			)
			amount_q_values.append(prediction)

		# predict "price" actions
		for price_network in self.networks['price']:
			prediction = price_network.send_message_predict_with_learning_off(
				timestamp,
				observation,
				0, # just use zero while network is not learning.
			)
			price_q_values.append(prediction)
		return (primary_q_values, amount_q_values, primary_q_values)

	def optimize(self, replay_memory, timestamp, observation, ):
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
		raise NotImplementedError(
			'connect this method to the Nupic predictor using HTTP POST request.'
		)
		# CURRENT: send observation, and Q-values from replay memory to nupic model.
		#  NOTE: replay_memory.q_values should contain the adjusted q-values, which means
		#  	the q-values the nupic model should have predicted.  In other words, the
		#   q-values contain the "desired prediction", so just feed them into
		#   the nupic model with learning turned on.


