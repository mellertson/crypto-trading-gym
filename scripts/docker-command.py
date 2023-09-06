#!/usr/bin/env python3
import os
from crypto_gym.agents.qlearn import *
from crypto_gym.envs.crypto_env import CryptoEnv
from crypto_gym.models import set_predictor_server_base_url


GYM_ENVIRONMENT_NAME = 'QLearningAgent'
EXCHANGE = os.environ.get(
	'EXCHANGE',
	default='bitmex',
).lower()
BASE_CURRENCY = os.environ.get(
	'BASE_CURRENCY',
	default='BTC',
).upper()
QUOTE_CURRENCY = os.environ.get(
	'QUOTE_CURRENCY',
	default='USD',
).upper()
PERIOD_SECONDS = int(os.environ.get(
	'PERIOD_SECONDS',
	default=2,
))
INPUT_ORDER_BOOK_LEVELS = int(os.environ.get(
	'INPUT_ORDER_BOOK_LEVELS',
	default=3,
))
if INPUT_ORDER_BOOK_LEVELS <= 0:
	raise Exception(f'INPUT_ORDER_BOOK_LEVELS must be at least 1')
REPLAY_MEMORY_SIZE = int(os.environ.get(
	'REPLAY_MEMORY_SIZE',
	default=500,
))
if REPLAY_MEMORY_SIZE <= 0:
	raise Exception(f'REPLAY_MEMORY_SIZE must be at least 1')

# Nupic Predictor REST API URL
NUPIC_PREDICTOR_URL = set_predictor_server_base_url(
	os.environ.get('PREDICTOR_SERVER_BASE_URL'),
)

# Django REST API URL
BASE_URL = os.environ.get('DJANGO_REST_API_URL')


# set to None to run forever
NUM_EPISODES = os.environ.get(
	'NUM_EPISODES',
	default=None,
)
if NUM_EPISODES is not None:
	NUM_EPISODES = int(NUM_EPISODES)

args = dict(
	GYM_ENVIRONMENT_NAME=GYM_ENVIRONMENT_NAME,
	EXCHANGE=EXCHANGE,
	BASE_CURRENCY=BASE_CURRENCY,
	QUOTE_CURRENCY=QUOTE_CURRENCY,
	PERIOD_SECONDS=PERIOD_SECONDS,
	INPUT_ORDER_BOOK_LEVELS=INPUT_ORDER_BOOK_LEVELS,
	BASE_URL=BASE_URL,
	REPLAY_MEMORY_SIZE=REPLAY_MEMORY_SIZE,
	NUM_EPISODES=NUM_EPISODES,
)
args = dict(sorted(args.items()))

if __name__ == '__main__':
	print(f'Initializing QLearningAgent with:')
	for k, v in args.items():
		print(f'\t{k} = {v}')
	agent = QLearningAgent(
		GYM_ENVIRONMENT_NAME,
		EXCHANGE,
		BASE_CURRENCY,
		QUOTE_CURRENCY,
		PERIOD_SECONDS,
		INPUT_ORDER_BOOK_LEVELS,
		BASE_URL,
		rp_mem_size=REPLAY_MEMORY_SIZE,
	)
	agent.is_running = True
	agent.run(num_episodes=NUM_EPISODES)



