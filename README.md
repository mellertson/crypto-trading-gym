
# Crypto Trading Open AI Gym

`CryptoTradingGym` is a collection of [OpenAI Gym](https://github.com/openai/gym) 
environments for reinforcement learning-based trading algorithms.  It was 
forked from [`AnyTrading`](https://github.com/AminHP/gym-anytrading).

`CryptoTradingGym` is intended to be used with another project I authored, 
namely [`nupic_predictor`](https://github.com/mellertson/nupic-predictor).

## Required Dependencies

This project is dependant on the following external projects:

* nupic-predictor
* Crypto Ninja REST API (api)

## Running the Crypto Gym on LOCALHOST

1. Install required pip packages using `python3 -m pip install numpy cython pyparsing==2.4.7`
2. Install the Crypto Gym by executing `python ./setup.py install`
3. Run the test case in 
`Test_QLearnAgent_class.test_run_the_agent_for_one_episode`


# Building The Docker Container

Because, Docker Compose is used you can build the container by 
executing the following in a terminal.

```shell script
docker-compose build
```

# Deploying on JONIN (production)

To begin running the Nupic Q-Learning Trading Agent executing the following on 
a Docker Swarm manager node.

```shell script
docker stack deploy -c stack.jonin.yaml --with-registry-auth bamm-crypto-trading-gym
```

Once launched the you can view the agents progress training in the 
Crypto Trading Gym at https://testnet.bitmex.com.











