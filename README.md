
# Crypto Trading Open AI Gym

`CryptoTradingGym` is a collection of [OpenAI Gym](https://github.com/openai/gym) 
environments for reinforcement learning-based trading algorithms.  It was 
forked from [`AnyTrading`](https://github.com/AminHP/gym-anytrading).

`CryptoTradingGym` is intended to be used with another project I authored, 
namely [`nupic_predictor`](https://gitlab.com/cybertronninja/nupic-rest-interface).

## Required Dependencies

This project is dependant on the following external projects:

* nupic-predictor
* Crypto Ninja REST API (api)

## Running the Crypto Gym

1. Install required pip packages using `python3 -m pip install numpy cython pyparsing==2.4.7`
2. Install the Crypto Gym by executing `python ./setup.py install`
3. Run the test case in 
`Test_QLearnAgent_class.test_run_the_agent_for_one_episode`





