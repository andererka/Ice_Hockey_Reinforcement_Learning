# Reinforcement Learning Project (winter term 2020/2021)



Team project within Reinforcement Learning Course

Training an agent that learns to play IceHockey within the following environment: https://github.com/martius-lab/laser-hockey-env.git

Code: 

- **code/laserhockey** is were the environment is being defined
- **code/PPO** implements Proximal Policy Optimization for the ice hockey agent
- Training_PPO_LSTM_main_running.py trains the agent with the PPO method plus an LSTM component such that previous hidden states can be accessed as well. This could help to predict global motion, but learning was to slow in our case with limited computational power, such that we did not pursue this further.
- Training_PPO.py train the agent with PPO, but without an LSTM component.

For further information, as well as the results of our training, one can refer to the **final_report.pdf**.

