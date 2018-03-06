from track_env import TrackEnv
from agents.agent import TestAgent, DqnAgent
from phazlib.rll.ImplAgents import SarsaLearner, QLearner
from phazlib.utils import ShowVariableLogger


env = TrackEnv(track_file='tracks/track_0.npy')

# agent = QLearner(env=env,
#                      logger=ShowVariableLogger(average_window=100),
#                      boxes_resolution=(5,5,5,5,5,5,5,5,3,3))

agent = DqnAgent(env=env,
                     logger=ShowVariableLogger(average_window=100),
                     boxes_resolution=(3,6))

agent.train(nb_episodes=1000,visualize=False,verbose=1)

agent.test(nb_episodes=10, display=True)