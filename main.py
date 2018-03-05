from track_env import TrackEnv
from agents.agent import TestAgent
from phazlib.utils import ShowVariableLogger


env = TrackEnv(track_file='tracks/track_0.npy')

agent = TestAgent(env=env, logger=ShowVariableLogger(average_window=100))

agent.train(nb_episodes=1000)

agent.test(nb_episodes=10, display=True)



