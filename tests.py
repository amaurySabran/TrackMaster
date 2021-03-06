from track_env import TrackEnv
from agents.agent import TestAgent

env = TrackEnv(track_file='tracks/track_0.npy')

state = env.reset()
action = (0.0, 0.1)  # acceleration, deviation_angle

for i in range(200):
    observation, reward, done, info = env.step(action)
    env.render()


