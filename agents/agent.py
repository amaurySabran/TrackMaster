# in this file we will implement reinforcement learning algorithms to apply to the env.
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy

from phazlib.rll import SpaceGrid, Agent, ContinuousStateActionMap
from phazlib.rll.ImplAgents import QLearner
from rl import *
from rl.memory import SequentialMemory
from keras.layers import Dense, Flatten, Reshape
from keras.models import Sequential
from phazlib.utils import Logger, ndargmax
from keras.optimizers import Adam
import gym
import numpy as np
from rl.core import Processor


class TestAgent(QLearner):
    pass


class DqnProcessor(Processor):
    # TODO move this elsewhere
    def __init__(self, shape, low, high):
        super()
        self.shape = shape
        self.low = low
        self.high = high

    def process_action(self, action):
        # print("action : {0}".format(action))
        processed_index = np.unravel_index(action, self.shape)
        processed_action = ()
        for i in range(len(processed_index)):
            processed_action = processed_action + (
                np.linspace(self.low[i], self.high[i], self.shape[i])[processed_index[i]],)
        print("processed action {0}".format(processed_action))
        return processed_action


class DqnAgent(Agent):

    def __init__(self,
                 env: gym.Env,
                 memory=SequentialMemory(limit=50000, window_length=1),
                 logger=Logger(),
                 boxes_resolution=10,
                 nb_steps_warmup=20,
                 hidden_layers=[16, 16, 16],
                 policy=BoltzmannQPolicy(),
                 target_model_update=1e-2,
                 optimizer=Adam(lr=1e-3)
                 ):

        self.env = env

        if isinstance(boxes_resolution, int):
            boxes_resolution = (boxes_resolution,) * len(env.action_space.shape)

        self.boxes_resolution = boxes_resolution
        self.nb_actions = np.zeros(boxes_resolution).size

        model = Sequential()
        model.add(Flatten(input_shape=(1,) + env.observation_space.shape))  # TODO check this
        for l in hidden_layers:
            model.add(Dense(l, activation='relu'))
        model.add(Dense(self.nb_actions, activation='linear'))  # TODO move this to util file?

        self.model = model
        print("dqn model summary :{0}".format(model.summary()))

        self.dqn = DQNAgent(model=model,
                            nb_actions=self.nb_actions,
                            memory=memory,
                            nb_steps_warmup=nb_steps_warmup,
                            target_model_update=target_model_update,
                            policy=policy,
                            processor=DqnProcessor(self.boxes_resolution, env.action_space.low, env.action_space.high))
        self.dqn.compile(optimizer=optimizer, metrics=['mae'])
        super().__init__(env, logger)

    def act(self, state, explore):
        action = self.dqn.processor.process_action(self.dqn.forward(state))
        return action

    def train(self, nb_episodes=1000, verbose=2, visualize=True):
        self.dqn.fit(env=self.env, nb_steps=nb_episodes, verbose=verbose, visualize=visualize)
        # TODO callbacks to log
