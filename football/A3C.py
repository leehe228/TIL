import gfootball.env as football_env

import tensorflow as tf
import numpy as np
import random
import datetime
from collections import deque

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

state_size = 48
moving_action_size = 8
skill_action_size = 11

load_model = False
train_mode = True
render_mode = False

num_to_control = 1
academy_scenario = '11_vs_11_stochastic'
scoring = 'scoring,checkpoints'

discount_factor = 0.99

# Actor-Critic Learning Rate
actor_lr = 0.0001
critic_lr = 0.0005
tau = 0.001

mu = 0
theta = 0.001
sigma = 0.002

start_train_episode = 100
run_episode = 3000
test_episode = 100

save_interval = 300

date_time = datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S")

save_path = "./saved_models/DDPG/" + date_time
load_path = "./saved_models/DDPG/" + date_time + "/model0/model"


class Actor_Moving:
    def __init__(self):
        pass


class Actor_Skill:
    def __init__(self):
        pass


class Critic_Moving:
    def __init__(self, name):
        pass

    
class Critic_Skill:
    def __init__(self, name):
        pass


class Agent:
    def __init__(self):
        self.actor_moving = Actor_Moving()
        self.actor_skill = Actor_Skill()
        self.critic_moving = Critic_Moving()
        self.critic_skill = Critic_Skill()

    

    def train_model(self):
        pass

    def get_moving_action(self):
        pass

    def get_skill_action(self):
        pass

    def append_sample(self):
        pass

    def save_model(self):
        pass

    def make_Summary(self):
        pass

    def write_Summary(self):
        pass
    

if __name__ == "__main__":
    env = football_env.create_environment(
        env_name=academy_scenario,
        rewards=scoring,
        render=render_mode,
        number_of_left_players_agent_controls=num_to_control,
        number_of_right_players_agent_controls=num_to_control)

    agent1, agent2 = Agent(), Agent()
    step = 0

    for episode in range(run_episode + test_episode):
        if episode == run_episode:
            train_mode = False

        env.reset()
        episode_reward1 = 0.0
        episode_reward2 = 0.0

        obs, reward, done, info = env.step([12, 12])

        state1 = np.concatenate(
            (np.where(obs[0] == 255)[0], np.where(obs[0] == 255)[1]), axis=None)
        state2 = np.concatenate(
            (np.where(obs[1] == 255)[0], np.where(obs[1] == 255)[1]), axis=None)

        while not done:
            step += 1

            next_obs, reward, done, info = env.step([12, 12])

            next_state1 = np.concatenate(
                (np.where(next_obs[0] == 255)[0], np.where(next_obs[0] == 255)[1]), axis=None)
            next_state2 = np.concatenate(
                (np.where(next_obs[1] == 255)[0], np.where(next_obs[1] == 255)[1]), axis=None)

            if step % 20 == 0:
                agent1.train_model()
                agent2.train_model()
