#-*-coding:utf-8-*-

import gfootball.env as football_env

import tensorflow as tf
import numpy as np
import random
import datetime
from collections import deque

action_set = ['0', '←', '↖', '↑', '↗', '→', '↘', '↓', '↙', 'long', 'high', 'short', 'shoot', '+run', '-', '-run', 'sliding', '+dribble', '-dribble']

state_size = 115
moving_action_size = 8
skill_action_size = 10

load_model = False
train_mode = True
render_mode = False

num_to_control = 1
academy_scenario = '11_vs_11_easy_stochastic'
scoring = 'scoring,checkpoints'

batch_size = 1024
mem_maxlen = int(1e6)
discount_factor = 0.9999
actor_lr = 0.0000001
critic_lr = 0.0000005
tau = 0.0000005

mu = 0
theta = 0.000001
sigma = 0.000003

start_train_episode = 1000
run_episode = 5000000
test_episode = 1000

epsilon_init = 0.6
epsilon_min = 0.05
epsilon_desc = (epsilon_init - epsilon_min) / run_episode

print_interval = 1
# save_interval = 1

date_time = datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S")


save_path = "./saved_models/A2C/" + date_time
load_path = "./saved_models/A2C/" + date_time + "/model0/model"


class Actor_Moving:
    def __init__(self, name):
        with tf.variable_scope(name):
            self.state = tf.placeholder(tf.float32, [None, state_size])
            self.fc1 = tf.layers.dense(self.state, 128, activation=tf.nn.relu)
            self.fc2 = tf.layers.dense(self.fc1, 128, activation=tf.nn.relu)
            self.action = tf.layers.dense(self.fc2, moving_action_size, activation=tf.nn.tanh)

        self.trainable_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, name)


class Actor_Skill:
    def __init__(self, name):
        with tf.variable_scope(name):
            self.state = tf.placeholder(tf.float32, [None, state_size])
            self.fc1 = tf.layers.dense(self.state, 128, activation=tf.nn.relu)
            self.fc2 = tf.layers.dense(self.fc1, 128, activation=tf.nn.relu)
            self.action = tf.layers.dense(self.fc2, skill_action_size, activation=tf.nn.tanh)

        self.trainable_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, name)


class Critic_Moving:
    def __init__(self, name):
        with tf.variable_scope(name):
            self.state = tf.placeholder(tf.float32, [None, state_size])
            self.fc1 = tf.layers.dense(self.state, 128, activation=tf.nn.relu)
            self.action = tf.placeholder(tf.float32, [None, moving_action_size])
            self.concat = tf.concat([self.fc1, self.action],axis=-1)
            self.fc2 = tf.layers.dense(self.concat, 128, activation=tf.nn.relu)
            self.fc3 = tf.layers.dense(self.fc2, 128, activation=tf.nn.relu)
            self.predict_q = tf.layers.dense(self.fc3, 1, activation=None)

        self.trainable_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, name)
    

class Critic_Skill:
    def __init__(self, name):
        with tf.variable_scope(name):
            self.state = tf.placeholder(tf.float32, [None, state_size])
            self.fc1 = tf.layers.dense(self.state, 128, activation=tf.nn.relu)
            self.action = tf.placeholder(tf.float32, [None, skill_action_size])
            self.concat = tf.concat([self.fc1, self.action],axis=-1)
            self.fc2 = tf.layers.dense(self.concat, 128, activation=tf.nn.relu)
            self.fc3 = tf.layers.dense(self.fc2, 128, activation=tf.nn.relu)
            self.predict_q = tf.layers.dense(self.fc3, 1, activation=None)

        self.trainable_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, name)


class Agent:
    def __init__(self, number):
        self.actor_moving = Actor_Moving("actor_moving" + number)
        self.critic_moving = Critic_Moving("critic_moving" + number)

        self.actor_skill = Actor_Skill("actor_skill" + number)
        self.critic_skill = Critic_Skill("critic_skill" + number)

        self.target_actor_moving = Actor_Moving("target_actor_moving" + number)
        self.target_critic_moving = Critic_Moving("target_critic_moving" + number)

        self.target_actor_skill = Actor_Skill("target_actor_skill" + number)
        self.target_critic_skill = Critic_Skill("target_critic_skill" + number)
        
        self.target_q_moving = tf.placeholder(tf.float32, [None, 1])
        self.target_q_skill = tf.placeholder(tf.float32, [None, 1])

        self.critic_moving_loss = tf.losses.mean_squared_error(self.target_q_moving, self.critic_moving.predict_q)
        self.critic_skill_loss = tf.losses.mean_squared_error(self.target_q_skill, self.critic_skill.predict_q)

        self.train_critic_moving = tf.train.AdamOptimizer(critic_lr).minimize(self.critic_moving_loss)
        self.train_critic_skill = tf.train.AdamOptimizer(critic_lr).minimize(self.critic_skill_loss)

        self.action_grad_moving = tf.gradients(tf.squeeze(self.critic_moving.predict_q), self.critic_moving.action)
        self.action_grad_skill = tf.gradients(tf.squeeze(self.critic_skill.predict_q), self.critic_skill.action)

        self.policy_grad_moving = tf.gradients(self.actor_moving.action, self.actor_moving.trainable_var, self.action_grad_moving)
        self.policy_grad_skill = tf.gradients(self.actor_skill.action, self.actor_skill.trainable_var, self.action_grad_skill)

        for idx, grads in enumerate(self.policy_grad_moving):
            self.policy_grad_moving[idx] = -grads/batch_size
        self.train_actor_moving = tf.train.AdamOptimizer(actor_lr).apply_gradients(zip(self.policy_grad_moving, self.actor_moving.trainable_var))

        for idx, grads in enumerate(self.policy_grad_skill):
            self.policy_grad_skill[idx] = -grads/batch_size
        self.train_actor_skill = tf.train.AdamOptimizer(actor_lr).apply_gradients(zip(self.policy_grad_skill, self.actor_skill.trainable_var))
  
        self.sess_moving = tf.Session()
        self.sess_moving.run(tf.global_variables_initializer())
        self.sess_skill = tf.Session()
        self.sess_skill.run(tf.global_variables_initializer())

        self.Saver = tf.train.Saver()
        self.Summary, self.Merge = self.Make_Summary()
        self.memory_moving = deque(maxlen=mem_maxlen)
        self.memory_skill = deque(maxlen=mem_maxlen)

        self.soft_update_target_moving = []
        self.soft_update_target_skill = []
        
        for idx in range(len(self.actor_moving.trainable_var)):
            self.soft_update_target_moving.append(self.target_actor_moving.trainable_var[idx].assign(((1 - tau) * self.target_actor_moving.trainable_var[idx].value()) + (tau * self.actor_moving.trainable_var[idx].value())))
        for idx in range(len(self.critic_moving.trainable_var)):
            self.soft_update_target_moving.append(self.target_critic_moving.trainable_var[idx].assign(((1 - tau) * self.target_critic_moving.trainable_var[idx].value()) + (tau * self.critic_moving.trainable_var[idx].value())))

        for idx in range(len(self.actor_skill.trainable_var)):
            self.soft_update_target_skill.append(self.target_actor_skill.trainable_var[idx].assign(((1 - tau) * self.target_actor_skill.trainable_var[idx].value()) + (tau * self.actor_skill.trainable_var[idx].value())))
        for idx in range(len(self.critic_skill.trainable_var)):
            self.soft_update_target_skill.append(self.target_critic_skill.trainable_var[idx].assign(((1 - tau) * self.target_critic_skill.trainable_var[idx].value()) + (tau * self.critic_skill.trainable_var[idx].value())))
        
        init_update_target_moving = []
        init_update_target_skill = []

        for idx in range(len(self.actor_moving.trainable_var)):
            init_update_target_moving.append(self.target_actor_moving.trainable_var[idx].assign(self.actor_moving.trainable_var[idx]))
        for idx in range(len(self.critic_moving.trainable_var)):
            init_update_target_moving.append(self.target_critic_moving.trainable_var[idx].assign(self.critic_moving.trainable_var[idx]))
        self.sess_moving.run(init_update_target_moving)

        
        for idx in range(len(self.actor_skill.trainable_var)):
            init_update_target_skill.append(self.target_actor_skill.trainable_var[idx].assign(self.actor_skill.trainable_var[idx]))
        for idx in range(len(self.critic_skill.trainable_var)):
            init_update_target_skill.append(self.target_critic_skill.trainable_var[idx].assign(self.critic_skill.trainable_var[idx]))
        self.sess_skill.run(init_update_target_skill)

        self.epsilon = epsilon_init

        #if load_model == True:
        #    self.Saver.restore(self.sess, load_path)


    def get_action_moving(self, state):
        actions = self.sess_moving.run(self.actor_moving.action, feed_dict={self.actor_moving.state: [state]})
        return actions

    def get_action_skill(self, state):
        actions = self.sess_skill.run(self.actor_skill.action, feed_dict={self.actor_skill.state: [state]})
        return actions

    def append_sample_moving(self, state, action, reward, next_state, done):
        self.memory_moving.append((state, action, reward, next_state, done))

    def append_sample_skill(self, state, action, reward, next_state, done):
        self.memory_skill.append((state, action, reward, next_state, done))

    #def save_model(self):
    #    self.Saver.save(self.sess, save_path + "/model/model")
    
    def train_model_moving(self):
        mini_batch = random.sample(self.memory_moving, batch_size)
        states = np.asarray([sample[0] for sample in mini_batch])
        actions = np.asarray([sample[1] for sample in mini_batch])
        rewards = np.asarray([sample[2] for sample in mini_batch])
        next_states = np.asarray([sample[3] for sample in mini_batch])
        dones = np.asarray([sample[4] for sample in mini_batch])

        target_actor_actions_moving = self.sess_moving.run(self.target_actor_moving.action,
                                            feed_dict={self.target_actor_moving.state: next_states})
        target_critic_predict_qs_moving = self.sess_moving.run(self.target_critic_moving.predict_q,
                                                feed_dict={self.target_critic_moving.state: next_states,
                                                self.target_critic_moving.action: target_actor_actions_moving})
        target_qs_moving = np.asarray([reward + discount_factor * (1 - done) * target_critic_predict_q
                                for reward, target_critic_predict_q, done in zip(rewards, target_critic_predict_qs_moving, dones)])
        self.sess_moving.run(self.train_critic_moving, feed_dict={self.critic_moving.state: states,
                                                    self.critic_moving.action: actions,
                                                    self.target_q_moving: target_qs_moving})

        actions_for_train_moving = self.sess_moving.run(self.actor_moving.action, feed_dict={self.actor_moving.state: states})
        self.sess_moving.run(self.train_actor_moving, feed_dict={self.actor_moving.state: states,
                                                   self.critic_moving.state: states,
                                                   self.critic_moving.action: actions_for_train_moving})
                                                   
        self.sess_moving.run(self.soft_update_target_moving)


    def train_model_skill(self):
        mini_batch = random.sample(self.memory_skill, batch_size)
        states = np.asarray([sample[0] for sample in mini_batch])
        actions = np.asarray([sample[1] for sample in mini_batch])
        rewards = np.asarray([sample[2] for sample in mini_batch])
        next_states = np.asarray([sample[3] for sample in mini_batch])
        dones = np.asarray([sample[4] for sample in mini_batch])

        target_actor_actions_skill = self.sess_skill.run(self.target_actor_skill.action,
                                            feed_dict={self.target_actor_skill.state: next_states})
        target_critic_predict_qs_skill = self.sess_skill.run(self.target_critic_skill.predict_q,
                                                feed_dict={self.target_critic_skill.state: next_states,
                                                self.target_critic_skill.action: target_actor_actions_skill})
        target_qs_skill = np.asarray([reward + discount_factor * (1 - done) * target_critic_predict_q
                                for reward, target_critic_predict_q, done in zip(
                                                        rewards, target_critic_predict_qs_skill, dones)])
        self.sess_skill.run(self.train_critic_skill, feed_dict={self.critic_skill.state: states,
                                                    self.critic_skill.action: actions,
                                                    self.target_q_skill: target_qs_skill})

        actions_for_train_skill = self.sess_skill.run(self.actor_skill.action, feed_dict={self.actor_skill.state: states})
        self.sess_skill.run(self.train_actor_skill, feed_dict={self.actor_skill.state: states,
                                                   self.critic_skill.state: states,
                                                   self.critic_skill.action: actions_for_train_skill})
                                                   
        self.sess_skill.run(self.soft_update_target_skill)


    def Make_Summary(self):
        self.summary_rewards = tf.placeholder(tf.float32)
        tf.summary.scalar("mean reward", self.summary_rewards)
        Summary = tf.summary.FileWriter(
            logdir=save_path, graph=self.sess_moving.graph)
        Merge = tf.summary.merge_all()

        return Summary, Merge
        
    def Write_Summray(self, rewards, episode):
        self.Summary.add_summary(self.sess_moving.run(self.Merge, feed_dict={
            self.summary_rewards: rewards}), episode)


if __name__ == '__main__':
    env = football_env.create_environment(
        env_name=academy_scenario,
        rewards=scoring,
        render=render_mode,
        number_of_left_players_agent_controls=num_to_control,
        representation='simple115v2')

    agent = Agent('1')
    step = 0

    drib_time = 0.0

    for episode in range(run_episode + test_episode):
        if episode == run_episode:
            train_mode = False

        done = False
        
        obs = env.reset()
        episode_rewards = 0.0

        action_moving_arr = agent.get_action_moving(obs)
        action_moving = np.argmax(action_moving_arr) + 1

        action_skill_arr = agent.get_action_skill(obs)
        action_skill = np.argmax(action_skill_arr) + 9

        step = 0

        while not done:
            step += 1

            if random.random() < agent.epsilon:
                while True:
                    action_sample = env.action_space.sample()
                    if action_sample != 0: break

                if moving_action_size >= action_sample >= 1:
                    action_moving = action_sample

                else:
                    action_skill = action_sample

                next_obs, reward, done, info = env.step(action_sample)

            else:
                if step % 10 == 0:
                    action_moving_arr = agent.get_action_moving(obs)
                    action_moving = np.argmax(action_moving_arr) + 1

                    next_obs, reward, done, info = env.step(action_moving)
                else:
                    action_skill_arr = agent.get_action_skill(obs)
                    action_skill = np.argmax(action_skill_arr) + 9

                    next_obs, reward, done, info = env.step(action_skill)

            print("s: {} | ep: {} | r: {:.3f} | a : {} | s : {} |               ".format(step, episode, episode_rewards, action_set[action_moving], action_set[action_skill]), end='\r')

            reward = reward - (0.00001 * 11) 

            episode_rewards += reward
            
            if train_mode:
                agent.append_sample_moving(obs, action_moving_arr[0], reward, next_obs, done)
                agent.append_sample_skill(obs, action_skill_arr[0], reward, next_obs, done)
            
            obs = next_obs
   
            if episode > start_train_episode and train_mode and step % 10 == 0:
                agent.train_model_moving()
                agent.train_model_skill()

        if episode % print_interval == 0 and episode != 0:
            print("s: {} | ep: {} | r: {:.3f} |                                                  ".format(step, episode, episode_rewards))
            agent.Write_Summray(episode_rewards, episode)
        
        if agent.epsilon > epsilon_min:
            agent.epsilon -= epsilon_desc

    env.close()