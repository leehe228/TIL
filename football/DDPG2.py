import gfootball.env as football_env

import tensorflow as tf
import numpy as np
import random
import datetime
from collections import deque

state_size = 48
moving_action_size = 8
skill_action_size = 11

load_model = False
train_mode = True
render_mode = False

num_to_control = 1
academy_scenario = '11_vs_11_stochastic'
scoring = 'scoring,checkpoints'

batch_size = 128
mem_maxlen = 50000
discount_factor = 0.99
actor_lr = 1e-4
critic_lr = 5e-4
tau = 1e-3

mu = 0
theta = 1e-3
sigma = 2e-3

start_train_episode = 100
run_episode = 500
test_episode = 100

print_interval = 5
save_interval = 100

date_time = datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S")


save_path = "./saved_models/DDPG/" + date_time
load_path = "./saved_models/DDPG/" + date_time + "/model0/model"


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
        critic_moving_loss = tf.losses.mean_squared_error(self.target_q_moving, self.critic_moving.predict_q)
        critic_skill_loss = tf.losses.mean_squared_error(self.target_q_skill, self.critic_skill.predict_q)
        self.train_critic_moving = tf.train.AdamOptimizer(critic_lr).minimize(critic_moving_loss)
        self.train_critic_skill = tf.train.AdamOptimizer(critic_lr).minimize(critic_skill_loss)

        action_grad_moving = tf.gradients(tf.squeeze(self.critic_moving.predict_q), self.critic_moving.action)
        action_grad_skill = tf.gradients(tf.squeeze(self.critic_skill.predict_q), self.critic_skill.action)
        policy_grad_moving = tf.gradients(self.actor_moving.action, self.actor_moving.trainable_var, action_grad_moving)
        policy_grad_skill = tf.gradients(self.actor_skill.action, self.actor_skill.trainable_var, action_grad_skill)
        for idx, grads in enumerate(policy_grad_moving):
            policy_grad_moving[idx] = -grads/batch_size
        self.train_actor_moving = tf.train.AdamOptimizer(actor_lr).apply_gradients(
                                                            zip(policy_grad_moving, self.actor_moving.trainable_var))

        for idx, grads in enumerate(policy_grad_skill):
            policy_grad_skill[idx] = -grads/batch_size
        self.train_actor_skill = tf.train.AdamOptimizer(actor_lr).apply_gradients(
                                                            zip(policy_grad_skill, self.actor_skill.trainable_var))
  
        self.sess_moving = tf.Session()
        self.sess_skill = tf.Session()
        self.sess_moving.run(tf.global_variables_initializer())
        self.sess_skill.run(tf.global_variables_initializer())

        # self.Saver = tf.train.Saver()
        self.Summary, self.Merge = self.Make_Summary()
        self.memory_moving = deque(maxlen=mem_maxlen)
        self.memory_skill = deque(maxlen=mem_maxlen)

        self.soft_update_target_moving = []
        self.soft_update_target_skill = []
        
        for idx in range(len(self.actor.trainable_var)):
            self.soft_update_target_moving.append(self.target_actor_moving.trainable_var[idx].assign(((1 - tau) * self.target_actor_moving.trainable_var[idx].value()) + (tau * self.actor_moving.trainable_var[idx].value())))
        for idx in range(len(self.critic.trainable_var)):
            self.soft_update_target_moving.append(self.target_critic_moving.trainable_var[idx].assign(((1 - tau) * self.target_critic_moving.trainable_var[idx].value()) + (tau * self.critic_moving.trainable_var[idx].value())))

        for idx in range(len(self.actor.trainable_var)):
            self.soft_update_target_skill.append(self.target_actor_skill.trainable_var[idx].assign(((1 - tau) * self.target_actor_skill.trainable_var[idx].value()) + (tau * self.actor_skill.trainable_var[idx].value())))
        for idx in range(len(self.critic.trainable_var)):
            self.soft_update_target_skill.append(self.target_critic_skill.trainable_var[idx].assign(((1 - tau) * self.target_critic_skill.trainable_var[idx].value()) + (tau * self.critic_skill.trainable_var[idx].value())))
        
        init_update_target_moving = []
        init_update_target_skill = []

        for idx in range(len(self.actor.trainable_var)):
            init_update_target_moving.append(self.target_actor_moving.trainable_var[idx].assign(self.actor_moving.trainable_var[idx]))
        for idx in range(len(self.critic.trainable_var)):
            init_update_target_moving.append(self.target_critic_moving.trainable_var[idx].assign(self.critic_moving.trainable_var[idx]))
        self.sess_moving.run(init_update_target_moving)

        
        for idx in range(len(self.actor.trainable_var)):
            init_update_target_skill.append(self.target_actor_skill.trainable_var[idx].assign(self.actor_skill.trainable_var[idx]))
        for idx in range(len(self.critic.trainable_var)):
            init_update_target_skill.append(self.target_critic_skill.trainable_var[idx].assign(self.critic_skill.trainable_var[idx]))
        self.sess_skill.run(init_update_target_skill)

        #if load_model == True:
        #    self.Saver.restore(self.sess, load_path)


    def get_action_moving(self, state):
        if self.epsilon > np.random.rand():
            action = env.action_space.sample()
            actions = []
            for a in action:
                temp = [0.0 for k in range(moving_action_size)]
                temp[a] = 1.0
                actions.append(temp)
        else:
            actions = self.sess_moving.run(self.actor_moving.action, feed_dict={self.actor_moving.state: state})

        return actions


    def get_action_skill(self, state):
        if self.epsilon > np.random.rand():
            action = env.action_space.sample()
            actions = []
            for a in action:
                temp = [0.0 for k in range(moving_action_size)]
                temp[a] = 1.0
                actions.append(temp)
        else:
            actions = self.sess_skill.run(self.actor_skill.action, feed_dict={self.actor_skill.state: state})

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
                                for reward, target_critic_predict_q, done in zip(
                                                        rewards, target_critic_predict_qs_moving, dones)])
        self.sess_moving.run(self.train_critic_moving, feed_dict={self.critic_moving.state: states,
                                                    self.critic_moving.action: actions,
                                                    self.target_q: target_qs_moving})

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
                                                    self.target_q: target_qs_skill})

        actions_for_train_skill = self.sess_skill.run(self.actor_skill.action, feed_dict={self.actor_skill.state: states})
        self.sess_skill.run(self.train_actor_skill, feed_dict={self.actor_skill.state: states,
                                                   self.critic_skill.state: states,
                                                   self.critic_skill.action: actions_for_train_skill})
                                                   
        self.sess_skill.run(self.soft_update_target_skill)


    def Make_Summary(self):
        self.summary_rewards = tf.placeholder(tf.float32)
        self.summary_reward1 = tf.placeholder(tf.float32)
        self.summary_reward2 = tf.placeholder(tf.float32)
        tf.summary.scalar("mean rewards", self.summary_rewards)
        tf.summary.scalar("reward1", self.summary_reward1)
        tf.summary.scalar("reward2", self.summary_reward2)
        Summary = tf.summary.FileWriter(
            logdir=save_path, graph=self.sess.graph)
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
        number_of_right_players_agent_controls=num_to_control)

    agent1 = Agent('1')
    agent2 = Agent('2')

    step = 0

    for episode in range(run_episode + test_episode):
        if episode == run_episode:
            train_mode = False

        env.reset()
        episode_rewards1 = 0.0
        episode_rewards2 = 0.0
        observation, reward, done, info = env.step([12, 12])

        state1 = np.concatenate((np.where(observation[0] == 255)[0], np.where(observation[0] == 255)[1]), axis=None)
        state2 = np.concatenate((np.where(observation[1] == 255)[0], np.where(observation[1] == 255)[1]), axis=None)

        active1 = (np.where(observation[0, :, :, 3] == 255)[0][0], np.where(observation[0, :, :, 3] == 255)[1][0])
        active2 = (np.where(observation[1, :, :, 3] == 255)[0][0], np.where(observation[1, :, :, 3] == 255)[1][0])

        while not done:
            step += 1
            print("step: {} | episode: {} | reward1: {:.3f} | reward2: {:.3f}".format(step, episode, episode_rewards1, episode_rewards2), end='\r')

            if step % 10 == 0:
                action1_moving = np.argmax(agent1.get_action_moving(state1))
                action2_moving = np.argmax(agent2.get_action_moving(state2))
            else:
                action1_skill = np.argmax(agent1.get_action_skill(state1)) + moving_action_size
                action2_skill = np.argmax(agent1.get_action_skill(state2)) + moving_action_size

            if step % 10 == 0:
                next_obs, reward, done, info = env.step([action1_moving, action2_moving])
            else:
                next_obs, reward, done, info = env.step([action1_skill, action2_skill])

            reward1 = reward[0]
            reward2 = reward[1]

            next_state1 = np.concatenate((np.where(next_obs[0] == 255)[0], np.where(next_obs[0] == 255)[1]), axis=None)
            next_state2 = np.concatenate((np.where(next_obs[1] == 255)[0], np.where(next_obs[1] == 255)[1]), axis=None)

            now_active1 = (np.where(next_obs[0, :, :, 3] == 255)[0][0], np.where(next_obs[0, :, :, 3] == 255)[1][0])
            now_active2 = (np.where(next_obs[1, :, :, 3] == 255)[0][0], np.where(next_obs[1, :, :, 3] == 255)[1][0])

            if active1 == now_active1:
                reward1 -= 0.1
            elif active1[0] > 72 // 2:
                reward1 += 0.02
                reward2 -= 0.01
                

            if active2 == now_active2:
                reward2 -= 0.1
            elif active2[0] < 72 // 2:
                reward2 += 0.02
                reward1 -= 0.01

            episode_rewards1 += reward1
            episode_rewards2 += reward2
            
            if train_mode:
                agent1.append_sample_moving(state1, action1_moving, reward1, next_state1, done)
                agent1.append_sample_skill(state1, action1_skill, reward1, next_state1, done)
                agent2.append_sample_moving(state2, action2_moving, reward2, next_state2, done)
                agent2.append_sample_skill(state2, action2_skill, reward2, next_state2, done)
            
            state1 = next_state1
            state2 = next_state2
            active1 = now_active1
            active2 = now_active2

            # train_mode 이고 일정 이상 에피소드가 지나면 학습
            if episode > start_train_episode and train_mode and step % 25 == 0:
                agent1.train_model_moving(step)
                agent1.train_model_skill(step)
                agent2.train_model_moving(step)
                agent2.train_model_skill(step)

        # 일정 이상의 episode를 진행 시 log 출력
        if episode % print_interval == 0 and episode != 0:
            print("step: {} | episode: {} | reward1: {:.3f} | reward2: {:.3f}".format(step, episode, episode_rewards1, episode_rewards2))
            agent1.Write_Summray(episode_rewards1, episode)
            agent2.Write_Summray(episode_rewards2, episode)

    env.close()
