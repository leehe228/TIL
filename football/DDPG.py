import gfootball.env as football_env

import tensorflow as tf
import numpy as np
import random
import datetime
from collections import deque

# 각 팀에서 control 할 agent 수
agent_num_to_control = 1

state_size = [72, 96, 4]
action_size = 19

load_model = False
train_mode = True

render_mode = False

batch_size = 512
mem_maxlen = 100000
discount_factor = 0.99

# actor-critic
actor_lr = 0.0001
critic_lr = 0.0005
tau = 0.001

mu = 0
theta = 0.001
sigma = 0.002

epsilon_init = 1.0
epsilon_min = 0.1

start_train_episode = 100
run_episode = 3000
test_episode = 100

print_interval = 1
save_interval = 300

date_time = datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S")

save_path = "./saved_models/DDPG/" + date_time
load_path = "./saved_models/DDPG/" + date_time + "/model0/model"


class Actor: # Actor Class
    def __init__(self, model_name):
        self.state = tf.placeholder(shape=[None, state_size[0], state_size[1], state_size[2]], dtype=tf.float32)
        self.input_normalize = (self.state - (255.0 / 2)) / (255.0 / 2)

        with tf.variable_scope(name_or_scope=model_name):
            self.conv1 = tf.layers.conv2d(inputs=self.input_normalize, filters=32, activation=tf.nn.relu, kernel_size=[8, 8], strides=[4, 4], padding="SAME")
            self.conv2 = tf.layers.conv2d(inputs=self.conv1, filters=64, activation=tf.nn.relu, kernel_size=[4, 4], strides=[2, 2], padding="SAME")
            self.conv3 = tf.layers.conv2d(inputs=self.conv2, filters=64, activation=tf.nn.relu, kernel_size=[3, 3], strides=[1, 1], padding="SAME")
            self.flat = tf.layers.flatten(self.conv3)
            self.fc1 = tf.layers.dense(self.flat, 128, activation=tf.nn.relu)
            self.fc2 = tf.layers.dense(self.fc1, 128, activation=tf.nn.relu)
            self.action = tf.layers.dense(self.fc2, action_size, activation=tf.tanh)

        self.trainable_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, model_name)


class Critic: # Critic Class
    def __init__(self, model_name):
        self.state = tf.placeholder(shape=[None, state_size[0], state_size[1], state_size[2]], dtype=tf.float32)
        self.input_normalize = (self.state - (255.0 / 2)) / (255.0 / 2)

        with tf.variable_scope(name_or_scope=model_name):
            self.conv1 = tf.layers.conv2d(inputs=self.input_normalize, filters=32, activation=tf.nn.relu, kernel_size=[8, 8], strides=[4, 4], padding="SAME")
            self.conv2 = tf.layers.conv2d(inputs=self.conv1, filters=64, activation=tf.nn.relu, kernel_size=[4, 4], strides=[2, 2], padding="SAME")
            self.conv3 = tf.layers.conv2d(inputs=self.conv2, filters=64, activation=tf.nn.relu, kernel_size=[3, 3], strides=[1, 1], padding="SAME")
            self.flat = tf.layers.flatten(self.conv3)
            self.fc1 = tf.layers.dense(self.flat, 128, activation=tf.nn.relu)
            self.action = tf.placeholder(tf.float32, [None, action_size])
            self.concat = tf.concat([self.fc1, self.action], axis=-1)
            self.fc2 = tf.layers.dense(self.concat, 128, activation=tf.nn.relu)
            self.fc3 = tf.layers.dense(self.fc2, 128, activation=tf.nn.relu)
            self.predict_q = tf.layers.dense(self.fc3, 1, activation=None)

        self.trainable_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, model_name)


class DDPGAgent: # DDPG Agent Class
    def __init__(self):
        self.actor = Actor("actor") 
        self.critic = Critic("critic") 
        self.target_actor = Actor("target_actor") 
        self.target_critic = Critic("target_critic") 

        self.target_q = tf.placeholder(tf.float32, [None, 1]) 
        critic_loss = tf.losses.mean_squared_error(self.target_q, self.critic.predict_q)
        self.train_critic = tf.train.AdamOptimizer(critic_lr).minimize(critic_loss)

        action_grad = tf.gradients(tf.squeeze(self.critic.predict_q), self.critic.action)
        policy_grad = tf.gradients(self.actor.action, self.actor.trainable_var, action_grad)
        for idx, grads in enumerate(policy_grad):
            policy_grad[idx] = -grads/batch_size
        self.train_actor = tf.train.AdamOptimizer(actor_lr).apply_gradients(zip(policy_grad, self.actor.trainable_var))

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.Saver = tf.train.Saver()
        self.Summary, self.Merge = self.Make_Summary()
        self.memory = deque(maxlen=mem_maxlen)

        self.epsilon = epsilon_init

        self.soft_update_target = []
        for idx in range(len(self.actor.trainable_var)):
            self.soft_update_target.append(self.target_actor.trainable_var[idx].assign(((1 - tau) * self.target_actor.trainable_var[idx].value()) + (tau * self.actor.trainable_var[idx].value())))
        for idx in range(len(self.critic.trainable_var)):
            self.soft_update_target.append(self.target_critic.trainable_var[idx].assign(((1 - tau) * self.target_critic.trainable_var[idx].value()) + (tau * self.critic.trainable_var[idx].value())))

        init_update_target = []
        for idx in range(len(self.actor.trainable_var)):
            init_update_target.append(self.target_actor.trainable_var[idx].assign(self.actor.trainable_var[idx]))
        for idx in range(len(self.critic.trainable_var)):
            init_update_target.append(self.target_critic.trainable_var[idx].assign(self.critic.trainable_var[idx]))
        self.sess.run(init_update_target)

        if load_model == True:
            self.Saver.restore(self.sess, load_path)


    def get_action(self, state):
        if self.epsilon > np.random.rand():
            action = env.action_space.sample()
            actions = []
            for a in action:
                temp = [0.0 for k in range(action_size)]
                temp[a] = 1.0
                actions.append(temp)
        else:
            actions = self.sess.run(self.actor.action, feed_dict={self.actor.state: [state]})

        return actions


    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    def save_model(self, episode):
        self.Saver.save(self.sess, save_path + "/model" + str(episode) + "/model")


    def train_model(self, step):
        if step % 100 == 0:
            if self.epsilon > epsilon_min:
                self.epsilon -= 0.3 / (run_episode - start_train_episode)

        mini_batch = random.sample(self.memory, batch_size)
        states = np.asarray([sample[0] for sample in mini_batch])
        actions = np.asarray([sample[1] for sample in mini_batch])
        rewards = np.asarray([sample[2] for sample in mini_batch])
        next_states = np.asarray([sample[3] for sample in mini_batch])
        dones = np.asarray([sample[4] for sample in mini_batch])

        target_actor_actions = self.sess.run(self.target_actor.action, feed_dict={self.target_actor.state: next_states})
        target_critic_predict_qs = self.sess.run(self.target_critic.predict_q, feed_dict={self.target_critic.state: next_states, self.target_critic.action: target_actor_actions})
        target_qs = np.asarray([reward + discount_factor * (1 - done) * target_critic_predict_q for reward, target_critic_predict_q, done in zip(rewards, target_critic_predict_qs, dones)])

        self.sess.run(self.train_critic, feed_dict={self.critic.state: states,
                                                    self.critic.action: actions,
                                                    self.target_q: target_qs})

        actions_for_train = self.sess.run(self.actor.action, feed_dict={
                                          self.actor.state: states})

        self.sess.run(self.train_actor, feed_dict={self.actor.state: states,
                                                   self.critic.state: states,
                                                   self.critic.action: actions_for_train})

        self.sess.run(self.soft_update_target)


    def Make_Summary(self):
        self.summary_rewards = tf.placeholder(tf.float32)
        self.summary_reward1 = tf.placeholder(tf.float32)
        self.summary_reward2 = tf.placeholder(tf.float32)
        tf.summary.scalar("mean rewards", self.summary_rewards)
        tf.summary.scalar("reward1", self.summary_reward1)
        tf.summary.scalar("reward2", self.summary_reward2)
        Summary = tf.summary.FileWriter(logdir=save_path, graph=self.sess.graph)
        Merge = tf.summary.merge_all()

        return Summary, Merge


    def Write_Summray(self, rewards, reward1, reward2, episode):
        self.Summary.add_summary(self.sess.run(self.Merge, feed_dict={
            self.summary_rewards: rewards,
            self.summary_reward1: reward1,
            self.summary_reward2: reward2}), episode)


if __name__ == '__main__':
    env = football_env.create_environment(
        env_name='11_vs_11_stochastic',
        rewards='scoring,checkpoints',
        render=render_mode,
        number_of_left_players_agent_controls=agent_num_to_control,
        number_of_right_players_agent_controls=agent_num_to_control)

    agent = DDPGAgent()
    # rewards = deque(maxlen=print_interval)
    step = 0

    for episode in range(run_episode + test_episode):
        if episode == run_episode: train_mode = False

        env.reset()
        episode_rewards1 = 0.0
        episode_rewards2 = 0.0
        observation, reward, done, info = env.step([8, 8])

        state1 = observation[0]
        state2 = observation[1]
        # state1 = np.concatenate((np.where(observation[0] == 255)[0], np.where(observation[0] == 255)[1]), axis=None)
        # state2 = np.concatenate((np.where(observation[1] == 255)[0], np.where(observation[1] == 255)[1]), axis=None)

        done = False

        while not done:
            step += 1
            print("step: {} | episode: {} | reward1: {:.3f} | reward2: {:.3f} | eps: {:.4f}".format(
                step, episode, episode_rewards1, episode_rewards2, agent.epsilon), end='\r')

            action1 = agent.get_action(state1)
            action2 = agent.get_action(state2)

            no, reward, done, info = env.step(
                [np.argmax(action1), np.argmax(action2)])

            next_state1 = no[0]
            next_state2 = no[1]
            #next_state1 = np.concatenate((np.where(no[0] == 255)[0], np.where(no[0] == 255)[1]), axis=None)
            #next_state2 = np.concatenate((np.where(no[1] == 255)[0], np.where(no[1] == 255)[1]), axis=None)
            episode_rewards1 += reward[0]
            episode_rewards2 += reward[1]

            if train_mode:
                agent.append_sample(
                    state1, action1[0], reward[0], next_state1, done)
                agent.append_sample(
                    state2, action2[0], reward[1], next_state2, done)
            else:
                agent.epsilon = 0.05

            state1 = next_state1
            state2 = next_state2

            # train_mode 이고 일정 이상 에피소드가 지나면 학습
            if episode > start_train_episode and train_mode and step % 25 == 0:
                agent.train_model(step)

        # rewards.append(episode_rewards1)
        # rewards.append(episode_rewards2)

        # 일정 이상의 episode를 진행 시 log 출력
        if episode % print_interval == 0 and episode != 0:
            print("step: {} | episode: {} | reward1: {:.3f} | reward2: {:.3f} | eps: {:.4f}".format(
                step, episode, episode_rewards1, episode_rewards2, agent.epsilon))
            agent.Write_Summray(((episode_rewards1 + episode_rewards2) / 2.0),
                                episode_rewards1, episode_rewards2, episode)

        # 일정 이상의 episode를 진행 시 현재 모델 저장
        if train_mode and episode % save_interval == 0 and episode != 0:
            print("model saved at episode", episode)
            agent.save_model(episode)

    env.close()
