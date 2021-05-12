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

batch_size = 4
mem_maxlen = 1000
discount_factor = 0.99

# actor-critic
actor_lr = 0.0001
critic_lr = 0.0005
tau = 0.001

mu = 0
theta = 0.001
sigma = 0.002

start_train_episode = 0
run_episode = 500
test_episode = 10

print_interval = 1
save_interval = 2

date_time = datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S")

save_path = "./saved_models/" + date_time + "_DDPG"
load_path = ""

# OU_noise 클래스 -> ou noise 정의 및 파라미터 결정
class OU_noise:
    def __init__(self):
        self.reset()

    def reset(self):
        self.X = np.ones(action_size) * mu

    def sample(self):
        dx = theta * (mu - self.X) + sigma * np.random.randn(len(self.X))
        self.X += dx
        return self.X

# Actor 클래스 -> Actor 클래스를 통해 action을 출력
class Actor:
    def __init__(self, model_name):
        self.state = tf.placeholder(shape=[None, state_size[0], state_size[1],
                                           state_size[2]], dtype=tf.float32)
        # 입력을 -1 ~ 1까지 값을 가지도록 정규화
        self.input_normalize = (self.state - (255.0 / 2)) / (255.0 / 2)

        # CNN Network 구축 -> 3개의 Convolutional layer와 2개의 Fully connected layer
        with tf.variable_scope(name_or_scope=model_name):
            self.conv1 = tf.layers.conv2d(inputs=self.input_normalize, filters=32, activation=tf.nn.relu, kernel_size=[8, 8],
                                          strides=[4, 4], padding="SAME")
            self.conv2 = tf.layers.conv2d(inputs=self.conv1, filters=64, activation=tf.nn.relu, kernel_size=[4, 4],
                                          strides=[2, 2], padding="SAME")
            self.conv3 = tf.layers.conv2d(inputs=self.conv2, filters=64, activation=tf.nn.relu, kernel_size=[3, 3],
                                          strides=[1, 1], padding="SAME")

            self.flat = tf.layers.flatten(self.conv3)

            self.fc1 = tf.layers.dense(self.flat, 512, activation=tf.nn.relu)
            self.action = tf.layers.dense(self.fc1, action_size, activation=tf.tanh)

        self.trainable_var = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, model_name)

# Critic 클래스 -> Critic 클래스를 통해 state와 action에 대한 Q-value를 출력
class Critic:
    def __init__(self, model_name):
        self.state = tf.placeholder(shape=[None, state_size[0], state_size[1],
                                           state_size[2]], dtype=tf.float32)
        # 입력을 -1 ~ 1까지 값을 가지도록 정규화
        self.input_normalize = (self.state - (255.0 / 2)) / (255.0 / 2)

        # CNN Network 구축 -> 3개의 Convolutional layer와 2개의 Fully connected layer
        with tf.variable_scope(name_or_scope=model_name):
            self.conv1 = tf.layers.conv2d(inputs=self.input_normalize, filters=32, activation=tf.nn.relu, kernel_size=[8, 8],
                                          strides=[4, 4], padding="SAME")
            self.conv2 = tf.layers.conv2d(inputs=self.conv1, filters=64, activation=tf.nn.relu, kernel_size=[4, 4],
                                          strides=[2, 2], padding="SAME")
            self.conv3 = tf.layers.conv2d(inputs=self.conv2, filters=64, activation=tf.nn.relu, kernel_size=[3, 3],
                                          strides=[1, 1], padding="SAME")

            self.flat = tf.layers.flatten(self.conv3)

            self.fc1 = tf.layers.dense(self.flat, 32, activation=tf.nn.relu)
            self.action = tf.placeholder(tf.float32, [None, action_size])
            self.concat = tf.concat([self.fc1, self.action], axis=-1)
            self.fc2 = tf.layers.dense(self.concat, 128, activation=tf.nn.relu)
            self.fc3 = tf.layers.dense(self.fc2, 128, activation=tf.nn.relu)
            self.predict_q = tf.layers.dense(self.fc3, 1, activation=None)

        self.trainable_var = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, model_name)

# DDPGAgnet 클래스 -> Actor-Critic을 기반으로 학습하는 에이전트 클래스
class DDPGAgent:
    def __init__(self):
        self.actor = Actor("actor")
        self.critic = Critic("critic")
        self.target_actor = Actor("target_actor")
        self.target_critic = Critic("target_critic")

        self.target_q = tf.placeholder(tf.float32, [None, 1])
        critic_loss = tf.losses.mean_squared_error(
            self.target_q, self.critic.predict_q)
        self.train_critic = tf.train.AdamOptimizer(
            critic_lr).minimize(critic_loss)

        action_grad = tf.gradients(tf.squeeze(
            self.critic.predict_q), self.critic.action)
        policy_grad = tf.gradients(
            self.actor.action, self.actor.trainable_var, action_grad)
        for idx, grads in enumerate(policy_grad):
            policy_grad[idx] = -grads/batch_size
        self.train_actor = tf.train.AdamOptimizer(actor_lr).apply_gradients(
            zip(policy_grad, self.actor.trainable_var))

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.Saver = tf.train.Saver()
        self.Summary, self.Merge = self.Make_Summary()
        self.OU = OU_noise()
        self.memory = deque(maxlen=mem_maxlen)

        self.soft_update_target = []
        for idx in range(len(self.actor.trainable_var)):
            self.soft_update_target.append(self.target_actor.trainable_var[idx].assign(
                ((1 - tau) * self.target_actor.trainable_var[idx].value())
                + (tau * self.actor.trainable_var[idx].value())))
        for idx in range(len(self.critic.trainable_var)):
            self.soft_update_target.append(self.target_critic.trainable_var[idx].assign(
                ((1 - tau) * self.target_critic.trainable_var[idx].value())
                + (tau * self.critic.trainable_var[idx].value())))

        init_update_target = []
        for idx in range(len(self.actor.trainable_var)):
            init_update_target.append(self.target_actor.trainable_var[idx].assign(
                                      self.actor.trainable_var[idx]))
        for idx in range(len(self.critic.trainable_var)):
            init_update_target.append(self.target_critic.trainable_var[idx].assign(
                                      self.critic.trainable_var[idx]))
        self.sess.run(init_update_target)

        if load_model == True:
            self.Saver.restore(self.sess, load_path)

    # Actor model에서 action을 예측하고 noise 설정
    def get_action(self, state):
        action = self.sess.run(self.actor.action, feed_dict={
                               self.actor.state: [state]})
        noise = self.OU.sample()
        return np.argmax(action + noise if train_mode else action)

    # replay memory에 입력
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # model 저장
    def save_model(self, episode):
        self.Saver.save(self.sess, save_path + "/model/model" + str(episode))

    # replay memory를 통해 모델을 학습
    def train_model(self):
        mini_batch = random.sample(self.memory, batch_size)
        states = np.asarray([sample[0] for sample in mini_batch])
        actions = np.asarray([sample[1] for sample in mini_batch])
        rewards = np.asarray([sample[2] for sample in mini_batch])
        next_states = np.asarray([sample[3] for sample in mini_batch])
        dones = np.asarray([sample[4] for sample in mini_batch])

        target_actor_actions = self.sess.run(self.target_actor.action,
                                             feed_dict={self.target_actor.state: next_states})
        target_critic_predict_qs = self.sess.run(self.target_critic.predict_q,
                                                 feed_dict={self.target_critic.state: next_states,
                                                            self.target_critic.action: target_actor_actions})
        target_qs = np.asarray([reward + discount_factor * (1 - done) * target_critic_predict_q
                                for reward, target_critic_predict_q, done in zip(
            rewards, target_critic_predict_qs, dones)])
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
        self.summary_reward1 = tf.placeholder(tf.float32)
        self.summary_reward2 = tf.placeholder(tf.float32)
        self.summary_success_cnt = tf.placeholder(tf.float32)
        tf.summary.scalar("reward1", self.summary_reward1)
        tf.summary.scalar("reward2", self.summary_reward2)
        tf.summary.scalar("success_cnt", self.summary_success_cnt)
        Summary = tf.summary.FileWriter(
            logdir=save_path, graph=self.sess.graph)
        Merge = tf.summary.merge_all()

        return Summary, Merge

    def Write_Summray(self, reward1, reward2, success_cnt,  episode):
        self.Summary.add_summary(self.sess.run(self.Merge, feed_dict={
            self.summary_reward1: reward1,
            self.summary_reward2: reward2,
            self.summary_success_cnt: success_cnt}), episode)


# Main 함수
if __name__ == '__main__':
    env = football_env.create_environment(
        env_name='11_vs_11_stochastic',
        rewards='scoring,checkpoints',
        render=render_mode,
        number_of_left_players_agent_controls=agent_num_to_control,
        number_of_right_players_agent_controls=agent_num_to_control)

    # DDPGAgnet 선언
    agent = DDPGAgent()
    rewards = deque(maxlen=print_interval)
    success_cnt = 0
    step = 0

    for episode in range(run_episode + test_episode):
        if episode == run_episode:
            train_mode = False

        env.reset()
        episode_rewards1 = 0.0
        episode_rewards2 = 0.0
        observation, reward, done, info = env.step([0, 0])

        state1 = observation[0]
        state2 = observation[1]
        # state1 = np.concatenate((np.where(observation[0] == 255)[0], np.where(observation[0] == 255)[1]), axis=None)
        # state2 = np.concatenate((np.where(observation[1] == 255)[0], np.where(observation[1] == 255)[1]), axis=None)

        done = False

        while not done:
            step += 1
            print(f"step : {step} | reward1 : {round(episode_rewards1, 4)} | reward2 : {round(episode_rewards2, 4)}", end='\r')

            action1 = agent.get_action(state1)
            action2 = agent.get_action(state2)
            no, reward, done, info = env.step([action1, action2])

            next_state1 = no[0]
            next_state2 = no[1]
            #next_state1 = np.concatenate((np.where(no[0] == 255)[0], np.where(no[0] == 255)[1]), axis=None)
            #next_state2 = np.concatenate((np.where(no[1] == 255)[0], np.where(no[1] == 255)[1]), axis=None)
            episode_rewards1 += reward[0]
            episode_rewards2 += reward[1]

            if train_mode:
                agent.append_sample(state1, action1, reward[0], next_state1, done)
                agent.append_sample(state2, action2, reward[1], next_state2, done)

            state1 = next_state1
            state2 = next_state2

            # train_mode 이고 일정 이상 에피소드가 지나면 학습
            if episode > start_train_episode and train_mode:
                agent.train_model()

        success_cnt = success_cnt + 1 if reward == 2.0 else success_cnt
        rewards.append(episode_rewards1)
        rewards.append(episode_rewards2)

        # 일정 이상의 episode를 진행 시 log 출력
        if episode % print_interval == 0 and episode != 0:
            print("step: {} / episode: {} / reward1: {:.3f} / reward2: {:.3f} / success_cnt: {}".format
                  (step, episode, np.mean(episode_rewards1), np.mean(episode_rewards2), success_cnt))
            agent.Write_Summray(np.mean(rewards), np.mean(episode_rewards1), np.mean(episode_rewards2), success_cnt, episode)
            success_cnt = 0

        # 일정 이상의 episode를 진행 시 현재 모델 저장
        if train_mode and episode % save_interval == 0 and episode != 0:
            print("model saved")
            agent.save_model(episode)

    env.close()
