import gfootball.env as football_env

import numpy as np
import random
import datetime
import time
import tensorflow as tf
from collections import deque

state_size = [72, 96, 4]
action_size = 19

load_model = False
train_mode = True

batch_size = 32
mem_maxlen = 50000
discount_factor = 0.99
learning_rate = 0.00025

run_episode = 1000
test_episode = 100

start_train_episode = 0

target_update_step = 2
print_interval = 1
save_interval = 10

epsilon_init = 1.0
epsilon_min = 0.1

date_time = datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S")

save_path = "./saved_models/" + date_time + "_DQN"
load_path = "./saved_models/model/model"

# Model 클래스 -> 네트워크 정의 및 손실함수 설정, 네트워크 최적화 알고리즘 결정


class Model():
    def __init__(self, model_name):
        self.input = tf.placeholder(shape=[None, state_size[0], state_size[1],
                                           state_size[2]], dtype=tf.float32)
        # 입력을 -1 ~ 1까지 값을 가지도록 정규화
        self.input_normalize = (self.input - (255.0 / 2)) / (255.0 / 2)

        # CNN Network 구축 -> 3개의 Convolutional layer와 2개의 Fully connected layer
        with tf.variable_scope(name_or_scope=model_name):
            self.conv1 = tf.layers.conv2d(inputs=self.input_normalize, filters=32,
                                          activation=tf.nn.relu, kernel_size=[
                                              8, 8],
                                          strides=[4, 4], padding="SAME")
            self.conv2 = tf.layers.conv2d(inputs=self.conv1, filters=64,
                                          activation=tf.nn.relu, kernel_size=[
                                              4, 4],
                                          strides=[2, 2], padding="SAME")
            self.conv3 = tf.layers.conv2d(inputs=self.conv2, filters=64,
                                          activation=tf.nn.relu, kernel_size=[
                                              3, 3],
                                          strides=[1, 1], padding="SAME")

            self.flat = tf.layers.flatten(self.conv3)

            self.fc1 = tf.layers.dense(self.flat, 512, activation=tf.nn.relu)
            self.Q_Out = tf.layers.dense(self.fc1, action_size, activation=None)
        self.predict = tf.argmax(self.Q_Out, 1)

        self.target_Q = tf.placeholder(
            shape=[None, action_size], dtype=tf.float32)

        # 손실함수 값 계산 및 네트워크 학습 수행
        self.loss = tf.losses.huber_loss(self.target_Q, self.Q_Out)
        self.UpdateModel = tf.train.AdamOptimizer(
            learning_rate).minimize(self.loss)
        self.trainable_var = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, model_name)


# DQNAgent 클래스 -> DQN 알고리즘을 위한 다양한 함수
class DQNAgent():
    def __init__(self):
        self.model1 = Model("Q1")
        self.target_model1 = Model("target1")
        self.model2 = Model("Q2")
        self.target_model2 = Model("target2")

        self.memory1 = deque(maxlen=mem_maxlen)
        self.memory2 = deque(maxlen=mem_maxlen)

        self.sess = tf.Session()
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

        self.epsilon = epsilon_init

        self.Saver = tf.train.Saver()
        self.Summary, self.Merge = self.Make_Summary()

        if load_model == True:
            self.Saver.restore(self.sess, load_path)

    # Epsilon greedy 기법에 따라 액션 결정
    def get_action(self, state1, state2):
        if self.epsilon > np.random.rand():
            # 랜덤하게 액션 결정
            actions = env.action_space.sample()
            return actions
        else:
            # 네트워크 연산에 따라 액션 결정
            predict1 = self.sess.run(self.model1.predict,
                                    feed_dict={self.model1.input: state1})
            predict2 = self.sess.run(self.model2.predict,
                                    feed_dict={self.model2.input: state2})
            return [np.asscalar(predict1), np.asscalar(predict2)]

    # 리플레이 메모리에 데이터 추가
    def append_sample(self, data1, data2):
        # state, action, reward, next state, done
        self.memory1.append((data1[0], data1[1], data1[2], data1[3], data1[4]))
        self.memory2.append((data2[0], data2[1], data2[2], data2[3], data2[4]))

    # 네트워크 모델 저장
    def save_model(self):
        self.Saver.save(self.sess, save_path + "/model/model")

    # 학습 수행
    def train_model(self, model, target_model, memory, done):
        # Epsilon 값 감소
        if done:
            if self.epsilon > epsilon_min:
                self.epsilon -= 0.5 / (run_episode - start_train_episode)

        # DQN의 학습을 위한 미니 배치 데이터 샘플링
        mini_batch = random.sample(memory, batch_size)

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for i in range(batch_size):
            states.append(mini_batch[i][0])
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_states.append(mini_batch[i][3])
            dones.append(mini_batch[i][4])

        # DQN의 학습을 위한 타겟값 계산
        target = self.sess.run(model.Q_Out, feed_dict={model.input: states})
        target_val = self.sess.run(target_model.Q_Out, feed_dict={target_model.input: next_states})

        for i in range(batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + discount_factor * np.amax(target_val[i])
    
        # DQN 알고리즘 학습 수행 및 loss 계산
        _, loss = self.sess.run([model.UpdateModel, model.loss],
                                feed_dict={model.input: states,
                                           model.target_Q: target})
	
        return loss

    # 타겟 네트워크 업데이트
    def update_target(self, model, target_model):
        for i in range(len(model.trainable_var)):
            self.sess.run(target_model.trainable_var[i].assign(
                model.trainable_var[i]))

    # 텐서 보드에 기록할 값 설정 및 데이터 기록
    def Make_Summary(self):
        self.summary_loss1 = tf.placeholder(dtype=tf.float32)
        self.summary_reward1 = tf.placeholder(dtype=tf.float32)
        self.summary_loss2 = tf.placeholder(dtype=tf.float32)
        self.summary_reward2 = tf.placeholder(dtype=tf.float32)

        tf.summary.scalar("loss1", self.summary_loss1)
        tf.summary.scalar("reward1", self.summary_reward1)
        tf.summary.scalar("loss2", self.summary_loss2)
        tf.summary.scalar("reward2", self.summary_reward2)

        Summary = tf.summary.FileWriter(
            logdir=save_path, graph=self.sess.graph)
        Merge = tf.summary.merge_all()

        return Summary, Merge

    def Write_Summray(self, reward1, loss1, reward2, loss2, episode):
        self.Summary.add_summary(
            self.sess.run(self.Merge, feed_dict={self.summary_loss1: loss1,
                                                 self.summary_reward1: reward1,
                                                 self.summary_loss2: loss2,
                                                 self.summary_reward2: reward2}), episode)


if __name__ == "__main__":
    env = football_env.create_environment(
        env_name='11_vs_11_stochastic',
        rewards='scoring,checkpoints',
        render=False,
        number_of_left_players_agent_controls=1,
        number_of_right_players_agent_controls=1)

    agent = DQNAgent()

    step = 0

    rewards1 = []
    losses1 = []
    rewards2 = []
    losses2 = []

    for episode in range(run_episode + test_episode):
        if episode == run_episode:
            train_mode = False

        env.reset()
        done = False

        observation, reward, done, info = env.step([0, 0])
        episode_rewards1 = 0.0
        episode_rewards2 = 0.0

        while not done:
            step += 1
            print(f"step : {step}", end='\r')

            action = agent.get_action(observation[0], observation[1])

            next_observation, reward, done, info = env.step(action)
            episode_rewards1 += reward[0]
            episode_rewards2 += reward[1]

            if train_mode:
                data1 = [observation[0], action[0], reward[0], next_observation[0], done]
                data2 = [observation[1], action[1], reward[1], next_observation[1], done]
                agent.append_sample(data1, data2)
            else:
                time.sleep(0.01)
                agent.epsilon = 0.05

            observation = next_observation

            if episode > start_train_episode and train_mode:
                loss1 = agent.train_model(
                    agent.model1, agent.target_model1, agent.memory1, done)
                loss2 = agent.train_model(
                    agent.model2, agent.target_model2, agent.memory2, done)
                losses1.append(loss1)
                losses2.append(loss2)

                if step % (target_update_step) == 0:
                    agent.update_target(agent.model1, agent.target_model1)
                    agent.update_target(agent.model2, agent.target_model2)

        rewards1.append(episode_rewards1)
        rewards2.append(episode_rewards2)

        # 게임 진행 상황 출력 및 텐서 보드에 보상과 손실함수값 기록
        if episode % print_interval == 0 and episode != 0:
            print("step: {} / episode: {} / epsilon: {:.3f}".format(step, episode, agent.epsilon))
            print("reward1: {:.2f} | loss1: {:.4f} | reward2: {:.2f} | loss2: {:.4f}".format(np.mean(rewards1), np.mean(losses1), np.mean(rewards2), np.mean(losses2)))
            print('------------------------------------------------------------')

            agent.Write_Summray(np.mean(rewards1), np.mean(losses1), np.mean(rewards2), np.mean(losses2), episode)
            rewards1 = []
            losses1 = []
            rewards2 = []
            losses2 = []

        # 네트워크 모델 저장
        if episode % save_interval == 0 and episode != 0:
            agent.save_model()
            print("Save Model {}".format(episode))

    exit(0)
