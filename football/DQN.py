import gfootball.env as football_env

import numpy as np
import random
import datetime
import time
import tensorflow as tf
from collections import deque

state_size = [72, 96, 4]
action_size = 18

load_model = False
train_mode = True

batch_size = 128
mem_maxlen = 50000
discount_factor = 0.99
learning_rate = 0.00025

run_episode = 60000
test_episode = 5000

start_train_episode = 10000

target_update_step = 1000
print_interval = 10
save_interval = 500

epsilon_init = 1.0
epsilon_min = 0.1

date_time = datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S")

save_path = "./saved_models/"+ date_time + "_DQN"
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
                                          activation=tf.nn.relu, kernel_size=[8,8], 
                                          strides=[4,4], padding="SAME")
            self.conv2 = tf.layers.conv2d(inputs=self.conv1, filters=64, 
                                          activation=tf.nn.relu, kernel_size=[4,4],
                                          strides=[2,2],padding="SAME")
            self.conv3 = tf.layers.conv2d(inputs=self.conv2, filters=64, 
                                          activation=tf.nn.relu, kernel_size=[3,3],
                                          strides=[1,1],padding="SAME")
 
            self.flat = tf.layers.flatten(self.conv3)

            self.fc1 = tf.layers.dense(self.flat,512,activation=tf.nn.relu)
            self.Q_Out = tf.layers.dense(self.fc1, action_size, activation=None)
        self.predict = tf.argmax(self.Q_Out, 1)

        self.target_Q = tf.placeholder(shape=[None, action_size], dtype=tf.float32)

        # 손실함수 값 계산 및 네트워크 학습 수행 
        self.loss = tf.losses.huber_loss(self.target_Q, self.Q_Out)
        self.UpdateModel = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        self.trainable_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, model_name)


# DQNAgent 클래스 -> DQN 알고리즘을 위한 다양한 함수
class DQNAgent():
    def __init__(self):
        self.model = Model("Q")
        self.target_model = Model("target")

        self.memory = deque(maxlen=mem_maxlen)

        self.sess = tf.Session()
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

        self.epsilon = epsilon_init

        self.Saver = tf.train.Saver()
        self.Summary, self.Merge = self.Make_Summary()

        if load_model == True:
            self.Saver.restore(self.sess, load_path)

    # Epsilon greedy 기법에 따라 액션 결정
    def get_action(self, state):
        if self.epsilon > np.random.rand():
            # 랜덤하게 액션 결정
            random_action = np.random.randint(0, action_size)
            return random_action
        else:
            # 네트워크 연산에 따라 액션 결정
            predict = self.sess.run(self.model.predict,
                                    feed_dict={self.model.input: [state]})
            return np.asscalar(predict)

    # 리플레이 메모리에 데이터 추가 (observation, reward, done, info)
    def append_sample(self, data):
        self.memory.append((data[0], data[1], data[2], data[3], data[4]))

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
        target_val = self.sess.run(target_model.Q_Out,
                                   feed_dict={target_model.input: next_states})

        for i in range(batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + \
                    discount_factor * np.amax(target_val[i])

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
        self.summary_loss = tf.placeholder(dtype=tf.float32)
        self.summary_reward = tf.placeholder(dtype=tf.float32)

        tf.summary.scalar("loss", self.summary_loss)
        tf.summary.scalar("reward", self.summary_reward)

        Summary = tf.summary.FileWriter(
            logdir=save_path, graph=self.sess.graph)
        Merge = tf.summary.merge_all()

        return Summary, Merge

    def Write_Summray(self, reward, loss, episode):
        self.Summary.add_summary(
            self.sess.run(self.Merge, feed_dict={self.summary_loss: loss,
                                                 self.summary_reward: reward}), episode)


if __name__ == "__main__":
	env = football_env.create_environment(
		env_name='11_vs_11_stochastic',
		render=True)

	agent = DQNAgent()

	step = 0

	rewards = []
	losses = []

	for episode in range(run_episode + test_episode):
		if episode == run_episode:
			train_mode = False

		env.reset()
		done = False

		observation, reward, done, info = env.step(0)
		episode_rewards = 0

		while not done:
			step += 1
			action = agent.get_action(observation)

			next_observation, reward, done, info = env.step(action)

			if train_mode:
				data = [observation, action, reward, next_observation, done]
				agent.append_sample(data)
			else:
				time.sleep(0.01)
				agent.epsilon = 0.05

			observation = next_observation

			if episode > start_train_episode and train_mode:
				loss = agent.train_model(
				    agent.model, agent.target_model, agent.memory, done)
				losses.append(loss)

				if step % (target_update_step) == 0:
					agent.update_target(agent.model, agent.target_model)

		rewards.append(episode_rewards)

		# 게임 진행 상황 출력 및 텐서 보드에 보상과 손실함수값 기록
		if episode % print_interval == 0 and episode != 0:
			print("step: {} / episode: {} / epsilon: {:.3f}".format(step, episode, agent.epsilon))
			print(
			    "reward: {:.2f} / loss: {:.4f}".format(np.mean(rewards), np.mean(losses)))
			print('------------------------------------------------------------')
			agent.Write_Summray(np.mean(rewards), np.mean(losses), episode)
			rewards = []
			losses = []

        # 네트워크 모델 저장
		if episode % save_interval == 0 and episode != 0:
			agent.save_model()
			print("Save Model {}".format(episode))

	exit(0)
