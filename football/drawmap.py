import gfootball.env as football_env
import random
import os

env = football_env.create_environment(
		env_name='11_vs_11_stochastic',
		rewards='scoring,checkpoints',
		render=False,
		number_of_left_players_agent_controls=1,
		number_of_right_players_agent_controls=1)
env.reset()
done = False
rewards1 = 0.0
rewards2 = 0.0

while not done:
	actions = env.action_space.sample()
	"""actions = []
	for i in range(11):
		actions.append(random.randrange(0, 18) + 1)"""
	observation, reward, done, info = env.step(actions)

	rewards1 = rewards1 + reward[0]
	rewards2 = rewards2 + reward[1]
	home_players = []
	side_players = []
	ball = None
	active_player = None

	for i in range(72):
		for j in range(96):
			if observation[0, i, j, 0] != 0.0:
				home_players.append([i, j])
	for i in range(72):
		for j in range(96):
			if observation[0, i, j, 1] != 0.0:
				side_players.append([i, j])
	for i in range(72):
		for j in range(96):
			if observation[0, i, j, 2] != 0.0:
				ball = [i, j]

	for i in range(72):
		for j in range(96):
			if observation[0, i, j, 3] != 0.0:
				active_player = [i, j]

	m = [[0 for i in range(96)] for j in range(72)]
	img = [[0 for i in range(96)] for j in range(72)]

	for t in home_players:
		m[t[0]][t[1]] = 1
		img[t[0]][t[1]] = 255

	for t in side_players:
		m[t[0]][t[1]] = 2
		img[t[0]][t[1]] = 255

	m[ball[0]][ball[1]] = 3
	img[ball[0]][ball[1]] = 255

	m[active_player[0]][active_player[1]] = 4
	img[active_player[0]][active_player[1]] = 255
	
	os.system('clear')
	for k in range(72):
		for j in range(96):
			
			if (k == 0) or (k == 71) or (j == 0) or (j == 95):
				print('■ ', end='')
			elif m[k][j] == 0:
				print('  ', end='')
			elif m[k][j] == 1:
				print('● ', end='')
			elif m[k][j] == 2:
				print('○ ', end='')
			elif m[k][j] == 3:
				print('★ ', end='')
			elif m[k][j] == 4:
				print('▲ ', end='')
			
		print()
	print(f"r1 : {rewards1} | r2 : {rewards2} | info : {info}")
	