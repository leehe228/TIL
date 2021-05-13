import gfootball.env as football_env
import numpy as np

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
	s1 = np.where(observation[0] == 255)[0]
	s2 = np.where(observation[0] == 255)[1]
	state = np.concatenate((s1, s2), axis=None)
	print(state)
	
	

	
