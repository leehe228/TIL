import gfootball.env as football_env

env = football_env.create_environment(
		env_name='11_vs_11_stochastic',
		render=True,
		number_of_left_players_agent_controls=11,
		number_of_right_players_agent_controls=0)
env.reset()
done = False

while not done:
	actions = []
	for i in range(11):
		actions.append(env.action_space.sample())

	observation, reward, done, info = env.step(actions)