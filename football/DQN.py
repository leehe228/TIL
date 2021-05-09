import gfootball.env as football_env

env = football_env.create_environment(
		env_name='11_vs_11_stochastic',
		render=True)
env.reset()
done = False

while not done:
	action = env.action_space.sample()
	observation, reward, done, info = env.step(action)