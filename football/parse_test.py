import gfootball.env as football_env
import numpy as np

env = football_env.create_environment(
    env_name='11_vs_11_easy_stochastic',
    render=False,
    number_of_left_players_agent_controls=1,
    number_of_right_players_agent_controls=1,
    representation='simple115v2')
env.reset()
done = False

while not done:
    actions = env.action_space.sample()
    observation, reward, done, info = env.step(actions)
    
    for i in observation[0]:
        print(i, end=', ')