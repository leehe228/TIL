import gfootball.env as football_env

env = football_env.create_environment(
        env_name="11_vs_11_easy_stochastic",
        rewards="scoring,checkpoints",
        render=False,
        number_of_left_players_agent_controls=11,
        representation='simple115v2')
env.reset()
done = False

while not done:
    actions = env.action_space.sample()

    observation, reward, done, info = env.step(actions)

    print(observation.shape)
    print(reward.shape)
    print(done)