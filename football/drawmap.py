import gfootball.env as football_env
import numpy as np
import pygame
import math

def calc_diff_ball(player, ball):
	return math.sqrt((player[0] - ball[0])**2 + (player[1] - ball[1])**2)


def have_ball(player, ball):
	return player == ball

if __name__ == "__main__":
	
	pygame.init()

	xnum = 8

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
	screen = pygame.display.set_mode([96*xnum, 72*xnum])
	step = 0
	myFont = pygame.font.Font(None, 30)
	
	while not done:
		step += 1
		actions = env.action_space.sample()
		observation, reward, done, info = env.step(actions)

		rewards1 = rewards1 + reward[0]
		rewards2 = rewards2 + reward[1]

		p1 = np.where(observation[0, :, :, 0] == 255.0)
		p2 = np.where(observation[0, :, :, 1] == 255.0)
		ball = np.where(observation[0, :, :, 2] == 255.0)
		active1 = np.where(observation[0, :, :, 3] == 255.0)
		active2 = np.where(observation[1, :, :, 3] == 255.0)
		
		now_active1 = (np.where(next_obs[0, :, :, 3] != 0)[0][0], np.where(next_obs[0, :, :, 3] != 0)[1][0])
		now_active2 = (np.where(next_obs[1, :, :, 3] != 0)[0][0], np.where(next_obs[1, :, :, 3] != 0)[1][0])
		now_ball = (np.where(observation[0, :, :, 2] != 0)[0][0], np.where(observation[0, :, :, 2] != 0)[1][0])

		screen.fill((255, 255, 255))
		z = 1
		for i1, j1, i2, j2 in zip(p1[0], p1[1], p2[0], p2[1]):
			numberling1 = myFont.render(str(z), True, (100, 100, 255))
			numberling2 = myFont.render(str(z), True, (255, 100, 100))
			# pygame.draw.circle(screen, (100, 100, 255), [i1*xnum, j1*xnum], 2, 2)
			# pygame.draw.circle(screen, (255, 100, 100), [i2*xnum, j2*xnum], 2, 2)
			screen.blit(numberling1, [j1*xnum, i1*xnum])
			screen.blit(numberling2, [j2*xnum, i2*xnum])
			z += 1

		pygame.draw.circle(screen, (0, 0, 0), [ball[1][0]*xnum, ball[0][0]*xnum], 4, 4)
		pygame.draw.circle(screen, (0, 0, 255), [active1[1][0]*xnum, active1[0][0]*xnum], 4, 4)
		pygame.draw.circle(screen, (255, 0, 0), [active2[1][0]*xnum, active2[0][0]*xnum], 4, 4)

		pygame.display.flip()
		print("step : {} | A : {} | B : {} | diff1 : {} | diff2 : {}".format(step, have_ball(now_active1, now_ball), have_ball(now_active2, now_ball), calc_diff_ball(now_active1, now_ball), calc_diff_ball(now_active2, now_ball)), end='\r')
		