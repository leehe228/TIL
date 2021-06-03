#-*-coding:utf-8-*-
#!/usr/bin/env python3

from pathlib import Path
import os
from algorithms.attention_sac import AttentionSAC
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from utils.buffer import ReplayBuffer
from utils.make_env import make_env
import numpy as np
from tensorboardX import SummaryWriter
from torch.autograd import Variable
import torch
import gfootball.env as football_env
from gym.spaces import Box, Discrete

# disable logger casting warning
import gym
gym.logger.set_level(40)


def make_state(obs):
    state = []

    for p in range(3):
        o = obs[p]
        tempList = []
        # activate and designated
        if o['active'] == p:
            tempList.append(1)
        else:
            tempList.append(0)

        if o['designated'] == p:
            tempList.append(1)
        else:
            tempList.append(0)    

        # home team
        tempList.append(o['left_team_tired_factor'][p])
        tempList.append(o['left_team'][p][0])
        tempList.append(o['left_team'][p][1])

        for l in range(len(o['left_team'])):
            if l == p: continue
            else:
                tempList.append(o['left_team'][l][0])
                tempList.append(o['left_team'][l][1])

        tempList.append(o['left_team_direction'][p][0])
        tempList.append(o['left_team_direction'][p][1])

        for l in range(len(o['left_team_direction'])):
            if l == p: continue
            else:
                tempList.append(o['left_team_direction'][l][0])
                tempList.append(o['left_team_direction'][l][1])
        
        # side team
        for r in o['right_team']:
            tempList.append(r[0])
            tempList.append(r[1])
        for r in o['right_team_direction']:
            tempList.append(r[0])
            tempList.append(r[1])

        # ball
        tempList.append(o['ball'][0])
        tempList.append(o['ball'][1])
        tempList.append(o['ball'][2])
        tempList.append(o['ball_rotation'][0])
        tempList.append(o['ball_rotation'][1])
        tempList.append(o['ball_rotation'][2])
        tempList.append(o['ball_direction'][0])
        tempList.append(o['ball_direction'][1])
        tempList.append(o['ball_direction'][2])
        tempList.append(o['ball_owned_team'])

        # etc
        tempList.append(o['score'][0])
        tempList.append(o['score'][1])
        tempList.append(o['game_mode'])

        state.append(np.array(tempList))

    state = np.array(state)
    return state


def make_parallel_env(n_rollout_threads, seed):
    def get_env_fn(rank):
        def init_env():
            # (** EDITED **) Google Football Env
            # Gym Env 설정 함수를 변경
            env = football_env.create_environment(
                env_name=config["academy_scenario"],
                rewards=config["scoring"],
                render=config["render_mode"],
                number_of_left_players_agent_controls=config["num_to_control"],
                representation='raw')

            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])


def run(config):
    model_dir = Path('./models') / config["env_id"] / config["model_name"]
    if not model_dir.exists():
        run_num = 1
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            run_num = 1
        else:
            run_num = max(exst_run_nums) + 1
    curr_run = 'run%i' % run_num
    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs'
    os.makedirs(log_dir)
    logger = SummaryWriter(str(log_dir))

    torch.manual_seed(run_num)
    np.random.seed(run_num)
    env = make_parallel_env(config["n_rollout_threads"], run_num)
    model = AttentionSAC.init_from_env(env,
                                       tau=config["tau"],
                                       pi_lr=config["pi_lr"],
                                       q_lr=config["q_lr"],
                                       gamma=config["gamma"],
                                       pol_hidden_dim=config["pol_hidden_dim"],
                                       critic_hidden_dim=config["critic_hidden_dim"],
                                       attend_heads=config["attend_heads"],
                                       reward_scale=config["reward_scale"])
    # (** EDITED **) Set Replay Buffer
    # env.action_space, env.observation_space 의 shape를 iteration을 통해 버퍼 설정
    replay_buffer = ReplayBuffer(config["buffer_length"], model.nagents,
                                 [40 for _ in range(model.nagents)],
                                 [19 for _ in range(model.nagents)])
    t = 0
    for ep_i in range(0, config["n_episodes"], config["n_rollout_threads"]):
        print("Episodes %i-%i of %i" % (ep_i + 1,
                                        ep_i + 1 + config["n_rollout_threads"],
                                        config["n_episodes"]))

        obs = env.reset()
        obs = np.array([make_state(o) for o in obs])
        model.prep_rollouts(device='cpu')

        for et_i in range(config["episode_length"]):
            print("episode : {} | step : {}".format(ep_i, et_i), end='\r')
            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [Variable(torch.Tensor(
                np.vstack(obs[:, i])), requires_grad=False) for i in range(model.nagents)]
            # get actions as torch Variables
            torch_agent_actions = model.step(torch_obs, explore=True)
            # convert actions to numpy arrays
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
            # rearrange actions to be per environment
            actions = [[ac[i] for ac in agent_actions]
                       for i in range(config["n_rollout_threads"])]

            # Reform Actions list to fit on Football Env
            # Google Football 환경은 액션 리스트 (one hot encoded)가 아닌 정수값을 받음
            actions_list = [[np.argmax(b) for b in a] for a in actions]

            # Step
            next_obs, rewards, dones, infos = env.step(actions_list)
            next_obs = np.array([make_state(n_o) for n_o in next_obs])
            
            # Prevention of divergence
            # 안해주면 발산해서 학습 불가 (NaN)
            rewards = rewards - 0.000001

            # Reform Done Flag list
            # replay buffer에 알맞도록 done 리스트 재구성
            dones = (np.array([dones for _ in range(model.nagents)])).T

            replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)
            obs = next_obs
            t += config["n_rollout_threads"]
            if (len(replay_buffer) >= config["batch_size"] and (t % config["steps_per_update"]) < config["n_rollout_threads"]):
                if config["use_gpu"]:
                    model.prep_training(device='gpu')
                else:
                    model.prep_training(device='cpu')
                for u_i in range(config["num_updates"]):
                    sample = replay_buffer.sample(
                        config["batch_size"], to_gpu=config["use_gpu"])
                    model.update_critic(sample, logger=logger)
                    model.update_policies(sample, logger=logger)
                    model.update_all_targets()
                model.prep_rollouts(device='cpu')
        ep_rews = replay_buffer.get_average_rewards(
            config["episode_length"] * config["n_rollout_threads"])
        for a_i, a_ep_rew in enumerate(ep_rews):
            logger.add_scalar('agent%i/mean_episode_rewards' % a_i,
                              a_ep_rew * config["episode_length"], ep_i)

        if ep_i % config["save_interval"] < config["n_rollout_threads"]:
            model.prep_rollouts(device='cpu')
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            model.save(run_dir / 'incremental' /
                       ('model_ep%i.pt' % (ep_i + 1)))
            model.save(run_dir / 'model.pt')

    model.save(run_dir / 'model.pt')
    env.close()
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()


if __name__ == '__main__':
    config = dict()

    config["env_id"] = "football"
    config["model_name"] = "MAAC3"
    config["n_rollout_threads"] = 2
    config["buffer_length"] = int(1e6)
    config["n_episodes"] = 1000000
    config["episode_length"] = 128
    config["steps_per_update"] = 100
    config["num_updates"] = 4
    config["batch_size"] = 4096
    config["save_interval"] = 1000
    config["pol_hidden_dim"] = 128
    config["critic_hidden_dim"] = 128
    config["attend_heads"] = 4
    config["pi_lr"] = 0.00001
    config["q_lr"] = 0.00001
    config["tau"] = 0.00005
    config["gamma"] = 0.99
    config["reward_scale"] = 100.0
    config["use_gpu"] = True

    # Google Football Configure Flags
    config["train_episode"] = 1000
    config["academy_scenario"] = "academy_3_vs_1_with_keeper"
    config["scoring"] = "scoring,checkpoints"
    config["render_mode"] = False
    config["num_to_control"] = 3

    run(config)
