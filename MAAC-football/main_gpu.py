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
                representation='simple115v2')

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
                                 [115 for _ in range(model.nagents)],
                                 [19 for _ in range(model.nagents)])
    t = 0
    for ep_i in range(0, config["n_episodes"], config["n_rollout_threads"]):
        print("Episodes %i-%i of %i" % (ep_i + 1,
                                        ep_i + 1 + config["n_rollout_threads"],
                                        config["n_episodes"]))

        obs = env.reset()
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
    config["model_name"] = "MAAC_GPU"
    config["n_rollout_threads"] = 4
    config["buffer_length"] = int(1e6)
    config["n_episodes"] = 5000000
    config["episode_length"] = 3000
    config["steps_per_update"] = 100
    config["num_updates"] = 2
    config["batch_size"] = 10240
    config["save_interval"] = 30000
    config["pol_hidden_dim"] = 512
    config["critic_hidden_dim"] = 512
    config["attend_heads"] = 4
    config["pi_lr"] = 0.0000001
    config["q_lr"] = 0.0000001
    config["tau"] = 0.0000005
    config["gamma"] = 0.9999
    config["reward_scale"] = 100.0
    config["use_gpu"] = True

    # Google Football Configure Flags
    config["academy_scenario"] = "11_vs_11_easy_stochastic"
    config["scoring"] = "scoring,checkpoints"
    config["render_mode"] = False
    config["num_to_control"] = 11

    run(config)
