import gfootball.env as football_env
from gym.spaces import Box, Discrete

import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import numpy as np

from utils.make_env import make_env
from utils.buffer import ReplayBuffer
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from algorithms.attention_sac import AttentionSAC

import os
from pathlib import Path


def make_parallel_env(env_id, n_rollout_threads, seed):
    def get_env_fn(rank):
        def init_env():
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
    env = make_parallel_env(config["env_id"], config["n_rollout_threads"], run_num)
    model = AttentionSAC.init_from_env(env,
                                       tau=config["tau"],
                                       pi_lr=config["pi_lr"],
                                       q_lr=config["q_lr"],
                                       gamma=config["gamma"],
                                       pol_hidden_dim=config["pol_hidden_dim"],
                                       critic_hidden_dim=config["critic_hidden_dim"],
                                       attend_heads=config["attend_heads"],
                                       reward_scale=config["reward_scale"])
    replay_buffer = ReplayBuffer(config["buffer_length"], model.nagents,
                                 [115 for _ in range(11)],
                                 [19 for _ in range(11)])
    t = 0
    for ep_i in range(0, config["n_episodes"], config["n_rollout_threads"]):
        print("Episodes %i-%i of %i" % (ep_i + 1,
                                        ep_i + 1 + config["n_rollout_threads"],
                                        config["n_episodes"]))
        obs = env.reset()
        model.prep_rollouts(device='cpu')

        done = [False]
        et_i = 0

        while not any(done):
            et_i += 1
            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                                  requires_grad=False)
                         for i in range(model.nagents)]
            # get actions as torch Variables
            torch_agent_actions = model.step(torch_obs, explore=True)
            # convert actions to numpy arrays
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
            # rearrange actions to be per environment
            actions = [[ac[i] for ac in agent_actions] for i in range(config["n_rollout_threads"])]
            
            actions_list = []
            for a in actions:
                temp = []
                for b in a:
                    temp.append(np.argmax(b))
                actions_list.append(temp)

            next_obs, rewards, done, infos = env.step(actions_list)

            dones = [done for _ in range(11)]

            replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)
            obs = next_obs
            t += config["n_rollout_threads"]
            if (len(replay_buffer) >= config["batch_size"] and
                (t % config["steps_per_update"]) < config["n_rollout_threads"]):
                if config["use_gpu"]:
                    model.prep_training(device='gpu')
                else:
                    model.prep_training(device='cpu')
                for u_i in range(config["num_updates"]):
                    sample = replay_buffer.sample(config["batch_size"],
                                                  to_gpu=config["use_gpu"])
                    model.update_critic(sample, logger=logger)
                    model.update_policies(sample, logger=logger)
                    model.update_all_targets()
                model.prep_rollouts(device='cpu')

            print("ep_i : {} | et_i : {}".format(ep_i, et_i), end='\r')

        ep_rews = replay_buffer.get_average_rewards(config["episode_length"] * config["n_rollout_threads"])
        
        for a_i, a_ep_rew in enumerate(ep_rews):
            logger.add_scalar('agent%i/mean_episode_rewards' % a_i,
                              a_ep_rew * config["episode_length"], ep_i)

        if ep_i % config["save_interval"] < config["n_rollout_threads"]:
            model.prep_rollouts(device='cpu')
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            model.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1)))
            model.save(run_dir / 'model.pt')

    model.save(run_dir / 'model.pt')
    env.close()
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()


if __name__ == '__main__':
    config = dict()

    config["env_id"] = "football"
    config["model_name"] = "MAAC"
    config["n_rollout_threads"] = 8
    config["buffer_length"] = int(1e6)
    config["n_episodes"] = 100000
    config["episode_length"] = 3000
    config["steps_per_update"] = 100
    config["num_updates"] = 4
    config["batch_size"] = 1024
    config["save_interval"] = 1000
    config["pol_hidden_dim"] = 256
    config["critic_hidden_dim"] = 256
    config["attend_heads"] = 4
    config["pi_lr"] = 0.000001
    config["q_lr"] = 0.000001
    config["tau"] = 0.000005
    config["gamma"] = 0.999
    config["reward_scale"] = 100.0
    config["use_gpu"] = False

    config["academy_scenario"] = "11_vs_11_easy_stochastic"
    config["scoring"] = "scoring,checkpoints"
    config["render_mode"] = False
    config["num_to_control"] = 11

    run(config)