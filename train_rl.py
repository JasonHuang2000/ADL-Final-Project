import argparse
from tqdm import tqdm
from typing import List
import numpy as np

import torch
from transformers import set_seed

from policy_gradient import PolicyGradientNetwork, PolicyGradientAgent
from env import ChatEnv

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model', type=str, default='facebook/blenderbot-400M-distill')
    parser.add_argument('--sim_pretrained_model', type=str, default='facebook/blenderbot-400M-distill')
    parser.add_argument('--max_len', type=int, default=64)
    parser.add_argument('--dataset_split', type=str, default='train')
    parser.add_argument('--keywords_file', type=str, default='keywords.json')
    parser.add_argument('--num_batch', type=int, default=500)
    parser.add_argument('--episode_per_batch', type=int, default=4)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--chat_log_file', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=777)

    args = parser.parse_args()

    return args

def compute_acc_reward(imm_rewards: List, gamma: float) -> List:
    acc_rewards = [0] * len(imm_rewards)
    for i in reversed(range(len(imm_rewards))):
        cur_reward = imm_rewards[i]
        for j in reversed(range(i+1)):
            acc_rewards[j] += cur_reward
            cur_reward *= gamma
    return acc_rewards

def main(configs):

    # fix random seed
    set_seed(configs.seed)

    # build policy gradient model/agent
    network = PolicyGradientNetwork(configs).to(configs.device)
    agent = PolicyGradientAgent(network, configs)
    agent.network.train()

    # training loop
    env = ChatEnv(configs)
    pbar = tqdm(range(configs.num_batch))

    if configs.chat_log_file:
        chat_log_f = open(configs.chat_log_file, 'w', encoding='utf-8')

    for b_idx in pbar:

        log_probs, acc_rewards = [], []
        tot_rewards, final_rewards = [], []

        for eps_idx in range(configs.episode_per_batch):

            state, _, _ = env.step(reset=True)
            imm_rewards = []
            tot_reward, tot_step = 0, 0

            if configs.chat_log_file:
                print(f'====================', file=chat_log_f)
                print(f'Episode {b_idx}-{eps_idx}', file=chat_log_f)
                print(f'====================', file=chat_log_f)
                print(f'SIM: {state}', file=chat_log_f)
                chat_log_f.flush()

            while True:

                # sample action
                action, log_prob = agent.sample(state)
                log_probs.append(log_prob)

                # environment step
                next_state, reward, done = env.step(action)
                imm_rewards.append(reward)
                tot_reward += reward
                tot_step += 1
                state = next_state

                if configs.chat_log_file:
                    print(f'BOT: {action}', file=chat_log_f)
                    print(f'SIM: {state}', file=chat_log_f)
                    chat_log_f.flush()

                if done:
                    tot_rewards.append(tot_reward)
                    final_rewards.append(reward)
                    break

            # compute accumulative decaying reward
            acc_rewards.append(compute_acc_reward(imm_rewards, configs.gamma))

        avg_total_reward = sum(tot_rewards) / len(tot_rewards)
        success_count = sum([1 if r > 0 else 0 for r in final_rewards])
        pbar.write(f'Total Reward: {avg_total_reward: 4.1f}, Success Ratio: {success_count}/{configs.episode_per_batch}')

        rewards = np.concatenate(acc_rewards, axis=0)
        rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-9)  # normalize the reward
        loss = agent.learn(torch.stack(log_probs), torch.from_numpy(rewards).to(configs.device))

        pbar.set_postfix({'loss': loss})

    if configs.chat_log_file:
        chat_log_f.close()

if __name__ == '__main__':
    configs = parse_args()
    main(configs)