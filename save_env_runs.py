#!/usr/bin/env python3

import gym
import time

log_file='data/rollouts.txt'


def reward_to_str(obs):
    return_str = 'obs: '
    for e in obs:
        return_str += f'{e:.1f} '

    return return_str


if __name__ == '__main__':
    start_time = time.time()
    env = gym.make('CartPole-v0')
    for episode in range(int(1e9)):
        observation = env.reset()
        episode_str = ''
        reward_sum = 0
        for _ in range(100):
            # env.render()
            action = env.action_space.sample()
            obs_str = reward_to_str(observation)



            observation, reward, done, _ = env.step(action) # take a random action
            reward_sum += reward

            # episode_str += f'{obs_str} reward_sum: xxx reward: {reward}  action: {action}   '
            episode_str += f'{obs_str} reward: {reward}  action: {action}   '
            if done:
                break

        if False:
            episode_str = episode_str.replace('xxx', str(int(reward_sum)))
        else:
            episode_str = f'reward_sum: {int(reward_sum)} ' + episode_str

        episodes_per_second = episode / (time.time() - start_time)
        if episode % 2000 == 0:
            print("episodes_per_second:", episodes_per_second)
        # print("episode_str:", episode_str)
        if reward_sum > 80:
            with open(log_file, 'a') as f:
                f.write(episode_str + '\n')


    env.close()
