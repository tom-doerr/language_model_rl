#!/usr/bin/env python3

import gym

log_file='data/rollouts.txt'


def reward_to_str(obs):
    return_str = 'obs: '
    for e in obs:
        return_str += f'{e:.1f} '

    return return_str


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    for i in range(int(1e6)):
        observation = env.reset()
        episode_str = ''
        for _ in range(100):
            # env.render()
            action = env.action_space.sample()
            obs_str = reward_to_str(observation)



            observation, reward, done, _ = env.step(action) # take a random action

            episode_str += f'{obs_str} reward: {reward}  action: {action}   '

        print("episode_str:", episode_str)
        with open(log_file, 'a') as f:
            f.write(episode_str + '\n')


    env.close()
