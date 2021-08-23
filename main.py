#!/usr/bin/env python3

import gym
from save_env_runs import reward_to_str
from configs import *
from transformers import pipeline

NUM_STEPS = 100
NUM_EPISODES = int(1e3)
AGENT_TYPE = 'lm'

env = gym.make('CartPole-v0')


fill_mask = pipeline(
    "fill-mask",
    model=TOKENIZER_DIR,
    tokenizer=TOKENIZER_DIR
)


episodes_reward_sum = 0
for i in range(NUM_EPISODES):
    observation = env.reset()
    episode_str = ''
    reward_sum = 0
    for step_num in range(100):
        print("step_num:", step_num)
        # env.render()
        obs_str = reward_to_str(observation)
        if AGENT_TYPE == 'random':
            action = env.action_space.sample()
        elif AGENT_TYPE == 'lm':
            episode_str_input = episode_str + f'{obs_str} reward: 1  action: <mask>'
            completions = fill_mask(episode_str_input[-200:])
            # print("completion:", completion)
            action = None
            for completion in completions:
                try:
                    print("completion['token_str']:", completion['token_str'])
                    action = int(completion['token_str'])
                    assert action == 0 or action == 1 or action == 2
                    break
                except (ValueError, RuntimeError, AssertionError) as e:
                    if type(e) == RuntimeError:
                        print(e)
                    print("type(e):", type(e))
                    print(e)
                    pass
            if action == None:
                action = env.action_space.sample()
                print('Taking random action')
            print("action:", action)


        observation, reward, done, _ = env.step(action) # take a random action
        reward_sum += reward

        episode_str += f'{obs_str} reward: {reward}  action: {action}   '

    print("episode_str:", episode_str)
    episodes_reward_sum += reward_sum
    print("reward_sum:", reward_sum)

avg_episode_reward = episodes_reward_sum / NUM_EPISODES
print("avg_episode_reward:", avg_episode_reward)


env.close()
