import numpy as np
import gym
import gym_cartpole_swingup
import pybullet_envs
from stable_baselines3 import DDPG, SAC, TD3
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

# env = gym.make('CartPoleSwingUp-v0')
# env = gym.make('CartPoleContinuousBulletEnv-v0') # 4s 1a
# env = gym.make('MountainCarContinuous-v0') # 2s 1a
# env = gym.make('Pendulum-v0') # 3s 1a
env = gym.make('InvertedPendulumBulletEnv-v0')

model = DDPG('MlpPolicy', env, verbose=1)
# model = TD3('MlpPolicy', env, verbose=1)
# model = SAC('MlpPolicy', env, verbose=1)

obs = env.reset()
gamma = 0.99
R = 0
Rs = []

T = 0
while T < 100000:
    bs = 500
    # eval loop
    R, _ = evaluate_policy(model, env, n_eval_episodes=15)
    print(R)
    Rs.append(R)
    model.learn(total_timesteps=bs)
    T += bs

y = moving_average(Rs, 5)
x = np.arange(len(y))
plt.plot(x, y)
plt.savefig('ddpg_{}.png'.format(env.spec.id))

env.close()




def evaluate(model, num_episodes=100):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_episodes: (int) number of episodes to evaluate it
    :return: (float) Mean reward for the last num_episodes
    """
    # This function will only work for a single Environment
    env = model.get_env()
    all_episode_rewards = []
    for i in range(num_episodes):
        episode_rewards = []
        done = False
        obs = env.reset()
        while not done:
            # _states are only useful when using LSTM policies
            action, _states = model.predict(obs)
            # here, action, rewards and dones are arrays
            # because we are using vectorized env
            obs, reward, done, info = env.step(action)
            episode_rewards.append(reward)

        all_episode_rewards.append(sum(episode_rewards))

    mean_episode_reward = np.mean(all_episode_rewards)
    print("Mean reward:", mean_episode_reward, "Num episodes:", num_episodes)

    return mean_episode_reward