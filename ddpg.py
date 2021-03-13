import numpy as np
import gym
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

model = DDPG('MlpPolicy', env, verbose=0, seed=0)
# model = TD3('MlpPolicy', env, verbose=1)
# model = SAC('MlpPolicy', env, verbose=1)

obs = env.reset()
gamma = 0.99
R = 0
Rs = []
Ss = []
Ts = []

T = 0
while T < 30000:
    bs = 100
    # eval loop
    R, std = evaluate_policy(model, env, n_eval_episodes=10)
    if R == 1000:
        model.save('models/ddpg_solved')

    print(R)
    Rs.append(R)
    Ss.append(std)
    Ts.append(T)
    model.learn(total_timesteps=bs)
    T += bs

# y = moving_average(Rs, 10)
# x = moving_average(x, 10)
# s = moving_average(stds, 10)

np.savetxt('ddpg_results/ddpg_t.npy', np.array(Ts))
np.savetxt('ddpg_results/ddpg_r.npy', np.array(Rs))
np.savetxt('ddpg_results/ddpg_s.npy', np.array(Ss))

env.close()


