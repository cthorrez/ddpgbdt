import math
import numpy as np
import gym
import lightgbm as lgbm
import pybullet_envs
import matplotlib.pyplot as plt
import copy
from box import Box
from ddpgbdt import GBDT
from ddpgbdt import actor_params
import time

def eval(env, actor):
    done = False
    env.render(mode="human")
    s = env.reset()
    env.render(mode="human")
    R = 0
    while not done:
        time.sleep(1/60)
        a = actor(s)[0]
        s, r, done, info = env.step([a])
        R += r
        env.render(mode="human")
        print(a, r)
    print(R)





def main():
    env = gym.make('InvertedPendulumBulletEnv-v0') # 5s 1a
    s_dim = env.reset().shape[0]
    a_dim = env.action_space.shape[0]

    lgbm_model = lgbm.Booster(model_file='models/best_actor.txt')
    lgbm_actor = GBDT(s_dim, a_dim, params=actor_params)
    lgbm_actor.model = lgbm_model

    eval(env, lgbm_actor)

if __name__ == '__main__':
    main()