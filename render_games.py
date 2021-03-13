import sys
import numpy as np
import gym
import lightgbm as lgbm
import pybullet_envs
import matplotlib.pyplot as plt
from stable_baselines3 import DDPG
import time
import pybullet as p



class wrapper:
    def __init__(self, model):
        self.model = model
    def __call__(self, s):
        return self.model.predict([s])[0]



# train = False
# if train:
#     id = p.connect(p.DIRECT)
# else:
#     id = p.connect(p.GUI)

# print(id)

def eval(env, actor):
    start_time = time.time()
    done = False
    s = env.reset()
    R = 0
    while not done:
        time.sleep(1/60)
        a = actor.predict([s])[0]
        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=0, cameraTargetPosition=[0.0, 0.0, 0.0])
        s, r, done, info = env.step([a])
        # print(a, r)
        R += r
    print(R)
    print(time.time() - start_time, "seconds")



def main():
    seed = np.random.randint(low=0, high=1000000)
    env = gym.make('InvertedPendulumBulletEnv-v0') # 5s 1a
    env.seed(seed)
    env.render(mode="human")
    env.reset()
    time.sleep(15)


    lgbm_actor = lgbm.Booster(model_file='models/best_actor.txt')
    baselines_actor = DDPG.load('models/ddpg_solved')


    if len(sys.argv) != 2:
        print('call this function with ddpg or ddpgbdt as an arg')
        exit(1)
    if sys.argv[1] == 'ddpg':
        actor = baselines_actor
    elif sys.argv[1] == 'ddpgbdt':
        actor = lgbm_actor
    else:
        print('call this function with ddpg or ddpgbdt as an arg')

    for _ in range(5):
        eval(env, actor)
        time.sleep(1)

if __name__ == '__main__':
    main()