import math
import numpy as np
import numdifftools as nd
import gym
import lightgbm as lgbm
import pybullet_envs
import matplotlib.pyplot as plt
import copy
from box import Box

def evaluate(model, env, num_episodes=10):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_episodes: (int) number of episodes to evaluate it
    :return: (float) Mean reward for the last num_episodes
    """
    # This function will only work for a single Environment
    all_episode_rewards = []
    for i in range(num_episodes):
        episode_rewards = []
        done = False
        obs = env.reset()
        while not done:
            action = model(obs)[0]
            obs, reward, done, info = env.step([action])
            episode_rewards.append(reward)

        all_episode_rewards.append(sum(episode_rewards))
    mean_episode_reward = np.mean(all_episode_rewards)
    std = np.std(all_episode_rewards)
    print("Mean reward:", mean_episode_reward, "Num episodes:", num_episodes)
    return mean_episode_reward, std

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

# epsilon solved it
def finite_differences_gradient(fun, X, epsilon=0.01, idx=-1):
    n, d = X.shape

    eps_matrix = np.zeros_like(X)
    eps_matrix[:,idx] += epsilon

    plus = X + eps_matrix
    minus = X - eps_matrix

    stacked = np.vstack([plus, minus])
    results = fun(stacked)

    plus_results = results[:n]
    minus_results = results[n:]

    grads = ((plus_results - minus_results) / 2) / epsilon
    return grads

actor_params = {
    'params': {
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'min_child_samples' : 1,
        'verbose': -1,
        'seed' : 0,
    },
    'num_boost_round': 2
}
actor_params = Box(actor_params)

critic_params = {
    'params': {
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'min_child_samples' : 1,
        'objective': 'regression',
        'verbose': -1,
        'seed' : 0,
    },
    'num_boost_round': 2
}
critic_params = Box(critic_params)


class GBDT():
    def __init__(self, in_dim, out_dim, params=None, fobj=None, action_scale=None):
        self.model = None
        self.fobj = fobj
        self.params = params
        dummy_X = (np.random.rand(2, in_dim) - 0.5) * 2
        dummy_y = np.zeros(2)
        self.train(dummy_X, dummy_y)
        self.action_scale = action_scale

    def __call__(self, x):
        if len(x.shape) == 1:
            x = x[None,:]
        preds = self.model.predict(x)
        if self.action_scale is not None:
            preds = np.tanh(preds) * self.action_scale
        return preds

    def train(self, X, y):
        if self.fobj is not None:
            self.fobj.set_data(X)
        dataset = lgbm.Dataset(X, y)
        self.model = lgbm.train(self.params.params,
                                fobj=self.fobj,
                                init_model=self.model,
                                train_set=dataset,
                                keep_training_booster=True,
                                num_boost_round=self.params.num_boost_round)

class Buffer():
    def __init__(self, max_size, s_dim, a_dim):
        self.max_size = max_size
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.s_buffer = np.zeros((max_size, s_dim), dtype=np.float32)
        self.a_buffer = np.zeros((max_size, a_dim), dtype=np.float32)
        self.s_p_buffer = np.zeros((max_size, s_dim), dtype=np.float32)
        self.r_buffer = np.zeros((max_size), dtype=np.float32)
        self.done_buffer = np.zeros((max_size), dtype=np.float32)
        self.size = 0
        self.next = 0

    def add_tuple(self, s, a, r, s_p, done):
        insert_idx = self.next % self.max_size
        self.s_buffer[insert_idx,:] = s
        self.a_buffer[insert_idx,:] = a
        self.s_p_buffer[insert_idx,:] = s_p
        self.r_buffer[insert_idx] = r
        self.done_buffer[insert_idx] = done
        self.next += 1
        self.size = max(self.size, insert_idx + 1)

    def sample(self, batch_size):
        batch_size = min(batch_size, self.size)
        idxs = np.random.randint(low=0, high=self.size, size=(batch_size,))
        s = self.s_buffer[idxs,:]
        a = self.a_buffer[idxs,:]
        s_p = self.s_p_buffer[idxs,:]
        r = self.r_buffer[idxs]
        done = self.done_buffer[idxs]
        return s, a, r, s_p, done


class CustomLoss():
    def __init__(self, critic):
        self.critic = critic

    def set_data(self, X):
        self.X = X

    def __call__(self, preds, data):
        '''preda is action prediction from actor
        actual data doesn't matter lol, there is no label
        '''
        X = np.hstack([self.X, preds[:,None]]) # new
        grad1 = -1 * finite_differences_gradient(self.critic, X, idx=-1)

        print('mean gradient size:', np.abs(grad1).mean())

        hess = np.ones_like(grad1)
        return grad1, hess


def main():
    seed = 0 # good
    seed = 0
    np.random.seed(seed)
    # env = gym.make('MountainCarContinuous-v0') # 2s 1a
    # env = gym.make('Pendulum-v0') # 3s 1a
    
    # maybe...
    # env = gym.make('CartPoleContinuousBulletEnv-v0') # 4s 1a

    # this one definitely learns from 25 -> 500
    # this one did solve
    env = gym.make('InvertedPendulumBulletEnv-v0') # 5s 1a
    
    # this one I can get to almost -20
    # env = gym.make('CartPoleSwingUp-v0') # 5s 1a

    # env = gym.make('InvertedPendulumSwingupBulletEnv-v0') # 5s 1a

    env.seed(seed)
    s_dim = env.reset().shape[0]
    a_dim = env.action_space.shape[0]
    print('state dim:', s_dim, 'action dim:', a_dim)
    print(env.action_space)
    action_scale = env.action_space.high[0]
    
    critic = GBDT(in_dim=(s_dim+a_dim), out_dim=1, params=critic_params)
    replay_memory = Buffer(max_size=60000, s_dim=s_dim, a_dim=a_dim) # 60k winner
    loss = CustomLoss(critic)
    actor = GBDT(in_dim=s_dim, out_dim=a_dim, params=actor_params, fobj=loss, action_scale=action_scale)

    gamma = 0.99
    # gamma = 1.0
    batch_size = 20000 # good~ 500
    batch_size = 50000 # winner?

    train_every = 600
    eval_every = train_every

    warmup = 0
    timesteps = 0
    train_timesteps = 400000 # winner
    train_timesteps = 350000

    eps = 0.75
    min_eps = 0.2
    eps_decay = 0.99

    # eps= 0.2
    # eps_decay = 1.0


    best_model = None
    best_R = -np.inf

    s = env.reset()
    Rs = []
    Ts = []
    Ss = []
    while timesteps < train_timesteps:
        explore = (np.random.rand() < eps) or (timesteps < warmup)
        if explore:
            a = np.random.uniform(low=env.action_space.low[0], high=env.action_space.high[0])
        else:
            a = actor(s)[0]
        s_p, r, done, info = env.step([a])
        if done:
            r = -10
        replay_memory.add_tuple(s, a, r, s_p, done)

        if (timesteps % train_every == 0) and (timesteps > 0):

            s_batch, a_batch, r_batch, s_p_batch, done_batch = replay_memory.sample(batch_size)

            # update critic
            X_critic = np.hstack([s_batch, a_batch])
            actor_preds = actor(s_p_batch)
            if len(actor_preds.shape) == 1:
                actor_preds = actor_preds[:,None]

            critic_preds = critic(np.hstack([s_p_batch, actor_preds]))
            y_critic = r_batch + (1.0 - done_batch) * gamma * critic_preds

            # mse = np.power(critic_preds - y_critic, 2).mean()
            # print('critic mse:', mse)

            critic.train(X_critic, y_critic)

            # update actor
            X_actor = s_batch

            # actor label isn't used
            y_actor = np.zeros(X_actor.shape[0])

            actor.train(X_actor, y_actor)

            if (timesteps > warmup) and (eps > min_eps):
                eps =  eps * eps_decay
                print('epsilon', eps)


            # eval
            if timesteps % eval_every == 0:
                eval_r, eval_std = evaluate(actor, env, num_episodes=15)

                if eval_r > best_R:
                    best_R = eval_r
                    print('saving new best model')
                    actor.model.save_model('models/best_actor.txt')

                prev = 0 if len(Rs) == 0 else Rs[-1]
                if ((prev - eval_r) / prev) > 0.25:
                    print('bad update, reverting')
                    actor.model.rollback_one_iter()
                    critic.model.rollback_one_iter()
                    actor_params.params['learning_rate'] *= 0.75 # I think I need these lines, idk
                    critic_params.params['learning_rate'] *= 0.75
                else:
                    Rs.append(eval_r)
                    Ts.append(timesteps)
                    Ss.append(eval_std)

        timesteps += 1
        s = s_p
        if done:
            s = env.reset()


    # r = moving_average(Rs, 5)
    # t = moving_average(Ts, 5)
    # s = moving_average(Ss, 5)

    np.savetxt('ddpgbdt_results/ddpgbdt_t.npy', Ts)
    np.savetxt('ddpgbdt_results/ddpgbdt_r.npy', Rs)
    np.savetxt('ddpgbdt_results/ddpgbdt_s.npy', Ss)

    

if __name__ == '__main__':
    main()