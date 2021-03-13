import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 15})

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def main():
    window = 5
    ddpgbdt_t = np.loadtxt('ddpgbdt_results/ddpgbdt_t.npy')
    ddpgbdt_r = np.loadtxt('ddpgbdt_results/ddpgbdt_r.npy')
    ddpgbdt_s = np.loadtxt('ddpgbdt_results/ddpgbdt_s.npy')
    ddpgbdt_t = moving_average(ddpgbdt_t, window)
    ddpgbdt_r = moving_average(ddpgbdt_r, window)
    ddpgbdt_s = moving_average(ddpgbdt_s, window)


    window = 5
    ddpg_t = np.loadtxt('ddpg_results/ddpg_t.npy')
    ddpg_r = np.loadtxt('ddpg_results/ddpg_r.npy')
    ddpg_s = np.loadtxt('ddpg_results/ddpg_s.npy')
    ddpg_t = moving_average(ddpg_t, window)
    ddpg_r = moving_average(ddpg_r, window)
    ddpg_s = moving_average(ddpg_s, window)

    # plot1 = plt.figure(1)
    
    plt.plot(ddpg_t, ddpg_r, color='orange', label='DDPG')
    plt.fill_between(ddpg_t, ddpg_r-ddpg_s, ddpg_r+ddpg_s, alpha = 0.5, color='orange')
    # plt.show()

    # plot2 = plt.figure()
    plt.plot(ddpgbdt_t, ddpgbdt_r, color='cornflowerblue', label='DDPGBDT')
    plt.fill_between(ddpgbdt_t, ddpgbdt_r-ddpgbdt_s, ddpgbdt_r+ddpgbdt_s, alpha = 0.5, color='cornflowerblue')

    plt.legend(loc = 'upper')
    plt.xlabel('Environment Timesteps')
    plt.ylabel('Episodic Reward')
    plt.title('InvertedPendulumBulletEnv-v0')
    
    
    
    plt.show()


if __name__ == '__main__':
    main()