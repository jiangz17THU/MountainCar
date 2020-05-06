import gym
import numpy as np
import matplotlib.pyplot as plt
import time
import seaborn as sns
#环境搭建
env = gym.make("MountainCarContinuous-v0")
env.reset()
#qtable setup shape(200,200,6),action_space.shape(6,1)
LEARNING_RATE = 0.5
DISCOUNT = 0.95
EPISODES = 10000
SHOW_EVERY = 200
Q_TABLE_LEN = 200

action_space = np.array(range(-10,11,4))/10.
action_space = action_space.reshape(len(action_space),1) #to fit with the enviroment
DISCRETE_OS_SIZE = [Q_TABLE_LEN] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE
q_table = np.random.uniform(low=0, high=1,
                            size=(DISCRETE_OS_SIZE + [len(action_space)]))

#e-greedy decayed
epsilon = 1  # not a constant, qoing to be decayed
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES//2
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)  #贪婪系数的损失率，使得贪婪系数由1减到0

#help function
def get_discrete_state (state):
    discrete_state = (state - env.observation_space.low) // discrete_os_win_size
    return tuple(discrete_state.astype(int))

def take_epilon_greedy_action(state, epsilon):  #贪婪策略的实现方式，保证智能体前期能够大胆探索，后期能较快收敛到最优策略
    discrete_state = get_discrete_state(state)
    if np.random.random() < epsilon:
        action_indx = np.random.randint(0,len(action_space))  #如果贪婪系数大于随机数，则随机采取action
    else:                                                     #否则按照已学习的最佳策略的action行动
        action_indx = np.argmax(q_table[discrete_state])
    return action_indx, action_space[action_indx]

#reward recorder setup
ep_rewards = []
aggr_ep_rewards = {'ep':[],'avg':[],'min':[],'max':[]}

#training
for episode in range(EPISODES):
    # initiate reward every episode
    ep_reward = 0
    if episode % SHOW_EVERY == 0:
        print("episode: {}".format(episode))
        render = True
    else:
        render = False

    state = env.reset()
    done = False
    while not done:
        action_indx, action = take_epilon_greedy_action(state, epsilon) #如果没有到达目的地，按照规定的贪婪策略采取行动
        next_state, reward, done, _ = env.step(action)    #环境根据行动反馈下一步状态，奖励值以及是否到达终点
        ep_reward += reward

        # if render:
        #     env.render()

        if not done:                     #如果采取行动后没有到达终点，则按照q-learning的更新公式更新Qtable

            td_target = reward + DISCOUNT * np.max(q_table[get_discrete_state(next_state)])
            q_table[get_discrete_state(state)][action_indx] += LEARNING_RATE * (td_target - q_table[get_discrete_state(state)][action_indx])

        elif next_state[0] >= 0.5:              #到达终点，Q表置零
            # print("I made it on episode: {} Reward: {}".format(episode,reward))
            q_table[get_discrete_state(state)][action_indx] = 0

        state = next_state

    # Decaying is being done every episode if episode number is within decaying range
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

    # recoard aggrated rewards on each epsoide
    ep_rewards.append(ep_reward)

    # every SHOW_EVERY calculate average rewords
    if episode % SHOW_EVERY == 0:
        avg_reward = sum(ep_rewards[-SHOW_EVERY:]) / len(ep_rewards[-SHOW_EVERY:])
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(avg_reward)
        aggr_ep_rewards['min'].append(min(ep_rewards[-SHOW_EVERY:]))
        aggr_ep_rewards['max'].append(max(ep_rewards[-SHOW_EVERY:]))

#绘制rewards-episode曲线
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label = 'avg')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label = 'min')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label = 'max')
plt.legend(loc='upper left')
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.show()


#rendering展示动画

done = False
state = env.reset()
while not done:
    action_indx = np.argmax(q_table[get_discrete_state(state)])
    next_state, _, done, _ = env.step(action_space[action_indx])
    state = next_state
    env.render()
time.sleep(60)
env.close()
