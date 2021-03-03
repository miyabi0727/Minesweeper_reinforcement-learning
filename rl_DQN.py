import gym
from keras.models import Model
from keras.layers import *
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
import matplotlib.pyplot as plt
import numpy as np
# from rl import agents

from my_env2 import Env


env = Env()
# # ゲームを作成
# print("action_space      : " + str(env.action_space))
# print("observation_space : " + str(env.observation_space))
# print("reward_range      : " + str(env.reward_range))

# # 入力と出力
window_length = 1
input_shape = (window_length,) + env.observation_space.shape
# print(input_shape)
nb_actions = env.action_space.n

# NNモデルを作成
c = input_ = Input(input_shape)
c = Flatten()(c)
c = Dense(16, activation='relu')(c)
c = Dense(16, activation='relu')(c)
c = Dense(16, activation='relu')(c)
c = Dense(16, activation='relu')(c)
c = Dense(nb_actions, activation='linear')(c)
model = Model(input_, c)
# #print(model.summary())  # modelを表示

# # DQNAgentを使うための準備
memory = SequentialMemory(limit=50000, window_length=window_length)
policy = BoltzmannQPolicy()
agent = DQNAgent(model=model, nb_actions=nb_actions, gamma=0.3, batch_size=16, memory=memory, nb_steps_warmup=1, target_model_update=0.1, policy=policy)
agent.compile(Adam())

# 最終的なmodelを表示
# print(agent.model.summary())

agent.load_weights('model/DQN_v4_agent.h5')

# 訓練
print("--- start ---")
print("'Ctrl + C' is stop.")
history = agent.fit(env, nb_steps=50000, visualize=False, verbose=1)

agent.save_weights('model/DQN_v4_agent.h5')
# model.save_weights('DQN_v2_model.h5')

import pdb; pdb.set_trace()

# 結果を表示
history.paramsplt.subplot(2,1,1)
plt.plot(history.history["nb_episode_steps"])
plt.ylabel("step")

plt.subplot(2,1,2)
plt.plot(history.history["episode_reward"])
plt.xlabel("episode")
plt.ylabel("reward")

plt.show()

# 訓練結果を見る
# agent.test(env, nb_episodes=5, visualize=True)

