#!/usr/bin/env python
# coding: utf-8
import os
import argparse
import numpy as np
import gym
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate, Conv3D
from keras.optimizers import Adam
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from keras.callbacks import TensorBoard
from gym_sensor_delay_wrapper.delay_env import DelayEnv


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
ENV_NAME = 'Pendulum-v0'
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--delay_steps", required=True,
                help="delay in steps")
ap.add_argument("-g", "--gpu", required=True,
                help="GPU number")
args = vars(ap.parse_args())
os.environ["CUDA_VISIBLE_DEVICES"] = args["gpu"]
is_save = False
# Get the environment and extract the number of actions.
base_env = gym.make(ENV_NAME)
delay_steps = int(args['delay_steps'])
delayed_env = DelayEnv(base_env, reset_action=np.array([0.]),
                       delay_steps=delay_steps)
delayed_env.seed(123456)
assert len(base_env.action_space.shape) == 1
nb_actions = base_env.action_space.shape[0]
base_env.observation_space.shape
# Next, we build a very simple model.
actor = Sequential()
actor.add(Flatten(input_shape=(1,) + base_env.observation_space.shape))
actor.add(Dense(16))
actor.add(Activation('relu'))
actor.add(Dense(16))
actor.add(Activation('relu'))
actor.add(Dense(16))
actor.add(Activation('relu'))
actor.add(Dense(nb_actions))
actor.add(Activation('linear'))
print(actor.summary())
action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(1,) + base_env.observation_space.shape, name='observation_input')
flattened_observation = Flatten()(observation_input)
x = Concatenate()([action_input, flattened_observation])
x = Dense(32)(x)
x = Activation('relu')(x)
x = Dense(32)(x)
x = Activation('relu')(x)
x = Dense(32)(x)
x = Activation('relu')(x)
x = Dense(1)(x)
x = Activation('linear')(x)
critic = Model(inputs=[action_input, observation_input], outputs=x)
print(critic.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=100000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.3)
agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                  random_process=random_process, gamma=.99, target_model_update=1e-3)
agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
tbCallback = TensorBoard(log_dir='./logs'+'/pendulum_delay_' + str(delay_steps) , histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')

if is_save:
    agent.fit(delayed_env, nb_steps=500000, visualize=False, verbose=1, nb_max_episode_steps=200, callbacks=[tbCallback])
else:
    agent.fit(delayed_env, nb_steps=500000, visualize=False, verbose=1, nb_max_episode_steps=200)
if is_save:
    actor.save('logs/actor_pendulum_delay_' + str(delay_steps) + '.h5')
    critic.save('logs/critic_pendulum_delay_' + str(delay_steps) + '.h5')

# Finally, evaluate our algorithm for 5 episodes.
#agent.test(env, nb_episodes=5, visualize=False, nb_max_episode_steps=200)

