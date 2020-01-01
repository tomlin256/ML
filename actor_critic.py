import random
from collections import deque

import gym
import keras.backend as K
import numpy as np
import tensorflow.compat.v1 as tf
from keras import losses
from keras.layers import Dense, Input
from keras.layers.merge import Add
from keras.models import Model
from keras.optimizers import Adam
from tqdm import tqdm

tf.disable_v2_behavior()


class ActorCritic:
    def __init__(self, env, sess):
        """ env - a gym.Env like object i.e our environment
            sess - tensor flow session """
        self.env = env
        self.sess = sess

        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.gamma = 0.95
        self.tau = 1.0

        self.memory = deque(maxlen=2000)
        self.train_batch_size = 16

        self.actor_state_input, self.actor_model = self.create_actor_model()
        _, self.target_actor_model = self.create_actor_model()

        self.actor_critic_grad = tf.placeholder(tf.float32, [None, self.env.action_space.shape[0]])

        actor_model_weights = self.actor_model.trainable_weights
        self.actor_grads = tf.gradients(self.actor_model.output,
                                        actor_model_weights,
                                        grad_ys=-self.actor_critic_grad)
        grads = zip(self.actor_grads, actor_model_weights)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)

        self.critic_state_input, self.critic_action_input, self.critic_model = self.create_critic_model()
        _, _, self.target_critic_model = self.create_critic_model()

        self.critic_grads = tf.gradients(self.critic_model.output, self.critic_action_input)

        self.sess.run(tf.initialize_all_variables())

    def create_actor_model(self):
        state_input = Input(shape=self.env.observation_space.shape)
        h1 = Dense(24, activation=K.relu)(state_input)
        h2 = Dense(48, activation=K.relu)(h1)
        h3 = Dense(24, activation=K.relu)(h2)
        output = Dense(self.env.action_space.shape[0], activation=K.relu)(h3)

        model = Model(input=state_input, output=output)
        model.compile(loss=losses.mse, optimizer=Adam(lr=self.learning_rate))
        return state_input, model

    def create_critic_model(self):
        state_input = Input(shape=self.env.observation_space.shape)
        state_h1 = Dense(24, activation=K.relu)(state_input)
        state_h2 = Dense(48)(state_h1)

        action_input = Input(shape=self.env.action_space.shape)
        action_h1 = Dense(48)(action_input)

        merged = Add()([state_h2, action_h1])
        merged_h1 = Dense(24, activation=K.relu)(merged)
        output = Dense(1, activation=K.relu)(merged_h1)

        model = Model(input=[state_input, action_input], output=output)
        model.compile(loss=losses.mse, optimizer=Adam(lr=self.learning_rate))
        return state_input, action_input, model

    def remember(self, cur_state, action, reward, new_state, done):
        self.memory.append([cur_state, action, reward, new_state, done])

    def train(self):
        if len(self.memory) < self.train_batch_size:
            return

        rewards = []
        samples = random.sample(self.memory, self.train_batch_size)
        self._train_critic(samples)
        self._train_actor(samples)

    def _train_critic(self, samples):
        for sample in samples:
            cur_state, action, reward, new_state, done = sample
            if not done:
                target_action = self.target_actor_model.predict(new_state)
                future_reward = self.target_critic_model.predict([new_state, target_action])[0][0]
                reward += self.gamma * future_reward
                self.critic_model.fit([cur_state, action], reward, verbose=0)

    def _train_actor(self, samples):
        for sample in samples:
            cur_state, action, reward, new_state, done = sample
            predicted_action = self.actor_model.predict(cur_state)

            grads = self.sess.run(self.critic_grads,
                                  feed_dict={self.critic_state_input: cur_state,
                                             self.critic_action_input: predicted_action})[0]

            self.sess.run(self.optimize,
                          feed_dict={self.actor_state_input: cur_state,
                                     self.actor_critic_grad: grads})

    def _update_target(self, model, target_model):
        model_weights = model.get_weights()
        target_weights = target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = self.tau * model_weights[i] + (1 - self.tau) * target_weights[i]
        target_model.set_weights(target_weights)

    def update_target(self):
        self._update_target(self.actor_model, self.target_actor_model)
        self._update_target(self.critic_model, self.target_critic_model)

    def act(self, cur_state):
        self.epsilon *= self.epsilon_decay
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return self.actor_model.predict(cur_state)


def main():

    num_episodes = 500
    update_period = 3
    render_every = 2

    sess = tf.Session()
    env = gym.make("Pendulum-v0")

    agent = ActorCritic(env, sess)

    episode_rewards = 0.

    for episode in tqdm(range(num_episodes), ascii=True, unit='episode'):

        cur_state = env.reset()
        action = env.action_space.sample()

        done = False
        counter = 0
        step = 0
        while not done:

            if step % render_every == 0:
                env.render()

            step += 1

            cur_state = cur_state.reshape((1, env.observation_space.shape[0]))
            action = agent.act(cur_state)
            action = action.reshape((1, env.action_space.shape[0]))

            new_state, reward, done, step_info = env.step(action)
            new_state = new_state.reshape((1, env.observation_space.shape[0]))

            agent.remember(cur_state, action, reward, new_state, done)
            agent.train()
            agent.update_target()

            cur_state = new_state

            if done:
                print(f"done on step {step} with info {step_info}")

if __name__ == "__main__":
    main()

