import gym
import keras.backend as K
import numpy as np
from keras.layers import Dense, Input
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.losses import LossFunctionWrapper


class Agent(object):
    """ first agent """

    def __init__(self,
                 env,
                 learning_rate=0.001,
                 discount=0.99,
                 model_file='testing'):

        self.model_file = model_file

        self.env = env

        self.obs_space_size = self.env.observation_space.shape[0]
        if getattr( self.env, 'continuous', False ):
            raise NotImplementedError("continuous env not supported")
        self.action_space_size = self.env.action_space.n
        self.action_space = list(range(self.action_space_size))

        self.learning_rate = learning_rate # alpha
        self.discount = discount # gamma

        self.memory = []

        self.policy = self.build_policy_network()
        self.predict = self.build_predict_network(self.policy.get_weights())

    def _build_network_layers(self):
        state_input = Input(shape=(self.obs_space_size,), name='state')
        d1 = Dense(64, activation=K.relu)(state_input)
        d2 = Dense(64, activation=K.relu)(d1)
        probs = Dense(self.action_space_size, activation=K.softmax)(d2)
        return state_input, probs

    def build_policy_network(self):

        state_input, probs = self._build_network_layers()

        advantages_input = Input(shape=(1,), name='advantages')

        def _loss(y_true, y_pred):
            # y_true = ground truth values
            # y_pred = predicted values
            out = K.clip(y_pred, 1e-8, 1-1e-8) # keep in range for a proba 0.>p<1.
            log_lik = y_true * K.log(out)
            return -log_lik*advantages_input

        policy = Model(input=[state_input, advantages_input], output=[probs])
        policy.compile(optimizer=Adam(lr=self.learning_rate),
                       loss=LossFunctionWrapper(_loss, name="A-weighted-log-loss"))
        return policy

    def build_predict_network(self, weights=None):
        state_input, probs = self._build_network_layers()
        predict = Model(input=[state_input], output=[probs])
        if weights:
            predict.set_weights(weights)
        return predict

    def choose_action(self, observation):
        state = observation[np.newaxis, :]
        probabilities = self.predict.predict(state)[0]
        action = np.random.choice(self.action_space, p=probabilities)
        return action

    def remember(self, observation, action, reward, done):
        self.memory.append((observation, action, reward, done))

    def forget(self):
        self.memory = []
        self.predict = self.build_predict_network(self.policy.get_weights())

    def train(self):

        def _from_memory(i, dtype):
            return np.array([x[i] for x in self.memory], dtype=dtype)

        state_memory = _from_memory(0, np.float)
        action_memory = _from_memory(1, np.int)
        reward_memory = _from_memory(2, np.float)

        # one-hot-encode aka pivot where cols=actions
        actions = np.zeros((len(action_memory), self.action_space_size))
        actions[np.arange(len(action_memory)), action_memory] = 1

        # compute discounted future rewards for each path
        advantages = np.zeros_like(reward_memory)
        for t in range(len(reward_memory)):
            advantage_sum = 0
            discount = 1
            for k in range(t, len(reward_memory)):
                advantage_sum += reward_memory[k] * discount
                discount *= self.discount

            advantages[t] = advantage_sum

        # normalise (taking care not to div0)
        mean = np.mean(advantages)
        std = np.std(advantages) if np.std(advantages) > 0 else 1
        advantages = (advantages-mean)/std

        cost = self.policy.train_on_batch({'state': state_memory,
                                           'advantages': advantages},
                                          actions,)

        return cost

    def save_model(self):
        self.policy.save(self.model_file)

    def load_model(self):
        self.policy = load_model(self.model_file)


def main():

    env = gym.make("CartPole-v0")
    agent = Agent(env)

    score_history = []
    n_episodes = 2000
    render_every = 10

    for episode in range(n_episodes):
        done = False
        score = 0
        observation = env.reset()

        while not done:

            if episode % render_every == 0:
                env.render()

            action = agent.choose_action(observation)
            new_observation, reward, done, info = env.step(action)
            agent.remember(observation, action, reward, done)
            observation = new_observation
            score += reward
        score_history.append(score)

        agent.train()
        agent.forget()

        print(f"episode {episode}, score {score}, average_score {np.mean(score_history[-100:])}")

    agent.save_model()

if __name__ == "__main__":
    main()
