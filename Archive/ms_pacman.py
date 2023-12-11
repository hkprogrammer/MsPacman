import os
import tensorflow as tf
import gymnasium as gym

import numpy as np
import matplotlib.pyplot as plt
from gymnasium import envs


# print(envs.registry.all())
# TODO: Load an environment

env = gym.make("ALE/MsPacman-ram-v5")

# TODO: Print observation and action spaces
print(env.observation_space)
print(env.action_space)

# TODO Build the policy gradient neural network

class Agent:
    def __init__(self, num_actions, state_size):
        initializer = tf.keras.initializers.glorot_uniform()

        self.input_layer = tf.keras.Input(shape=(state_size,))

        # Neural net starts here

        hidden_layer = tf.keras.layers.Dense(8, activation=tf.nn.relu, kernel_initializer=initializer)(self.input_layer)
        hidden_layer_2 = tf.keras.layers.Dense(8, activation=tf.nn.relu, kernel_initializer=initializer)(hidden_layer)

        # Output of neural net
        dropout2 = tf.keras.layers.Dropout(0.6)(hidden_layer_2)
        out = tf.keras.layers.Dense(num_actions, activation=None)(dropout2)

        self.outputs = tf.nn.softmax(out)
        self.choice = tf.argmax(self.outputs, axis=1)

        # Training Procedure
        self.rewards = tf.keras.Input(shape=(), dtype=tf.float32)
        self.actions = tf.keras.Input(shape=(), dtype=tf.int32)

        one_hot_actions = tf.one_hot(self.actions, num_actions)

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=one_hot_actions)

        self.loss = tf.reduce_mean(cross_entropy * self.rewards)

        self.gradients = tf.gradients(self.loss, tf.trainable_variables())

        # Create a placeholder list for gradients
        self.gradients_to_apply = []
        for index, variable in enumerate(tf.trainable_variables()):
            gradient_placeholder = tf.keras.Input(shape=variable.shape, dtype=tf.float32)
            self.gradients_to_apply.append(gradient_placeholder)

        # Create the operation to update gradients with the gradients placeholder.
        optimizer = tf.keras.optimizers.Adam(learning_rate=2e-2)
        self.update_gradients = optimizer.apply_gradients(zip(self.gradients_to_apply, tf.trainable_variables()))

# TODO Create the discounted and normalized rewards function
discount_rate = 0.95

def discount_normalize_rewards(rewards):
    discounted_rewards = np.zeros_like(rewards)
    total_rewards = 0

    for i in reversed(range(len(rewards))):
        total_rewards = total_rewards * discount_rate + rewards[i]
        discounted_rewards[i] = total_rewards

    discounted_rewards -= np.mean(discounted_rewards)
    discounted_rewards /= np.std(discounted_rewards)

    return discounted_rewards

# Environment setup
env = gym.make('CartPole-v1')

# Loops the entire process, including the testing
for i in range(10):
    # TODO Create the training loop
    tf.keras.backend.clear_session()

    # Modify these to match shape of actions and states in your environment
    num_actions = env.action_space.n
    state_size = env.observation_space.shape[0]

    path = "./cartpole-pg/"

    training_episodes = 50
    max_steps_per_episode = 10000
    episode_batch_size = 10

    agent = Agent(num_actions, state_size)

    checkpoint_dir = os.path.join(path, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    total_episode_rewards = []

    # Create a buffer of 0'd gradients
    gradient_buffer = [tf.zeros_like(var) for var in tf.trainable_variables()]

    for episode in range(training_episodes):

        state = env.reset()

        episode_history = []
        episode_rewards = 0

        for step in range(max_steps_per_episode):

            if episode % 10 == 0:
                env.render()

            # Get weights for each action
            action_probabilities = agent.outputs.numpy()  # Using agent outputs as numpy array
            action_choice = np.random.choice(range(num_actions), p=action_probabilities[0])

            state_next, reward, done, _ = env.step(action_choice)
            episode_history.append([state, action_choice, reward, state_next])
            state = state_next

            episode_rewards += reward

            if done or step + 1 == max_steps_per_episode:
                total_episode_rewards.append(episode_rewards)
                episode_history = np.array(episode_history)
                episode_history[:, 2] = discount_normalize_rewards(episode_history[:, 2])

                with tf.GradientTape() as tape:
                    tape.watch(agent.trainable_variables)
                    logits = agent(self, np.vstack(episode_history[:, 0]))
                    action_masks = tf.one_hot(episode_history[:, 1], num_actions)
                    action_logits = tf.reduce_sum(action_masks * logits, axis=1)
                    loss = -tf.reduce_mean(tf.math.log(action_logits) * episode_history[:, 2])

                gradients = tape.gradient(loss, agent.trainable_variables)
                for index, gradient in enumerate(gradients):
                    gradient_buffer[index] += gradient

                break

        if episode % episode_batch_size == 0:
            optimizer.apply_gradients(zip(gradient_buffer, agent.trainable_variables))
            gradient_buffer = [tf.zeros_like(var) for var in tf.trainable_variables()]

        if episode % 10 == 0:
            checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
            tf.train.save_checkpoint(checkpoint_prefix)

# TODO Create the testing loop
testing_episodes = 1

for episode in range(testing_episodes):
    state = env.reset()
    episode_rewards = 0

    for step in range(max_steps_per_episode):
        env.render()

        action_argmax = np.argmax(agent(state))
        state_next, reward, done, _ = env.step(action_argmax)
        state = state_next

        episode_rewards += reward

        if done or step + 1 == max_steps_per_episode:
            print("Rewards for episode " + str(episode) + ": " + str(episode_rewards))
            break
