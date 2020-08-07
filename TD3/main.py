import gym
import numpy as np
from agent import Agent
import matplotlib.pyplot as plt

if __name__ == '__main__':
    n_games = 1000

    env = gym.make('LunarLanderContinuous-v2')
    agent = Agent(input_dims=env.observation_space.shape,
                  n_actions=env.action_space.shape[0],
                  env=env,
                  fc1_dims=400,
                  fc2_dims=300,
                  alpha=0.001,
                  beta=0.001,
                  gamma=0.99,
                  tau=0.005,
                  noise1=0.1,
                  noise2=0.2,
                  clamp=0.5,
                  delay=2,
                  max_size=1000000,
                  batch_size=100,
                  warmup=1000)

    # agent.load_models()

    best_score = env.reward_range[0]
    scores, averages = [], []
    for n in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            agent.remember(observation, action, reward, observation_, done)
            agent.learn()
            score += reward
            observation = observation_

        scores.append(score)
        average = np.mean(scores[-100:])
        averages.append(average)

        if average > best_score:
            best_score = average
            # agent.save_models()

        print(f'{n} score: {round(score,2)}, average: \
{round(average,2)}, best: {round(best_score,2)}')

    plt.figure(figsize=(10,5))
    plt.plot(scores, label='scores')
    plt.plot(averages, label='average')
    plt.legend()
    plt.show()


