import gym

env = gym.make('CartPole-v0')
STATES = env.observation_space
ACTIONS = env.action_space

print(STATES, ACTIONS)

EPISODES = 20
MAX_STEPS = 100

for i_episodes in range(EPISODES):
    observation = env.reset()
    for t in range(MAX_STEPS):
        env.render()
        # print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print(f"Episode ended in {t} timesteps")
            break

env.close()
