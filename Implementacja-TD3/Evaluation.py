import numpy as np

# Runs policy for certain amount of episodes and returns average reward
def evaluate_algorithm(env, policy, episodes_number):

	average_reward = 0.
	for _ in range(episodes_number):
		observation, done = env.reset(), False
		while not done:
			#sometimes array and empty object was returned, so this is a special check to get only vector
			if (len(observation) == 2):
				observation_to_pass = observation[0]
			else:
				observation_to_pass = observation
			action = policy.select_action(np.array(observation_to_pass))
			observation, reward, done, additional_var, additional_var2 = env.step(action)
			average_reward += reward

	average_reward /= episodes_number

	print(f"Average Reward for {episodes_number} episodes: {average_reward:.3f}")
	return average_reward