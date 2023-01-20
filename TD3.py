import os
import gym
import argparse
import pybullet_envs
from stable_baselines3 import TD3

# Funkcja uruchamiająca algorytm TD3 z pakietu stable_baselines3 na wybranym środowisku
def learn_model(env_name, learning_rate, buffer_size, batch_size, gamma, total_timesteps):
    model_alg = 'TD3'
    saved_name = f'{env_name}-{model_alg}-lr={learning_rate}-buffer_size={buffer_size}-batch_size={batch_size}-gamma={gamma}-total_timesteps={total_timesteps}'
    model_path = f'models/{saved_name}.zip'

    if not os.path.exists('models'):
        os.makedirs('models')
    if not os.path.exists('TD3_tensorboard'):
        os.makedirs('TD3_tensorboard')
    
    env = gym.make(env_name)

    model = TD3(
        policy='MlpPolicy',
        env=env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        gamma=gamma,
        tensorboard_log='TD3_tensorboard')

    model.learn(
        total_timesteps=total_timesteps,
        tb_log_name=saved_name)
    model.save(model_path)


parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str)
parser.add_argument('--learning_rate', type=float)
parser.add_argument('--buffer_size', type=int)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--gamma', type=float)
parser.add_argument('--total_timesteps', type=int)
args = parser.parse_args()
learn_model(
    env_name=args.env_name,
    learning_rate=args.learning_rate,
    buffer_size=args.buffer_size,
    batch_size=args.batch_size,
    gamma=args.gamma,
    total_timesteps=args.total_timesteps)
