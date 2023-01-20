import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import Actor
import Critic


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TD3(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		discount,
		tau,
		policy_noise,
		noise_clip,
		policy_freq,
		learning_rate
	):

		self.actor = Actor.Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr = learning_rate)

		self.critic = Critic.Critic(state_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr = learning_rate)

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq

		self.total_iterations = 0

	#Choosing action by actor
	def select_action(self, state):
		observation = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor(observation[0]).cpu().data.numpy().flatten()


	def train(self, replay_buffer, batch_size):
		self.total_iterations += 1

		# Sample replay buffer 
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

		# Adding noise and updating actor and critic networks
		with torch.no_grad():
			noise = (
				torch.randn_like(action) * self.policy_noise
			).clamp(-self.noise_clip, self.noise_clip)

			next_action = (
				self.actor_target(next_state) + noise
			).clamp(-self.max_action, self.max_action)

			target_Q1, target_Q2 = self.critic_target(next_state, next_action)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = reward + not_done * self.discount * target_Q

		current_Q1, current_Q2 = self.critic(state, action)

		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		if self.total_iterations % self.policy_freq == 0:
			
			#Calculate loss function
			actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)