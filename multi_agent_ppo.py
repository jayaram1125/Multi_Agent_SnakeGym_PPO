import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import gym
import numpy as np
from video_recorder import VideoRecorder
import envpool

import random
from typing import Optional
import os
import pygame

import cv2
from mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar,num_procs,setup_mpi_gpus
from mpi4py import MPI

import faulthandler

faulthandler.enable()

BLACK = (0,0,0) #background 

DARK_GREEN = (0,100,0)#Snake head color
GREEN = (0, 255, 0) #Snake body color

RED = (255, 0, 0) #Fruit color
BROWN = (156,102,31) #Grid color

DARK_BLUE = (39,64,139) #Snake2 head color
BLUE = (135,206,250) #Snake2 body color 

COOPERATIVE = 4
COMPETITIVE = 5


#Action masking
class CategoricalMasked(Categorical):
	def __init__(self, logits, mask: Optional[torch.Tensor] = None):
		self.mask = mask
		self.sz = mask.size(dim=0)
		if self.mask is None:
			super(CategoricalMasked, self).__init__(logits=logits)
		else:
			num_envs,num_actions = logits.size()
			self.boolean_mask  = torch.ones((self.sz,num_actions),dtype=torch.bool,device=logits.device)
			for i in range(0,self.sz):
				for j in range(0,num_actions):
					if j == self.mask[i]:
						self.boolean_mask[i][j] = False
			self.mask_value = torch.tensor(torch.finfo(logits.dtype).min,dtype=logits.dtype ,device=logits.device)
			self.logits = torch.where(self.boolean_mask, logits, self.mask_value)
			super(CategoricalMasked, self).__init__(logits=self.logits)
	def entropy(self):
		if self.mask is None:
			return super().entropy()
		p_log_p = self.probs*self.logits
		p_log_p = torch.where(self.boolean_mask,p_log_p,torch.tensor(0, dtype=p_log_p.dtype, device=p_log_p.device))
		return -torch.sum(p_log_p, dim= 1)



def layer_init(m,std=np.sqrt(2)):
	#print("within init_weights_and_biases")
	nn.init.orthogonal_(m.weight,std)
	nn.init.constant_(m.bias.data,0)
	return m


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)

    def forward(self, x):
        inputs = x
        x = nn.functional.relu(x)
        x = self.conv0(x)
        x = nn.functional.relu(x)
        x = self.conv1(x)
        return x + inputs


class ConvSequence(nn.Module):
    def __init__(self, input_shape, out_channels):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=self._input_shape[0], out_channels=self._out_channels, kernel_size=3, padding=1)
        self.res_block0 = ResidualBlock(self._out_channels)
        self.res_block1 = ResidualBlock(self._out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res_block0(x)
        x = self.res_block1(x)
        assert x.shape[1:] == self.get_output_shape()
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        return (self._out_channels, (h + 1) // 2, (w + 1) // 2)

class MLPActorCritic(nn.Module):
	def __init__(self):
		super(MLPActorCritic, self).__init__()
		shape = (1, 84, 84)
		conv_seqs = []
		for out_channels in [16, 32, 32]:
			conv_seq = ConvSequence(shape, out_channels)
			shape = conv_seq.get_output_shape()
			conv_seqs.append(conv_seq)
		conv_seqs += [
			nn.Flatten(),
			nn.ReLU(),
			nn.Linear(in_features=shape[0] * shape[1] * shape[2], out_features=256),
			nn.ReLU(),
		]
		self.network = nn.Sequential(*conv_seqs)
		self.actor = layer_init(nn.Linear(256, 4), std=0.01)
		self.critic = layer_init(nn.Linear(256, 1), std=1)

	def step(self,obs,masked_action,a=None,grad_condition=False):
		with torch.set_grad_enabled(grad_condition):
			logits = self.actor(self.network(obs.permute((0, 3, 1, 2)) / 255.0))
			pi = CategoricalMasked(logits,masked_action)
			if a == None:
				a = pi.sample()
				logp_a = pi.log_prob(a)
			else:
				logp_a = pi.log_prob(a)
			v = torch.squeeze(self.critic(self.network(obs.permute((0, 3, 1, 2)) / 255.0)),-1)
		return a,v,logp_a,pi.entropy()




class MULTIPPO: 
	def __init__(self):
		self.envType = COMPETITIVE
		self.num_envs  = 16
		self.num_updates = 20000
		self.num_timesteps = 128
		self.gamma = 0.99
		self.lamda = 0.95
		self.mini_batch_size = 512
		self.learning_rate = 0.0002
		self.clip_coef = 0.2
		self.entropy_coef=0.01
		self.value_coef= 0.5
		self.max_grad_norm =0.5
		self.update = 0
		self.epochs = 4
		self.episode_length = 0

		self.snakeids = ["snake1","snake2"]
		self.total_reward_d = {"snake1":0.0 ,"snake2":0.0}



	def capped_cubic_video_schedule(self) -> bool:
		return True


	def render(self,fruit_position,snake1_head_position,snake1_body_positions,snake2_head_position,snake2_body_positions,
		display_size=84,scale=2.1,body_width=8.4):

		surf = pygame.Surface((display_size,display_size))
		surf.fill(BLACK)
		pygame.transform.scale(surf, (scale,scale))
		
		pygame.draw.rect(surf,BROWN,pygame.Rect(0,0,display_size,display_size),int(body_width))

		if fruit_position[0] !=0 and fruit_position[1] != 0:
			pygame.draw.rect(surf,RED,pygame.Rect(fruit_position[0]*scale-body_width/2,fruit_position[1]*scale-body_width/2,body_width,body_width))
		
		if snake1_head_position[0] !=0 and snake1_head_position[1] != 0:
			pygame.draw.rect(surf,DARK_GREEN,pygame.Rect(snake1_head_position[0]*scale-body_width/2,snake1_head_position[1]*scale-body_width/2,body_width,body_width))  

		for i in range(0,len(snake1_body_positions)):
			if snake1_body_positions[i][0] != 0 and snake1_body_positions[i][1] != 0 :
				pygame.draw.rect(surf,GREEN,pygame.Rect(snake1_body_positions[i][0]*scale-body_width/2,snake1_body_positions[i][1]*scale-body_width/2,body_width,body_width))  

		if snake2_head_position[0] !=0 and snake2_head_position[1] != 0:
			pygame.draw.rect(surf,DARK_BLUE,pygame.Rect(snake2_head_position[0]*scale-body_width/2,snake2_head_position[1]*scale-body_width/2,body_width,body_width))  

		for i in range(0,len(snake2_body_positions)):
			if snake2_body_positions[i][0] != 0 and snake2_body_positions[i][1] != 0 :
				pygame.draw.rect(surf,BLUE,pygame.Rect(snake2_body_positions[i][0]*scale-body_width/2,snake2_body_positions[i][1]*scale-body_width/2,body_width,body_width))  
	

		temp_array = np.transpose(np.array(pygame.surfarray.pixels3d(surf)), axes=(1, 0, 2))

		if display_size==84:
			temp_array = cv2.cvtColor(temp_array, cv2.COLOR_RGB2GRAY)
			temp_array = np.expand_dims(temp_array, -1)

		return temp_array

   

	def calculate_gae(self,last_values,last_dones,id):
		next_nonterminal = None
		last_gae_lam = 0

		for t in reversed(range(self.num_timesteps)):
			if t == self.num_timesteps - 1: 
				next_nonterminal = 1.0-last_dones
				next_values = last_values
			else:
				next_nonterminal = 1.0-self.batch_dones[id][t+1]
				next_values = self.batch_values[id][t+1]

			delta = self.batch_rewards[id][t]+self.gamma*next_nonterminal*next_values-self.batch_values[id][t] 

			self.batch_advantages[id][t] = last_gae_lam = delta +self.gamma*next_nonterminal*self.lamda*last_gae_lam

		self.batch_returns[id] = self.batch_advantages[id]+self.batch_values[id]


	def init_data(self,id):
		self.batch_actions[id] =  self.batch_actions[id].reshape(self.num_timesteps,self.num_envs)
		self.batch_values[id] = self.batch_values[id].reshape(self.num_timesteps,self.num_envs) 
		self.batch_logprobs_ac[id] = self.batch_logprobs_ac[id].reshape(self.num_timesteps,self.num_envs)
		self.batch_entropies_agent[id] = self.batch_entropies_agent[id].reshape(self.num_timesteps,self.num_envs) 
		self.batch_obs[id] = self.batch_obs[id].reshape(self.num_timesteps,self.num_envs,84,84,1) 
		self.batch_rewards[id] = self.batch_rewards[id].reshape(self.num_timesteps,self.num_envs)
		self.batch_dones[id] = self.batch_dones[id].reshape(self.num_timesteps,self.num_envs) 
		self.batch_masked_directions[id] =  self.batch_masked_directions[id].reshape(self.num_timesteps,self.num_envs)
		self.batch_advantages[id] = self.batch_advantages[id].reshape(self.num_timesteps,self.num_envs) 
		self.batch_returns[id] = self.batch_returns[id].reshape(self.num_timesteps,self.num_envs) 
		




	def step(self):
		#print("Step function enter:")
		#Below are list of tensors which will be converted to tensors each having size = batch_size after collecting data

		actions_for_step = np.zeros((self.num_envs,2))

		for id in self.snakeids:
			self.init_data(id)

		for i in range(0,self.num_timesteps):
			#print("----------------------------------TIMESTEP NO:%d---------------------------------------"%i)	
			for id in self.snakeids:

				self.batch_obs[id][i] = self.next_obs[id]

				self.batch_dones[id][i] = torch.as_tensor(self.next_dones[id],dtype =torch.float32).to(self.device)

				self.batch_actions[id][i],self.batch_values[id][i],self.batch_logprobs_ac[id][i],self.batch_entropies_agent[id][i] = self.actor_critic[id].step(self.next_obs[id],self.masked_directions_tensor[id])

				actions_for_step[:,self.snakeids.index(id)] = self.batch_actions[id][i].cpu().numpy()


			next_obs_tmp,_,next_dones_tmp,infos = self.snake_game_envs.step(actions_for_step)


			next_obs_env_tmp =torch.zeros(self.num_envs,84,84,1).to(self.device)

			for j in range(0,self.num_envs):
				next_obs_env_tmp[j] = torch.as_tensor(self.render(next_obs_tmp["fruit_position"][j],next_obs_tmp["snake1_head_position"][j],next_obs_tmp["snake1_body_positions"][j],
																				next_obs_tmp["snake2_head_position"][j],next_obs_tmp["snake2_body_positions"][j]),dtype =torch.float32).to(self.device)	


			for id in self.snakeids:
				self.next_obs[id] = next_obs_env_tmp #same observation for each actor critic
				self.next_dones[id] = next_dones_tmp
				tmp_rewards = np.zeros(self.num_envs)
				for j in range(0,self.num_envs):
					self.masked_directions_tensor[id][j] = torch.as_tensor(next_obs_tmp[id+"_masked_direction"][j],dtype =torch.float32).to(self.device)
					tmp_rewards[j] = infos[id+"_reward"][j][0]
				self.batch_rewards[id][i] = torch.as_tensor(tmp_rewards,dtype=torch.float32).to(self.device)	
				self.batch_masked_directions[id][i] = self.masked_directions_tensor[id]

			if(proc_id()==0):
				self.total_reward_d["snake1"] += infos["snake1_reward"][0][0]
				self.total_reward_d["snake2"] += infos["snake2_reward"][0][0]
				self.episode_length += 1

				if(next_dones_tmp[0]):
					filestr = "Episode"+str(self.episodeid)+":"+"Snake1 Return="+str(round(self.total_reward_d["snake1"],2))+",Snake2 Return="+str(round(self.total_reward_d["snake2"],2))+",Length="+str(self.episode_length)
					self.trainf.write(str(filestr)+"\n")
					self.episodeid +=1
					self.total_reward_d["snake1"]= 0.0
					self.total_reward_d["snake2"]= 0.0
					self.episode_length = 0	
				if(self.capped_cubic_video_schedule()):
					display_size = 400
					scale = 10
					body_width =40
					frame = self.render(next_obs_tmp["fruit_position"][0],next_obs_tmp["snake1_head_position"][0],next_obs_tmp["snake1_body_positions"][0],next_obs_tmp["snake2_head_position"][0],next_obs_tmp["snake2_body_positions"][0],display_size,scale,body_width)
					self.vi_rec.capture_frame(frame)
					if(next_dones_tmp[0]):
						self.vi_rec.close()
						video_path = os.path.abspath(os.getcwd())+"/video/"+"Episode_"+str(self.episodeid)+".mp4"
						self.vi_rec = VideoRecorder(video_path)


		next_values = {}
		for id in self.snakeids:
			_,next_values[id],_,_ = self.actor_critic[id].step(self.next_obs[id],self.masked_directions_tensor[id])

			self.batch_advantages[id] = torch.zeros_like(self.batch_values[id]).to(self.device)
			self.batch_returns[id] = torch.zeros_like(self.batch_values[id]).to(self.device)

			self.calculate_gae(next_values[id],torch.from_numpy(self.next_dones[id]).type(torch.float32).to(self.device),id)


			self.batch_actions[id] =  self.batch_actions[id].reshape(-1)
			self.batch_values[id] = self.batch_values[id].reshape(-1) 
			self.batch_logprobs_ac[id] = self.batch_logprobs_ac[id].reshape(-1)
			self.batch_entropies_agent[id] = self.batch_entropies_agent[id].reshape(-1) 
			self.batch_obs[id] = self.batch_obs[id].reshape((-1,)+(84,84,1)) 
			self.batch_rewards[id] = self.batch_rewards[id].reshape(-1)
			self.batch_dones[id] = self.batch_dones[id].reshape(-1) 
			self.batch_advantages[id] = self.batch_advantages[id].reshape(-1) 
			self.batch_returns[id] = self.batch_returns[id].reshape(-1) 
			self.batch_masked_directions[id] =  self.batch_masked_directions[id].reshape(-1)

		
		#print("Step function exit:")

	

	def train(self):
		setup_mpi_gpus()

		if(proc_id()==0):
			print("Enter train")

		setup_pytorch_for_mpi()	

		seed = 1
		np.random.seed(seed)
		random.seed(seed)

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		self.batch_size =  self.num_timesteps*self.num_envs

		self.snake_game_envs = envpool.make_gym('MultiSnakeDiscrete-v2',num_envs=self.num_envs,envType = self.envType)

		self.episodeid = 1

		self.vi_rec = None

		if(proc_id()==0):
			print('************************Device*************************')
			print(self.device)
			video_path = os.path.abspath(os.getcwd())+"/video/"+"Episode_"+str(self.episodeid)+".mp4"
			self.vi_rec = VideoRecorder(video_path)
	
		next_obs_tmp = self.snake_game_envs.reset()

		self.masked_directions_tensor = {}
		self.next_obs  = {}
		self.next_dones = {}
		self.actor_critic = {}
		self.optimizer = {}

		torch.manual_seed(seed)

		next_obs_env_tmp =torch.zeros(self.num_envs,84,84,1).to(self.device)

		for j in range(0,self.num_envs):
			next_obs_env_tmp[j] = torch.as_tensor(self.render(next_obs_tmp["fruit_position"][j],next_obs_tmp["snake1_head_position"][j],next_obs_tmp["snake1_body_positions"][j],
																			next_obs_tmp["snake2_head_position"][j],next_obs_tmp["snake2_body_positions"][j]),dtype =torch.float32).to(self.device)	
				
		for id in self.snakeids:
			self.masked_directions_tensor[id] = torch.from_numpy(np.zeros(self.num_envs)).type(torch.float32).to(self.device)	
			self.next_obs[id] = next_obs_env_tmp #same observation for each actor critic
			self.next_dones[id] = np.zeros(self.num_envs)
			self.actor_critic[id] = MLPActorCritic().to(self.device)
			sync_params(self.actor_critic[id])
			self.optimizer[id] = torch.optim.Adam(self.actor_critic[id].parameters(),self.learning_rate)

			for j in range(0,self.num_envs):
				self.masked_directions_tensor[id][j] = torch.as_tensor(next_obs_tmp[id+"_masked_direction"][j],dtype =torch.float32).to(self.device)

		self.lr_current = self.learning_rate	
	
		if(proc_id()==0):
			self.trainf = open('TrainLog.txt','a')

		self.batch_actions = {} 
		self.batch_values = {}
		self.batch_logprobs_ac = {}
		self.batch_entropies_agent = {}

		self.batch_obs = {}
		self.batch_rewards = {}
		self.batch_dones = {}
		self.batch_masked_directions = {}
		self.batch_advantages = {}
		self.batch_returns = {}
			
		for snakeid in self.snakeids:
			self.batch_actions[snakeid] = torch.zeros(self.num_timesteps,self.num_envs).to(self.device) 
			self.batch_values[snakeid] = torch.zeros(self.num_timesteps,self.num_envs).to(self.device)
			self.batch_logprobs_ac[snakeid] = torch.zeros(self.num_timesteps,self.num_envs).to(self.device)
			self.batch_entropies_agent[snakeid] = torch.zeros(self.num_timesteps,self.num_envs).to(self.device)
			self.batch_obs[snakeid] = torch.zeros(self.num_timesteps,self.num_envs,84,84,1).to(self.device)
			self.batch_rewards[snakeid] = torch.zeros(self.num_timesteps,self.num_envs).to(self.device)
			self.batch_dones[snakeid] = torch.zeros(self.num_timesteps,self.num_envs).to(self.device)
			self.batch_masked_directions[snakeid] = torch.zeros(self.num_timesteps,self.num_envs).to(self.device)
			self.batch_advantages[snakeid] = torch.zeros(self.num_timesteps,self.num_envs).to(self.device)
			self.batch_returns[snakeid] = torch.zeros(self.num_timesteps,self.num_envs).to(self.device)	

		for update in range(1,self.num_updates+1):
			self.update = update

			if(proc_id()==0):
				print("************Multi_Agent_PPO*****UpdateNum*******:",update)

			frac = 1.0-(update-1.0)/self.num_updates
			self.lr_current = frac*self.learning_rate
			
			self.step() #step the environment and actor critic to get one batch of data

			self.batch_indices = [i for i in range(0,self.batch_size)]

			random.shuffle(self.batch_indices)

			self.compute_gradients_and_optimize()
		
		if(proc_id()==0):
			self.vi_rec.close()
			self.trainf.close()
		self.snake_game_envs.close()



	def compute_gradients_and_optimize(self):

		optimizers_d = {}
		snakeids = ["snake1","snake2"] 

		for id in snakeids:	
			for group in self.optimizer[id].param_groups:
				group['lr'] = self.lr_current	
			
		#print("***********************************Enter compute_gradients_and_optimize***********************")
		
		for epoch in range(self.epochs):
			i = 0
			while (i < self.batch_size):
				start = i
				end = i+ self.mini_batch_size
				slice = self.batch_indices[start:end]

				for id in snakeids:
					_,new_v,new_logp_a,entropy = self.actor_critic[id].step(self.batch_obs[id][slice],self.batch_masked_directions[id][slice],self.batch_actions[id][slice],grad_condition=True)

					mini_batch_advantages = self.batch_advantages[id][slice]

					mini_batch_advantages_mean = mini_batch_advantages.mean()
					mini_batch_advantages_std = mini_batch_advantages.std()
					mini_batch_advantages = (mini_batch_advantages - mini_batch_advantages_mean)/(mini_batch_advantages_std + 1e-8)


					logratio = new_logp_a-self.batch_logprobs_ac[id][slice]

					ratio = logratio.exp()
					
					ploss1 = -mini_batch_advantages*ratio
					ploss2 = -mini_batch_advantages* torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef) 
					ploss = torch.max(ploss1,ploss2).mean()

					vloss1 = (new_v-self.batch_returns[id][slice])**2
					vloss2 = ((self.batch_values[id][slice]+torch.clamp(new_v-self.batch_values[id][slice],-self.clip_coef, self.clip_coef))-self.batch_returns[id][slice])**2
					vloss = 0.5*torch.max(vloss1,vloss2).mean()

					entropy_loss = entropy.mean()

					loss = ploss - self.entropy_coef*entropy_loss + self.value_coef*vloss

					self.optimizer[id].zero_grad()
					loss.backward()

					nn.utils.clip_grad_norm_(self.actor_critic[id].parameters(),self.max_grad_norm)
					mpi_avg_grads(self.actor_critic[id]) 
					self.optimizer[id].step()


				i = i+self.mini_batch_size	



	def test(self):
		if proc_id()==0:	
			test_count_max = 10
			test_count = 0

			snake_game_env_test = envpool.make_gym('MultiSnakeDiscrete-v2',num_envs=1,envType=self.envType)

			next_obs_tmp = snake_game_env_test.reset()

			masked_directions_tensor ={}

			masked_directions_tensor["snake1"] = torch.from_numpy(np.zeros(1)).type(torch.float32).to(self.device)
			masked_directions_tensor["snake2"] = torch.from_numpy(np.zeros(1)).type(torch.float32).to(self.device)
			
			next_obs = {}

			next_obs["snake1"] = torch.zeros(1,84,84,1).to(self.device)
			next_obs["snake2"] = torch.zeros(1,84,84,1).to(self.device)

			episode_return_snake1 = 0.0
			episode_return_snake2 = 0.0
			episode_length = 0
			episodeid=0

			display_size =400
			body_width=40
			scale=10

			video_path = os.path.abspath(os.getcwd())+"/video-test/"+"Test_Episode_"+str(episodeid)+".mp4"
			vi_rec = VideoRecorder(video_path) 
			frame=self.render(next_obs_tmp["fruit_position"][0],next_obs_tmp["snake1_head_position"][0],next_obs_tmp["snake1_body_positions"][0],next_obs_tmp["snake2_head_position"][0],next_obs_tmp["snake2_body_positions"][0],display_size,scale,body_width)
			vi_rec.capture_frame(frame)

			testf = open('TestLog.txt','a')

			actions_for_step = np.zeros((1,2))

			while(test_count<test_count_max):

				#print("----------------------------------TIMESTEP NO:%d---------------------------------------"%i)
				#print(self.next_obs.shape)				


				for id in self.snakeids:

					masked_directions_tensor[id][0] = torch.as_tensor(next_obs_tmp[id+"_masked_direction"],dtype =torch.float32).to(self.device)

					next_obs[id][0] = torch.as_tensor(self.render(next_obs_tmp["fruit_position"][0],next_obs_tmp["snake1_head_position"][0],next_obs_tmp["snake1_body_positions"][0],next_obs_tmp["snake2_head_position"][0],next_obs_tmp["snake2_body_positions"][0]),dtype =torch.float32).to(self.device)


					actions,values,logprobs_ac,entropies_agent = self.actor_critic[id].step(
						torch.as_tensor(next_obs[id],dtype = torch.float32).to(self.device),masked_directions_tensor[id])


					actions_for_step[0][self.snakeids.index(id)] = actions.cpu().numpy()[0]
					
				next_obs_tmp,rewards,next_dones,infos = snake_game_env_test.step(actions_for_step)


				episode_return_snake1 += infos["snake1_reward"][0][0]
				episode_return_snake2 += infos["snake2_reward"][0][0]
				episode_length += 1

				
				frame=self.render(next_obs_tmp["fruit_position"][0],next_obs_tmp["snake1_head_position"][0],next_obs_tmp["snake1_body_positions"][0],next_obs_tmp["snake2_head_position"][0],next_obs_tmp["snake2_body_positions"][0],display_size,scale,body_width)
				vi_rec.capture_frame(frame)
				if(next_dones[0]):					
					filestr = "Test_Episode"+str(episodeid)+":"+"snake1_return="+str(round(episode_return_snake1,2))+" snake2_return="+str(round(episode_return_snake2,2))+",length="+str(episode_length)
					testf.write(str(filestr)+"\n")
					episodeid +=1
					vi_rec.close()
					video_path = os.path.abspath(os.getcwd())+"/video-test/"+"Test_Episode_"+str(episodeid)+".mp4"
					vi_rec = VideoRecorder(video_path)
					episode_return_snake1 = 0.0
					episode_return_snake2 = 0.0
					episode_length = 0		
					test_count +=1

			testf.close()
			vi_rec.close()
			snake_game_env_test.close()

if __name__ == '__main__':	
	multi_ppo_obj = MULTIPPO()
	multi_ppo_obj.train()
	multi_ppo_obj.test()


