import envpool
import numpy as np
import gym
from getkey import getkey,keys
import os
import pygame

from video_recorder import VideoRecorder

BLACK = (0,0,0) #background 

DARK_GREEN = (0,100,0)#Snake head color
GREEN = (0, 255, 0) #Snake body color

RED = (255, 0, 0) #Fruit color
BROWN = (156,102,31) #Grid color

DARK_BLUE = (39,64,139) #Snake2 head color
BLUE = (135,206,250) #Snake2 body color 

display_size = 400
scale = 10
body_width =40

fruit_position = None
snake1_head_position = None
snake1_bodies_position = []
snake2_head_position = None
snake2_bodies_position = []

frame =None

def render(fruit_position,snake1_head_position,snake1_body_positions,snake2_head_position,snake2_body_positions):
	running = True
	pygame.init()
	pygame.display.init()
	window = pygame.display.set_mode((400,400))


	while running == True:
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
	
	 
		frame= np.transpose(np.array(pygame.surfarray.pixels3d(surf)), axes=(1, 0, 2))

	
		window.blit(surf, surf.get_rect())
		pygame.event.pump()
		pygame.display.update()

			
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.quit()
				running = False   

def capped_cubic_video_schedule(episode_id: int) -> bool:
	#print("Enter capped_cubic_video_schedule")
	'''if self.update  < self.num_updates: 
		if episode_id < 1000:
			return int(round(episode_id ** (1.0 / 3))) ** 3 == episode_id
		else:
			return episode_id % 1000 == 0
	else:'''
	return True 

COOPERATIVE=4
COMPETITIVE=5
envs = envpool.make_gym('MultiSnakeDiscrete-v2',num_envs=1,envType=4)
#episodeid = 1
#video_path = os.path.abspath(os.getcwd())+"/multisnake-video-env-test/"+"Episode_"+str(episodeid)+".mp4"
#vi_rec = VideoRecorder(video_path)

envs.reset()


while(True):
	actions = np.zeros((1,2))
	for i in range(2):
		key = getkey()
		action =-1
		if key == keys.LEFT:
			print('LEFT')
			action = 0 
		elif key == keys.RIGHT:
			print('RIGHT')
			action = 1
		elif key == keys.UP:
			print('UP')
			action = 2
		elif key == keys.DOWN:
			print('DOWN')
			action = 3
		actions[0][i] = action
			
	print("before step")	
	obs,_,dones,infos= envs.step(actions)

	print("is_game_over=",dones)
	print("snake1 reward=",infos["snake1_reward"])
	print("snake2 reward=",infos["snake2_reward"])


	fruit_position = obs["fruit_position"][0]
	
	snake1_head_position = obs["snake1_head_position"][0]
	snake1_body_positions = obs["snake1_body_positions"][0]

	snake2_head_position = obs["snake2_head_position"][0]
	snake2_body_positions = obs["snake2_body_positions"][0]

	snake1_masked_direction = obs["snake1_masked_direction"][0]
	snake2_masked_direction = obs["snake2_masked_direction"][0]

	
	print("snake1_head_position=",snake1_head_position)
	print("snake1_body_positions=",snake1_body_positions)
	print("snake1_masked_direction=",snake1_masked_direction)

	print("snake2_head_position=",snake2_head_position)
	print("snake2_body_positions=",snake2_body_positions)
	print("snake2_masked_direction=",snake2_masked_direction)


	frame=render(fruit_position,snake1_head_position,snake1_body_positions,snake2_head_position,snake2_body_positions)


