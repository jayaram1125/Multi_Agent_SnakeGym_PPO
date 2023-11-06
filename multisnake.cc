#include "envpool/box2d/multisnake.h"
#include <algorithm>
#include "envpool/box2d/utils.h"
#include <iterator>
#include <algorithm>
#include <random>
#include <cmath>
#include <queue>
#include <tuple>
#include <numeric>
#include <iostream>
#include <map>


using namespace std;

namespace box2d 
{

	MultiSnakeGameEnv::MultiSnakeGameEnv(int envType)
	:m_world(new b2World(b2Vec2(0.0, 0.0))),
	m_maze(NULL),
	m_mazeCollisionBound_1(0),
	m_mazeCollisionBound_2(0),
	m_mazeIndices(),
	m_isFruitEatenMap(),
	m_numMovesMap(),
	m_playAreaSet(),
	m_areaPosMap(),
	m_cols(0),
	m_headPosSetMap(),
	m_bodyPosSetMap(),
	m_totalSize(GAME_ENV_SIZE/BODY_WIDTH),
	m_rewardMap(),
	m_fruit(NULL),
	m_headMap(),
	m_bodyMap(),
	m_isGameOver(false),
	m_maskedDirectionMap(),
	m_snakeIDs(),
	m_snakeCollidedWithOtherSnake(),
	m_IsSnakeDead(),
	m_IsSnakeAlreadyPenalizedForDeath(),
	m_envType(envType),
	m_actionOrderIndex(0)
	{
		m_snakeIDs ={"snake1","snake2"};
		m_snakeCollidedWithOtherSnake={{"snake1",false},{"snake2",false}};
		m_IsSnakeDead ={{"snake1",false},{"snake2",false}};
		m_IsSnakeAlreadyPenalizedForDeath ={{"snake1",false},{"snake2",false}};
		createMaze();
	}


	MultiSnakeGameEnv::~MultiSnakeGameEnv()
	{
		//cout<<"destuctor called"<<endl;
		for(string id:m_snakeIDs)
		{
			destroySnake(id);
		}
		destroyFruit();
		m_world->DestroyBody(m_maze);
		m_maze = NULL;
	}

	void MultiSnakeGameEnv::createMaze()
	{
		//cout<<"create Maze called"<<endl;
		vector<b2Vec2>vertices_vec = {b2Vec2(0.0,0.0), b2Vec2(GAME_ENV_SIZE/SCALE,0.0), b2Vec2(GAME_ENV_SIZE/SCALE,GAME_ENV_SIZE/SCALE),b2Vec2(0.0,GAME_ENV_SIZE/SCALE),
					b2Vec2(0.0,0.0), b2Vec2(BODY_WIDTH/SCALE,BODY_WIDTH/SCALE),b2Vec2(GAME_ENV_SIZE/SCALE-BODY_WIDTH/SCALE,BODY_WIDTH/SCALE),b2Vec2(GAME_ENV_SIZE/SCALE-BODY_WIDTH/SCALE,GAME_ENV_SIZE/SCALE-BODY_WIDTH/SCALE),b2Vec2(BODY_WIDTH/SCALE,GAME_ENV_SIZE/SCALE-BODY_WIDTH/SCALE),b2Vec2(BODY_WIDTH/SCALE,BODY_WIDTH/SCALE)};

		b2Vec2*vertices = vertices_vec.data();

		b2ChainShape b2ChainShapeObj;

		b2ChainShapeObj.CreateLoop(vertices,10);

		m_mazeCollisionBound_1 = BODY_WIDTH/SCALE/2;
		m_mazeCollisionBound_2 = GAME_ENV_SIZE/SCALE-BODY_WIDTH/SCALE/2;


		b2BodyDef bd;
		bd.type = b2_staticBody;
		bd.position = b2Vec2(0.0f, 0.0f);
		bd.angle = 0.0;

	  	b2FixtureDef fd;
  		fd.shape = &b2ChainShapeObj;

		m_maze = m_world->CreateBody(&bd);
		m_maze->CreateFixture(&fd);


		int start = 0;
		int end = GAME_ENV_SIZE/BODY_WIDTH;

	

		//top side of maze
		for(int j=start;j<end ;j++)
			m_mazeIndices.push_back(make_pair(start,j));


		//left side of the maze
		int begin = start+1;
		for(int i=begin;i<end; i++)
			m_mazeIndices.push_back(make_pair(i,start));


		//bottom side of maze	
		for(int j=begin ;j<end ;j++)
			m_mazeIndices.push_back(make_pair(end-1,j));


		//right side of the maze
		for(int i=begin;i<end-1;i++)
			m_mazeIndices.push_back(make_pair(i,end-1)); 


		int playAreaBound_1 = BODY_WIDTH/SCALE + 2;
		int playAreaBound_2 = GAME_ENV_SIZE/SCALE-BODY_WIDTH/SCALE-2;

		m_cols = playAreaBound_2/4;



		for(int i=playAreaBound_1;i<playAreaBound_2+4;i=i+4)
		{
			for(int j = playAreaBound_1;j<playAreaBound_2+4;j=j+4)
			{
				//cout<<i<<","<<j;
				//i*self.cols+j helps to create unique key in the play area set
				m_playAreaSet.insert(i/4+(j/4)*m_cols);
				m_areaPosMap[i/4+(j/4)*m_cols] =make_pair(i,j);
			}
		}

	}

	optional<pair<int,int>> MultiSnakeGameEnv::samplePositionFromPlayArea(vector<unordered_set<int>>occupiedAreaSetsList)
	{
		//cout<<"samplePositionFromPlayArea()"<<endl;
		unordered_set<int> remainingAreaSet;
		std::copy_if(m_playAreaSet.begin(), m_playAreaSet.end(), inserter(remainingAreaSet, remainingAreaSet.begin()),
			[&occupiedAreaSetsList] (int val) { return occupiedAreaSetsList[0].count(val)==0;});

		
		for(int i =1;i<occupiedAreaSetsList.size();i++)
		{
			unordered_set<int> tempRemainingAreaSet;
			std::copy_if(remainingAreaSet.begin(), remainingAreaSet.end(), inserter(tempRemainingAreaSet, tempRemainingAreaSet.begin()),
				[&occupiedAreaSetsList,i] (int val) { return occupiedAreaSetsList[i].count(val)==0;});
			remainingAreaSet =tempRemainingAreaSet; 
		}

		/*cout<<"remaining_area_set="<<endl;
		for(auto a:remainingAreaSet)
			cout<<a<<"\t";
		cout<<endl;
		*/	
		optional<pair<int,int>>sampledPos;

		if(!remainingAreaSet.empty())
		{
			vector<int>sampleVec; 
			std::sample(remainingAreaSet.begin(), remainingAreaSet.end(), std::back_inserter(sampleVec),1,std::mt19937{std::random_device{}()});
			sampledPos = m_areaPosMap[sampleVec[0]];
			//cout<<"SAMPLED POSITION ="<<sampledPos.value().first<<":"<<sampledPos.value().second<<endl;
		}

		return sampledPos;
	}	 

	double MultiSnakeGameEnv::sampleAngle()
    { 
    	//cout<<"sampleAngle()"<<endl;
		vector<double>angles = {0.0,M_PI/2,3*M_PI/2,M_PI};
		vector<double>sampledAngle; 
		std::sample(angles.begin(), angles.end(), std::back_inserter(sampledAngle),1,std::mt19937{std::random_device{}()});
		return sampledAngle[0];
	}
		

	void MultiSnakeGameEnv::createFruit()
	{
		//cout<<"createFruit()"<<endl;
		auto sampledPos = samplePositionFromPlayArea({unordered_set<int>()});
		b2PolygonShape b2PolygonShapeObj;
		b2PolygonShapeObj.SetAsBox(2,2);

		b2BodyDef bd;
		bd.type = b2_staticBody;
		bd.position = b2Vec2(sampledPos.value().first, sampledPos.value().second);
		bd.angle = 0.0;

		b2FixtureDef fd;
		fd.shape = &b2PolygonShapeObj;

		m_fruit = m_world->CreateBody(&bd);
		if(m_fruit)
		{
			m_fruit->CreateFixture(&fd);

			//cout<<"inside create fruit"<<endl;
			//auto fruitPos =m_fruit->GetPosition();
			//cout<<fruitPos.x<<":"<<fruitPos.y<<endl;
		}
		else
		{
			cout<<"Error:Fruit is null.Could not be created"<<endl;
		}
	}	


	void MultiSnakeGameEnv::destroyFruit()
	{
		//cout<<"enter destroy fruit called"<<endl;
		m_world->DestroyBody(m_fruit);
		m_fruit = NULL;
	}

	void MultiSnakeGameEnv::moveFruitToAnotherLocation()
	{
		//cout<<"Enter moveFruitToAnotherLocation()"<<endl;
		auto sampledPos = samplePositionFromPlayArea({m_headPosSetMap["snake1"],m_headPosSetMap["snake2"],m_bodyPosSetMap["snake1"],m_bodyPosSetMap["snake2"]});   
		if(sampledPos)
		{
			if(m_fruit)
			{
				m_fruit->SetTransform(b2Vec2(sampledPos.value().first,sampledPos.value().second),0.0);
			}
			//auto pos = m_fruit->GetPosition();
			//cout<<"fruit x="<<pos.x<<"fruit y="<<pos.y<<endl;
		}
		else
		{
			//cout<<"Else block in moveFruitToAnotherLocation"<<endl;
			//Destroy fruit object when
			//case1:  when play area is filled with one snake
			//case2 : Both snakes are alive and long enough to fill the play area.
			destroyFruit();
		}
		//cout<<"Exit moveFruitToAnotherLocation"<<endl;
	}

	void MultiSnakeGameEnv::createSnake()
	{
		//cout<<"Create Snake"<<endl;	
		auto pos = m_fruit->GetPosition();
		unordered_set<int>occupiedAreaSet;
		occupiedAreaSet.insert(int(pos.x)/4+(int(pos.y)/4)*m_cols);

		for(string id:m_snakeIDs)
		{ 
			auto sampledPos = samplePositionFromPlayArea({occupiedAreaSet});	
			auto sampledAngle = sampleAngle();

			b2PolygonShape b2PolygonShapeObj;
			b2PolygonShapeObj.SetAsBox(2,2);

			b2BodyDef bd;
			bd.type = b2_staticBody;
			bd.position = b2Vec2(sampledPos.value().first, sampledPos.value().second);
			bd.angle = sampledAngle;

			b2FixtureDef fd;
			fd.shape = &b2PolygonShapeObj;

			m_headMap[id] = m_world->CreateBody(&bd);
			if(m_headMap[id])
			{
				m_headMap[id]->CreateFixture(&fd);
				auto headPos = m_headMap[id]->GetPosition();
				m_headPosSetMap[id].insert(int(headPos.x)/4+(int(headPos.y)/4)*m_cols);
				if(id == "snake1")
				{
					occupiedAreaSet.insert(m_headPosSetMap[id].begin(),m_headPosSetMap[id].end());
				}
			}
			else
			{
				cout<<"Error:head for:"<<id<<" could not be created"<<endl;
			}	
		}
	}


	void MultiSnakeGameEnv::moveSnake(string snakeid,const int nextDirection)
	{
		//cout<<"moveSnake()"<<endl;
		b2Vec2 headPosition;
		float headAngle =0.0;

		if(m_headMap[snakeid])
		{
			headPosition = m_headMap[snakeid]->GetPosition();
			headAngle = m_headMap[snakeid]->GetAngle();
		
			auto prevHeadPosition = headPosition;
			auto prevHeadAngle = headAngle;

			if(nextDirection == UP && round(headAngle*RAD2DEG) != 270)
			{ 
				headPosition.y = headPosition.y - BODY_WIDTH/SCALE; 
				headAngle = M_PI/2;
			}
			else if(nextDirection == DOWN && round(headAngle*RAD2DEG) != 90)
			{ 
				headPosition.y = headPosition.y + BODY_WIDTH/SCALE;
				headAngle = 3*M_PI/2;
			}		
			else if(nextDirection == RIGHT && round(headAngle*RAD2DEG) != 180)
			{ 				 
				headPosition.x = headPosition.x + BODY_WIDTH/SCALE;
				headAngle = 0;
			}
			else if(nextDirection == LEFT && round(headAngle*RAD2DEG) != 0)
			{
				headPosition.x = headPosition.x-BODY_WIDTH/SCALE;
				headAngle = M_PI;
			}

			m_headMap[snakeid]->SetTransform(headPosition,headAngle);
			m_headPosSetMap[snakeid].clear();
			m_headPosSetMap[snakeid].insert(int(headPosition.x)/4+(int(headPosition.y)/4)*m_cols);

			vector<pair<b2Vec2,float>> prevbodyPosVec; 
		
			//Update the body positions only if the head is moved .Snake cannot move in opposite direction	
			if(prevHeadPosition != headPosition && m_bodyMap[snakeid].size()>0)
			{	
				//cout<<"Enter that IF block"<<endl;	 	
				m_bodyPosSetMap[snakeid].clear();

				for(int i = 0; i<m_bodyMap[snakeid].size()-1;i++)
				{
					prevbodyPosVec.push_back(make_pair(m_bodyMap[snakeid][i]->GetPosition(),m_bodyMap[snakeid][i]->GetAngle()));
				}

				m_bodyMap[snakeid][0]->SetTransform(prevHeadPosition,prevHeadAngle);
				auto p_pos = m_bodyMap[snakeid][0]->GetPosition();
				m_bodyPosSetMap[snakeid].insert(int(p_pos.x)/4+(int(p_pos.y)/4)*m_cols);

				for(int i = 1; i<m_bodyMap[snakeid].size();i++)
				{
					m_bodyMap[snakeid][i]->SetTransform(prevbodyPosVec[i-1].first,prevbodyPosVec[i-1].second);
					auto p_pos = m_bodyMap[snakeid][i]->GetPosition();
					m_bodyPosSetMap[snakeid].insert(int(p_pos.x)/4+(int(p_pos.y)/4)*m_cols);
				}
			}
			m_numMovesMap[snakeid]++;
		}
	  	else
	  	{
	  		cout<<"Error:head is NULL.Could be destroyed earlier"<<endl;
	  	}	
	  	
	}
		
	void MultiSnakeGameEnv::checkContact(string snakeid, string othersnakeid)
	{
		//cout<<"checkContact()"<<endl;
        int headPosX = 0;
        int headPosY = 0;

        int fruitPosX =0;
        int fruitPosY =0;
 

		if(m_headMap[snakeid]) 
		{
			auto headPos = m_headMap[snakeid]->GetPosition();
			headPosX = int(headPos.x);
			headPosY = int(headPos.y);
		}
		else
		{
			cout<<"head is NULL.Could be destroyed earlier"<<endl;
		}

		if(m_fruit)
		{
			auto fruitPos = m_fruit->GetPosition();
			fruitPosX = int(fruitPos.x);
			fruitPosY = int(fruitPos.y);
		}
		else
		{
	
			//Play area is free.Now recreate the fruit which was destroyed earlier due to the play area getting filled
			//by both snakes
			int snake1Size = m_headPosSetMap["snake1"].size()+ m_bodyPosSetMap["snake1"].size();
			int snake2Size = m_headPosSetMap["snake2"].size()+ m_bodyPosSetMap["snake2"].size();
			int totalBodySizeofTwoSnakes = snake1Size+snake2Size;

			if(totalBodySizeofTwoSnakes !=64)
			{
				createFruit();
			}
		}

		bool snakeCollidedWithMaze = (headPosX == m_mazeCollisionBound_1 || headPosY == m_mazeCollisionBound_1 || headPosX  == m_mazeCollisionBound_2 or headPosY == m_mazeCollisionBound_2);

		int snakeHeadPosInPlayArea = int(headPosX/4+(headPosY/4)*m_cols);

		m_snakeCollidedWithOtherSnake[snakeid] = (m_headPosSetMap[othersnakeid].count(snakeHeadPosInPlayArea)||m_bodyPosSetMap[othersnakeid].count(snakeHeadPosInPlayArea));

		//Checking contact with Maze or itself or other snake
		if(snakeCollidedWithMaze || m_bodyPosSetMap[snakeid].count(snakeHeadPosInPlayArea)||m_snakeCollidedWithOtherSnake[snakeid])
		{	
			
			/*if(snakeCollidedWithMaze)
				cout<<"Snake collided with maze"<<endl;
			else
				cout<<"snake collided with itself"<<endl;*/

			destroySnake(snakeid);
			m_IsSnakeDead[snakeid] = true;
		}
		else if(fruitPosX == headPosX && fruitPosY == headPosY)
		{
			//cout<<"check4"<<endl;
			//Checking contact with fruit
			increaseSnakeLength(snakeid);
			m_isFruitEatenMap[snakeid] = true;
			moveFruitToAnotherLocation();
		}

	}


	void MultiSnakeGameEnv::increaseSnakeLength(string snakeid)
	{
		//cout<<"increaseSnakeLength()"<<endl;
		b2Vec2 lastUnitPos;
		int lastUnitAngle = 0;

		if(m_bodyMap[snakeid].size()>0)
		{
	 		lastUnitPos = m_bodyMap[snakeid][m_bodyMap[snakeid].size()-1]->GetPosition();
	 		lastUnitAngle =round((m_bodyMap[snakeid][m_bodyMap[snakeid].size()-1]->GetAngle())*RAD2DEG);
		}
	 	else
	 	{
			lastUnitPos = m_headMap[snakeid]->GetPosition();
			lastUnitAngle =round((m_headMap[snakeid]->GetAngle())*RAD2DEG); 
	 	}

		int lastUnitPosition_x = int(lastUnitPos.x);
		int lastUnitPosition_y = int(lastUnitPos.y);

		int distanceDelta = BODY_WIDTH/SCALE;

		int newBodyUnitPosition_x = 0 ;
		int newBodyUnitPosition_y = 0 ;
		
		if(lastUnitAngle == 0)
		{
			newBodyUnitPosition_x = lastUnitPosition_x-distanceDelta;
			newBodyUnitPosition_y = lastUnitPosition_y;
		}	
		else if(lastUnitAngle == 90)
		{
			newBodyUnitPosition_x = lastUnitPosition_x;
			newBodyUnitPosition_y = lastUnitPosition_y+distanceDelta;
		}
		else if(lastUnitAngle == 180)
		{
			newBodyUnitPosition_x = lastUnitPosition_x+distanceDelta;
			newBodyUnitPosition_y = lastUnitPosition_y;
		}
		else if(lastUnitAngle == 270)
		{
			newBodyUnitPosition_x = lastUnitPosition_x;
			newBodyUnitPosition_y = lastUnitPosition_y-distanceDelta;
		}

		b2PolygonShape b2PolygonShapeObj;
		b2PolygonShapeObj.SetAsBox(2,2);

		b2BodyDef bd;
		bd.type = b2_staticBody;
		bd.position = b2Vec2(newBodyUnitPosition_x, newBodyUnitPosition_y);
		bd.angle = lastUnitAngle;

		b2FixtureDef fd;
		fd.shape = &b2PolygonShapeObj;

		auto bodyPtr = m_world->CreateBody(&bd);
		if(bodyPtr)
		{
			bodyPtr->CreateFixture(&fd);
			m_bodyMap[snakeid].push_back(bodyPtr);
			m_bodyPosSetMap[snakeid].insert(newBodyUnitPosition_x/4+(newBodyUnitPosition_y/4)*m_cols);
		}
		else
		{
			cout<<"bodyPtr is NULL.Body could not be created"<<endl;
		}
	
	}



	int MultiSnakeGameEnv::findMaskedDirection(string snakeid)
	{
		//cout<<"findMaskedDirection()"<<endl;
		int maskedDirection = -1;
		if(m_headMap[snakeid])
		{
			auto angle = m_headMap[snakeid]->GetAngle();

			if(round(angle*RAD2DEG) == 270)
				maskedDirection = UP;
			else if(round(angle*RAD2DEG) == 90)
				maskedDirection = DOWN;
			else if(round(angle*RAD2DEG) == 180)
				maskedDirection = RIGHT;
			else if(round(angle*RAD2DEG) == 0)
				maskedDirection = LEFT;
		}
		return maskedDirection;
	}


	void MultiSnakeGameEnv::destroySnake(string snakeid)
	{
		//cout<<"destroySnake()"<<endl;
		if(m_headMap[snakeid])
		{
			m_world->DestroyBody(m_headMap[snakeid]);
			m_headMap[snakeid] = NULL;
			m_headPosSetMap[snakeid].clear();
		}
		for(int i=0 ;i<m_bodyMap[snakeid].size();i++)
		{
			//cout<<"enter destroy body:"<<endl;
			m_world->DestroyBody(m_bodyMap[snakeid][i]);
		}
		m_bodyMap[snakeid].clear();
		m_bodyPosSetMap[snakeid].clear();
		//cout<<"Snake Destroyed:"<<endl;	
	}


	void MultiSnakeGameEnv::MultiSnakeGameEnvReset()
	{
		//cout<<"SnakeGameEnvReset function enter called"<<endl;
		if(m_fruit)
		{
			destroyFruit();
			m_fruit = NULL;
		}

		for(string id:m_snakeIDs)
		{
			if(m_headMap[id])
			{	
				destroySnake(id);
			}
			m_IsSnakeDead[id]= false;
			m_IsSnakeAlreadyPenalizedForDeath[id]=false;
		}

		m_isGameOver = false;

		createFruit();

		createSnake();

		for(string id:m_snakeIDs)
		{
			m_numMovesMap[id] = 0;
			m_snakeCollidedWithOtherSnake[id] = false ;
			m_isFruitEatenMap[id] = false;			
			m_rewardMap[id] = 0;
		}

		//cout<<"SnakeGameEnvReset function exit"<<endl;
	}
	
	void MultiSnakeGameEnv::MultiSnakeGameEnvStep(vector<int>actions)
	{
		//cout<<"MultiSnakeGameEnvStep"<<endl;
	
		int count = 0;

		while(count<2)
		{
			string id = m_snakeIDs[m_actionOrderIndex];
			if(m_headMap[id])
			{

				moveSnake(id,actions[m_actionOrderIndex]);

				string otherSnakeID;
				if(id =="snake1")
				{
					otherSnakeID="snake2";
				}
				else
				{
					otherSnakeID="snake1";
				}
				
				checkContact(id,otherSnakeID);
			}

			if(count ==0  && m_actionOrderIndex == 0)
			{
				m_actionOrderIndex++;
			}
			else if(count ==0  && m_actionOrderIndex == 1)
			{
				m_actionOrderIndex--;
			}

			count++;
		}

		
		m_rewardMap["snake1"] = 0.0;
		m_rewardMap["snake2"] = 0.0;

		for(string id:m_snakeIDs)
		{
			string otherSnakeID;
			if(id =="snake1")
			{
				otherSnakeID="snake2";
			}
			else
			{
				otherSnakeID="snake1";
			}

			if(m_IsSnakeDead[id])
			{
				if(!m_IsSnakeAlreadyPenalizedForDeath[id])
				{
					m_rewardMap[id] = DEATH_REWARD;
					m_IsSnakeAlreadyPenalizedForDeath[id]=true;
					if(m_snakeCollidedWithOtherSnake[id])
					{
						m_rewardMap[otherSnakeID] += (m_envType==COMPETITIVE)?HALF_REWARD:-HALF_REWARD;	
					}
				}

				if(m_IsSnakeDead[otherSnakeID])
				{
					m_isGameOver =true;
				}

			}
			else if(m_isFruitEatenMap[id])
			{
				m_rewardMap[id] += FRUIT_REWARD;
				m_isFruitEatenMap[id] = false;
				m_numMovesMap[id] = 0;

				if(!m_IsSnakeDead[otherSnakeID])
				{
					m_rewardMap[otherSnakeID] += (m_envType==COMPETITIVE)?-HALF_REWARD: HALF_REWARD;
				}
				if(m_bodyMap[id].size()>=63)
				{
					m_rewardMap[id] += DOUBLE_REWARD;
					m_isGameOver =true;
				}
			}
			else if(m_numMovesMap[id]>=150)
			{
				m_rewardMap[id] = DEATH_REWARD;
				destroySnake(id);
				m_IsSnakeDead[id] = true;
				m_numMovesMap[id] = 0;
				if(m_IsSnakeDead[otherSnakeID])
				{
					m_isGameOver =true;
				}
			}
			else if(m_headMap[id])
			{
				m_rewardMap[id] += LIVE_MOVE_REWARD;
			}


			m_maskedDirectionMap[id] = findMaskedDirection(id);
 
			if(m_isGameOver)
			{
				break;
			}	
		}
	}
}
