#ifndef MULTI_SNAKE_ENV_H_
#define MULTI_SNAKE_ENV_H_

#include <box2d/box2d.h>
#include <array>
#include <memory>
#include <random>
#include <vector>
#include <utility>
#include <unordered_map>
#include <unordered_set>
#include <optional>
#include <array>
#include <cmath>

using namespace std;

namespace box2d {

	class MultiSnakeGameEnv
	{
		const float DEG2RAD = M_PI/180.0;
		const float RAD2DEG = 180.0/M_PI;

		const int LEFT = 0;
		const int RIGHT =1;
		const int UP = 2;
		const int DOWN = 3;

		const int COOPERATIVE = 4;
		const int COMPETITIVE = 5;

		const int GAME_ENV_SIZE = 400;

		const int BODY_WIDTH = 40;
		const int SCALE = 10;

		const double DEATH_REWARD = -1.0;
		const double FRUIT_REWARD = 1.0;
		const double HALF_REWARD = 0.5; 
		const double LIVE_MOVE_REWARD = 0.1;
		const double DOUBLE_REWARD = 64;

		std::unique_ptr<b2World>m_world;

		b2Body*m_maze;

		int m_mazeCollisionBound_1;
		int m_mazeCollisionBound_2;

		vector<pair<int,int>>m_mazeIndices;

		unordered_map<string,bool>m_isFruitEatenMap;

		unordered_map<string,int>m_numMovesMap;


		unordered_set<int>m_playAreaSet;
		unordered_map<int,pair<int,int>>m_areaPosMap;

		int m_cols;

		unordered_map<string,unordered_set<int>>m_headPosSetMap;
		unordered_map<string,unordered_set<int>>m_bodyPosSetMap;

		int m_totalSize;

		protected:
			unordered_map<string,double>m_rewardMap;
			b2Body*m_fruit;
			unordered_map<string,b2Body*>m_headMap;
			unordered_map<string,vector<b2Body*>>m_bodyMap;
			bool m_isGameOver;
			unordered_map<string,int>m_maskedDirectionMap;
			unordered_map<string,bool>m_IsSnakeDead;

		private:
			void createMaze();
			optional<pair<int,int>> samplePositionFromPlayArea(vector<unordered_set<int>>list_occupied_area_sets);
			double sampleAngle();

			void createFruit();
			void destroyFruit();
			void moveFruitToAnotherLocation();
			
			void createSnake();
			void moveSnake(string snakeid,const int next_direction);
			void checkContact(string snakeid, string othersnakeid);
			void increaseSnakeLength(string snakeid);
			int findDirection(b2Body*unit);
			int findMaskedDirection(string snakeid);
			
			void destroySnake(string snakeid);

			vector<string>m_snakeIDs;
			unordered_map<string,bool>m_snakeCollidedWithOtherSnake;
			unordered_map<string,bool>m_IsSnakeAlreadyPenalizedForDeath;
			int m_envType;
			int m_actionOrderIndex;  //helps in shuffling the order of action execution for each step

		public:
			MultiSnakeGameEnv(int envType);
			~MultiSnakeGameEnv();
			void MultiSnakeGameEnvReset();
			void MultiSnakeGameEnvStep(vector<int>action);
	};

}

#endif  // ENVPOOL_SNAKE_ENV_H_