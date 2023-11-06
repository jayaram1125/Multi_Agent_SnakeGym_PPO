/*
 * Copyright 2022 Garena Online Private Limited
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ENVPOOL_BOX2D_MULTISNAKE_DISCRETE_H_
#define ENVPOOL_BOX2D_MULTISNAKE_DISCRETE_H_

#include "envpool/box2d/multisnake.h"
#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include<string>
#include<typeinfo>

using namespace std;

namespace box2d {

class MultiSnakeDiscreteEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict("envType"_.Bind(0));
  }  
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    return MakeDict("obs:fruit_position"_.Bind(Spec<int>({2})),
    "obs:snake1_head_position"_.Bind(Spec<int>({2})),
    "obs:snake1_body_positions"_.Bind(Spec<int>({63,2})),
    "obs:snake2_head_position"_.Bind(Spec<int>({2})),
    "obs:snake2_body_positions"_.Bind(Spec<int>({63,2})),
    "obs:snake1_masked_direction"_.Bind(Spec<int>({1})),
    "obs:snake2_masked_direction"_.Bind(Spec<int>({1})),
    "info:snake1_reward"_.Bind(Spec<double>({1})),
    "info:snake2_reward"_.Bind(Spec<double>({1})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    return MakeDict("action"_.Bind(Spec<int>({2},std::tuple<int,int>{0, 3})));
  }
};

using MultiSnakeDiscreteEnvSpec = EnvSpec<MultiSnakeDiscreteEnvFns>;

class MultiSnakeDiscreteEnv : public Env<MultiSnakeDiscreteEnvSpec>,
                               public MultiSnakeGameEnv {
 public:
  MultiSnakeDiscreteEnv(const Spec& spec, int env_id)
      : Env<MultiSnakeDiscreteEnvSpec>(spec, env_id),
        MultiSnakeGameEnv(spec.config["envType"_]){}

  bool IsDone() override { return m_isGameOver; }

  void Reset() override {
    MultiSnakeGameEnvReset();
    WriteState();
  }

  void Step(const Action& action) override 
  {
    //cout<<"Multi snake discrete Step func called"<<endl;
    vector<int>actions(2,-1);

    actions[0] = action["action"_](0);
    actions[1] = action["action"_](1);
    //cout<<actions.size()<<endl;
    MultiSnakeGameEnvStep(actions);
    //cout<<"Multi snake discrete Step func before write state called"<<endl;
    WriteState();
    //cout<<"Multi snake discrete Step func end called"<<endl;
  }

 private:
  void WriteState() {
    State state = Allocate();

    state["info:snake1_reward"_] =  m_rewardMap["snake1"];
    state["info:snake2_reward"_] =  m_rewardMap["snake2"]; 

 
    int fruitX=0;
    int fruitY=0;

    if(m_fruit)
    {
      b2Vec2 fruitPos = m_fruit->GetPosition();
      //cout<<fruitPos.x<<"*****fruitPos****"<<fruitPos.y<<endl;
      fruitX=int(fruitPos.x);
      fruitY= int(fruitPos.y);
    }
    
    state["obs:fruit_position"_](0) = fruitX;
    state["obs:fruit_position"_](1) = fruitY;

    
    int headX=0;
    int headY=0;
    int maskedDirection=-1;

    if(m_headMap["snake1"])
    {
      b2Vec2 headPos = m_headMap["snake1"]->GetPosition();
      //cout<<headPos.x<<"*headPos********"<<headPos.y<<endl;
      headX = int(headPos.x);
      headY = int(headPos.y);
      maskedDirection = m_maskedDirectionMap["snake1"];
    }

    state["obs:snake1_head_position"_](0) = headX;
    state["obs:snake1_head_position"_](1) = headY;
    state["obs:snake1_masked_direction"_] = maskedDirection; 

    //cout<<"m_body size is:"<<m_body.size()<<endl;
    for(int j = 0;j<m_bodyMap["snake1"].size();j++)
    {
        b2Vec2 bodyPos;
        if(m_headMap["snake1"])
        {
            bodyPos = m_bodyMap["snake1"][j]->GetPosition();
            //cout<<bodyPos.x<<"*bodyPos********"<<bodyPos.y<<endl;
            state["obs:snake1_body_positions"_](j)(0) = int(bodyPos.x);
            state["obs:snake1_body_positions"_](j)(1) = int(bodyPos.y);
        }
    }

    for(int j = m_bodyMap["snake1"].size();j<63;j++)
    {
        state["obs:snake1_body_positions"_](j)(0) = 0;
        state["obs:snake1_body_positions"_](j)(1) = 0;
    } 
      //cout<<"Write state exit"<<endl;
    

    headX=0;
    headY=0;
    maskedDirection=-1;
  
    if(m_headMap["snake2"])
    {
      b2Vec2 headPos = m_headMap["snake2"]->GetPosition();
      //cout<<headPos.x<<"*headPos********"<<headPos.y<<endl;
      headX = int(headPos.x);
      headY = int(headPos.y);
      maskedDirection = m_maskedDirectionMap["snake2"];
    }

    state["obs:snake2_head_position"_](0) = headX;
    state["obs:snake2_head_position"_](1) = headY;
    state["obs:snake2_masked_direction"_] = maskedDirection;

    //cout<<"m_body size is:"<<m_body.size()<<endl;
    for(int j = 0;j<m_bodyMap["snake2"].size();j++)
    {
        b2Vec2 bodyPos;
        if(m_headMap["snake2"])
        {
            bodyPos = m_bodyMap["snake2"][j]->GetPosition();
            //cout<<bodyPos.x<<"*bodyPos********"<<bodyPos.y<<endl;
            state["obs:snake2_body_positions"_](j)(0) = int(bodyPos.x);
            state["obs:snake2_body_positions"_](j)(1) = int(bodyPos.y);
        }
    }

    for(int j = m_bodyMap["snake2"].size();j<63;j++)
    {
        state["obs:snake2_body_positions"_](j)(0) = 0;
        state["obs:snake2_body_positions"_](j)(1) = 0;
    }
  }
};

using MultiSnakeDiscreteEnvPool = AsyncEnvPool<MultiSnakeDiscreteEnv>;

}  // namespace box2d

#endif  // ENVPOOL_BOX2_SNAKE_DISCRETE_H_
