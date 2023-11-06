# Copyright 2022 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Box2D env in EnvPool."""

from envpool.python.api import py_env

from .box2d_envpool import (
  _BipedalWalkerEnvPool,
  _BipedalWalkerEnvSpec,
  _CarRacingEnvPool,
  _CarRacingEnvSpec,
  _LunarLanderContinuousEnvPool,
  _LunarLanderContinuousEnvSpec,
  _LunarLanderDiscreteEnvPool,
  _LunarLanderDiscreteEnvSpec,
  _SnakeDiscreteEnvPool,
  _SnakeDiscreteEnvSpec,
  _MultiSnakeDiscreteEnvPool,
  _MultiSnakeDiscreteEnvSpec
)

(
  BipedalWalkerEnvSpec, BipedalWalkerDMEnvPool, BipedalWalkerGymEnvPool,
  BipedalWalkerGymnasiumEnvPool
) = py_env(_BipedalWalkerEnvSpec, _BipedalWalkerEnvPool)

(
  CarRacingEnvSpec, CarRacingDMEnvPool, CarRacingGymEnvPool,
  CarRacingGymnasiumEnvPool
) = py_env(_CarRacingEnvSpec, _CarRacingEnvPool)

(
  LunarLanderContinuousEnvSpec,
  LunarLanderContinuousDMEnvPool,
  LunarLanderContinuousGymEnvPool,
  LunarLanderContinuousGymnasiumEnvPool,
) = py_env(_LunarLanderContinuousEnvSpec, _LunarLanderContinuousEnvPool)

(
  LunarLanderDiscreteEnvSpec,
  LunarLanderDiscreteDMEnvPool,
  LunarLanderDiscreteGymEnvPool,
  LunarLanderDiscreteGymnasiumEnvPool,
) = py_env(_LunarLanderDiscreteEnvSpec, _LunarLanderDiscreteEnvPool)

(
  SnakeDiscreteEnvSpec,
  SnakeDiscreteDMEnvPool,
  SnakeDiscreteGymEnvPool,
  SnakeDiscreteGymnasiumEnvPool,
) = py_env(_SnakeDiscreteEnvSpec, _SnakeDiscreteEnvPool)

(
  MultiSnakeDiscreteEnvSpec,
  MultiSnakeDiscreteDMEnvPool,
  MultiSnakeDiscreteGymEnvPool,
  MultiSnakeDiscreteGymnasiumEnvPool,
) = py_env(_MultiSnakeDiscreteEnvSpec, _MultiSnakeDiscreteEnvPool)


__all__ = [
  "CarRacingEnvSpec",
  "CarRacingDMEnvPool",
  "CarRacingGymEnvPool",
  "BipedalWalkerEnvSpec",
  "BipedalWalkerDMEnvPool",
  "BipedalWalkerGymEnvPool",
  "BipedalWalkerGymnasiumEnvPool",
  "LunarLanderContinuousEnvSpec",
  "LunarLanderContinuousDMEnvPool",
  "LunarLanderContinuousGymEnvPool",
  "LunarLanderContinuousGymnasiumEnvPool",
  "LunarLanderDiscreteEnvSpec",
  "LunarLanderDiscreteDMEnvPool",
  "LunarLanderDiscreteGymEnvPool",
  "LunarLanderDiscreteGymnasiumEnvPool",
  "SnakeDiscreteEnvSpec",
  "SnakeDiscreteDMEnvPool",
  "SnakeDiscreteGymEnvPool",
  "SnakeDiscreteGymnasiumEnvPool",
  "MultiSnakeDiscreteEnvSpec",
  "MultiSnakeDiscreteDMEnvPool",
  "MultiSnakeDiscreteGymEnvPool",
  "MultiSnakeDiscreteGymnasiumEnvPool",
]
