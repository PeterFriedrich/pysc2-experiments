# This base of this code is originally from skjb's pysc2 tutorial.
# https://github.com/skjb/pysc2-tutorial
# This program in particular is pulled from https://github.com/skjb/pysc2-tutorial/blob/master/Building%20a%20Basic%20Agent/simple_agent.py
# The blogpost is https://chatbotslife.com/building-a-basic-pysc2-agent-b109cde1477c

# I've simply experimented with it and more heavily
# commented it for the sake of my own learning/reference

# implement agent from its directory
# shell command to start agent is:
#python -m pysc2.bin.agent \
#--map Simple64 \
#--agent simple_agent_step.SimpleAgent \
#--agent_race T


# *****************************************************************************
# pysc2 is found at
# C:\Users\Peter\AppData\Local\Programs\Python\Python35\Lib\site-packages\pysc2

# imports a base agent from pysc2.agents, used to write custom scripted agents
# that we will inherit from
from pysc2.agents import base_agent
# imports actions from pysc2.lib folder.  Can read what all actions do there
from pysc2.lib import actions
# imports features.
from pysc2.lib import features

import time

# Constants with easier descriptions to read

# Functions
_BUILD_SUPPLYDEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
_NOOP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id
_RALLY_UNITS_MINIMAP = actions.FUNCTIONS.Rally_Units_minimap.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id

# Features
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index

# Unit IDs
_TERRAN_COMMANDCENTER = 18
_TERRAN_SUPPLYDEPOT = 19
_TERRAN_SCV = 45
_TERRAN_BARRACKS = 21

# Parameters
_PLAYER_SELF = 1
_NOT_QUEUED = [0]
_QUEUED = [1]
_SUPPLY_USED = 3
_SUPPLY_MAX = 4

# This simple agent class inherits from the base agent file
# its a class that has attributes reward, episode, steps, obs_spec, action_spec
class SimpleAgent(base_agent.BaseAgent):
    # flags for whether or not things have been done
    base_top_left = None
    supply_depot_built = False
    scv_selected = False
    barracks_built = False
    barracks_selected = False
    barracks_rallied = False
    army_selected = False
    army_rallied = False

    def transformLocation(self, x, x_distance, y, y_distance):
        if not self.base_top_left:
            return [x - x_distance, y - y_distance]

        return [x + x_distance, y + y_distance]

    # The game essentially calls the step method for each step
    # obs is a series of nested arrays with observations in it
    def step(self, obs):
        super(SimpleAgent, self).step(obs)

        # sleeps a bit each step
        time.sleep(0.10)

        if self.base_top_left is None:
            player_y, player_x = (obs.observation["minimap"][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
            self.base_top_left = player_y.mean() <= 31

        # section for finding and selecting random scv
        if not self.supply_depot_built:
            if not self.scv_selected:
                # using screen part of observation
                unit_type = obs.observation["screen"][_UNIT_TYPE]
                # get coordinates for all scvs on screen
                # this is a numpy array, notice data returned y, then x
                unit_y, unit_x = (unit_type == _TERRAN_SCV).nonzero()
                # select coordinates of first scv in list, must pass in x then y
                target = [unit_x[0], unit_y[0]]

                # preflag because going to select scv
                self.scv_selected = True
                # This time, function call includes select point of first scv
                return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])

            # Build supply depot
            elif _BUILD_SUPPLYDEPOT in obs.observation["available_actions"]:
                # find where the command center is
                unit_type = obs.observation["screen"][_UNIT_TYPE]
                # get the multiple coordinates of the command center
                unit_y, unit_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()

                # use transform location on the mean of the coordinates,
                # and select spot 20 squares from command center
                target = self.transformLocation(int(unit_x.mean()), 0, int(unit_y.mean()), 25)

                # preflag building supply depot
                self.supply_depot_built = True

                # return function call to build supply depot
                return actions.FunctionCall(_BUILD_SUPPLYDEPOT, [_NOT_QUEUED, target])

        # block for building barracks
        elif not self.barracks_built and _BUILD_BARRACKS in obs.observation["available_actions"]:
            unit_type = obs.observation["screen"][_UNIT_TYPE]
            unit_y, unit_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()

            target = self.transformLocation(int(unit_x.mean()), 15, int(unit_y.mean()), 15)

            self.barracks_built = True

            return actions.FunctionCall(_BUILD_BARRACKS, [_NOT_QUEUED, target])

        # code for selecting barracks rally point
        elif not self.barracks_rallied:
            if not self.barracks_selected:
                unit_type = obs.observation["screen"][_UNIT_TYPE]
                unit_y, unit_x = (unit_type == _TERRAN_BARRACKS).nonzero()

                if unit_y.any():
                    target = [int(unit_x.mean()), int(unit_y.mean())]

                    self.barracks_selected = True

                    return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])
            else:
                self.barracks_rallied = True

                if self.base_top_left:
                    return actions.FunctionCall( _RALLY_UNITS_MINIMAP, [_NOT_QUEUED, [29, 21]])

                return actions.FunctionCall(_RALLY_UNITS_MINIMAP, [_NOT_QUEUED, [29, 21]])

        elif obs.observation["player"][_SUPPLY_USED] < obs.observation["player"][_SUPPLY_MAX] and _TRAIN_MARINE in obs.observation["available_actions"]:
             return actions.FunctionCall(_TRAIN_MARINE, [_QUEUED])

        elif not self.army_rallied:
            if not self.army_selected:
                if _SELECT_ARMY in obs.observation["available_actions"]:
                    self.army_selected = True
                    self.barracks_selected = False

                    return actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])
            elif _ATTACK_MINIMAP in obs.observation["available_actions"]:
                self.army_rallied = True
                self.army_selected = False

                if self.base_top_left:
                    return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, [39, 45]])


                return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, [21, 24]])
        return actions.FunctionCall(_NOOP, [])
