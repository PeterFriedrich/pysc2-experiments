available# The base of this code is taken from the sjkb pysc2 agent tutorial at:
# https://itnext.io/build-a-sparse-reward-pysc2-agent-a44e94ba5255

# additions and modifications are made during my own learning process

# command to run the agent:

# python -m pysc2.bin.agent \
# --map Simple64 \
# --agent sparse_agent.SparseAgent \
# --agent_race T \
# --max_agent_steps 0 \
# --norender

# first line is standard
# second line is for map
# third line is for agent
# fourth is race
#

import random
import math
import os.path

import numpy as np
import pandas as pd

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.ib import features

_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_BUILD_SUPPLY_DEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id
_HARVEST_GATHER = actions.FUNCTIONS.Harvest_Gather_screen.id

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_PLAYER_ID = features.SCREEN_FEATURES.player_id.index

_PLAYER_SELF = 1
_PLAYER_HOSTILE = 4
_ARMY_SUPPLY = 5

_TERRAN_COMMANDCENTER = 18
_TERRAN_SCV = 45
_TERRAN_SUPPLY_DEPOT = 19
_TERRAN_BARRACKS = 21
_NEUTRAL_MINERAL_FIELD = 341

_NOT_QUEUED = [0]
_QUEUED = [1]
_SELECT_ALL = [2]

DATA_FILE = 'sparse_agent_data'

ACTION_DO_NOTHING = 'donothing'
ACTION_BUILD_SUPPLY_DEPOT = 'buildsupplydepot'
ACTION_BUILD_BARRACKS = 'buildbarracks'
ACTION_BUILD_MARINE = 'buildmarine'
ACTION_ATTACK = 'attack'

smart_actions = [
    ACTION_DO_NOTHING,
    ACTION_BUILD_SUPPLY_DEPOT,
    ACTION_BUILD_BARRACKS,
    ACTION_BUILD_MARINE,
]

for mm_x in range(0, 64):
    for mm_y in range(0, 64):
        if (mm_x + 1) % 32 == 0 and (mm_y + 1) % 32 == 0:
            smart_actions.append(ACTION_ATTACK + '_' + str(mm_x - 16) + '_' + str(mm_y - 16))


# Stolen from https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow
class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9)
        self.actions = actions # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrace(columns=self.actions, dtype=np.float64)

    def choose_action(self. observation):
        self.check_state_exist(observation)

        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.ix[observation, :]

            # some actions have the same value
            state_action = state_action.reindex(np.random.permutation(state_action.index))

            action = state_action.idmax()
        else:
            # choose random action
            actions = state_action.idmax()

        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        self.check_state_exist(s)

        q_predict = self.q_table.ix[s, a]

        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.ix[s_, :].max()
        else:
            q_target = r # next state is terminal

        # update
        self.q_table[s, a] += self.lr * (q_target - q_predict)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(pd.Series[0] * len(self.actions), index=self.q_table.columns, name=state))

class SparseAgent(base_agent.BaseAgent):
    def __init__(self):
        super(SparseAgent, self).__init__()

        self.qlearn = QlearningTable(actions = list(range(len(smart_actions))))

        self.previous_action = None
        self.previous_state = None

        self.cc_y = None
        self.cc_y = None

        self.move_number = 0

        if os.path.isfile(DATA_FILE + '.gz'):
            self.qlearn.q_table = pd.read_pickle(DATA_FILE + '.gz', compression='gzip')

            # transform function from previous agentss
            def transformDistance(self, x, x_distance, y, y_distance):
                if not self.base_top_left:
                    return [x - x_distance, y - y_distance]

                return [x + x_distance, y + y_distance]

            def transformLocation(self, x, y):
                if not self.base_top_left:
                    return [64 - x, 64 - y]

                return [x,y]

            # extracts information from selected action
            def splitAction(self, action_id):
                smart_action = smart_actions[action_id]

                x = 0
                y = 0
                if '_' in smart_action:
                    smart_action, x, y = smart_action.split('_')

                return (smart_action, x, y)

            def step(self, obs):
                super(SparseAgent, self).step(obs)

                # detects whether the last step is a win, and applies reward
                if obs.last():
                    reward = obs.reward

                    # pass in string terminal, saying its a special state, full reward, not discounted
                    self.qlearn.learn(str(self.previous_state), self.previous_action, reward, 'terminal')
                    # uses gzipped pickle file to store table data, after quiting game
                    self.qlearn.q_table.to_pickle(DATA_FILE + '.gz', 'gzip')

                    self.previous_action = None
                    self.previous_state = None

                    self.move_number = 0

                    return actions.FunctionCall(_NO_OP, [])

                unit_type = obs.observation['screen'][_UNIT_TYPE]

                if obs.first():
                    player_y, player_x = (obs.observation['minimap'][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
                    self.base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0

                    self.cc_y, self.cc_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()

                cc_y, cc_x = (unit_type == _TERRAN_COMMANDCENTER)
                cc_count = 1 if cc_y.any() else 0

                depot_y, depot_x = (unit_type == _TERRAN_SUPPLY_DEPOT).nonzero()
                supply_depot_count = int(round(len(depot_y)/69))

                barracks_y, barracks_x = (unit_type == _TERRAN_BARRACKS).nonzero()
                barracks_count = int(round(len(barracks_y)/137))

                # check if first step in multi step action
                if self.move_number == 0:
                    self.move_number += 1

                    # set state to include building type count and marine count
                    current_state = np.zeros(8)
                    current_state[0] = cc_count
                    current_state[1] = supply_depot_count
                    current_state[2] = barracks_count
                    current_state[3] = obs.observation['player'][_ARMY_SUPPLY]

                    # divides minimap into quadrants, mark as hot if
                    # enemy units are there
                    hot_squares = np.zeros(4)
                    enemy_y, enemy_x = (obs.observation['minimap'][_PLAYER_RELATIVE] == _PLAYER_HOSTILE).nonzero()
                    for i in range(0, len(enemy_y)):
                        y = int(math.ceil((enemy_y[i] + 1)/32))
                        x = int(math.ceil((enemy_x[i] + 1)/32))

                        hot_squares[((y - 1) * 2) + (x - 1)] = 1

                    if not self.base_top_left:
                        hot_squares = hot_squares[::-1]

                    for i in range(0,4):
                        current_state[i + 4] = hot_squares[i]

                    # learning is performed on every third game step
                    if self.previous_action is not None:
                        self.qlearn.learn(str(self.previous_state), self.previous_action, 0, str(current_state))

                    # an action is then chosen
                    rl_action = self.qlearn.choose_action(str(current_state))

                    self.previous_state = current_state
                    self.previous_action = rl_action

                    smart_action, x, y = self.splitAction(self.previous_action)

                    if smart_action == ACTION_BUILD_BARRACKS or smart_action == ACTION_BUILD_SUPPLY_DEPOT:
                        # first identify all scvs on screen
                        unit_y, unit_x = (unit_type == _TERRAN_SCV).nonzero()

                        # select an scv at random as target
                        if unit_y.any():
                            i = random.randint(0, len(unit_y) - 1)
                            target = [unit_x[]]

                            # select the random scv
                            return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])

                        elif smart_action == ACTION_BUILD_MARINE:
                            # first select the barracks
                            if barracks_y.any():
                                i = random.randint(0, len(barracks_y) - 1)
                                target = [barracks_x[i], barracks_y[i]]

                                # select all barracks
                                return actions.FunctionCall(_SELECT_POINT, [_SELECT_ALL, target])

                        elif smart_action == ACTION_ATTACK:
                            if _SELECT_ARMY in obs.observation['available_actions']:
                                # select army to use it for attacking
                                return actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])

        #----------starts adding next step of multi-step actions
                    elif self.move_number == 1:
                        # increment move number
                        self.move_number += 1

                        # extract action details
                        smart_action, x, y = self.splitAction(self.previous_action)


                        if smart_action == ACTION_BUILD_SUPPLY_DEPOT:
                            if supply_depot_count < 2 and _BUILD_SUPPLY_DEPOT in obs.observation['available_actions']:
                                if self.cc_y.any():
                                    # is going to use stored command center location
                                    if supply_depot_count == 0:
                                        target = self.transformDistance(round(self.cc_x.mean()), -35, round(self.cc_y.mean()), 0)
                                    elif supply_depot_count == 1:
                                        target = self.transformDistance(round(self.cc_x.mean()), -25, round(self.cc_y.mean()), -25)

                                    # build supply depot offset from stored command center location
                                    return actions.FunctionCall(_BUILD_SUPPLY_DEPOT, [_NOT_QUEUED, target])


                        elif smart_action == ACTION_BUILD_BARRACKS:
                            if barracks_count < 2 and _BUILD_BARRACKS in obs.observation['available_actions']:
                                if self.cc_y.any():
                                    if barracks_count == 0:
                                        target = self.transformDistance(round(self.cc_x.mean()), 15, round(self.cc_y.mean()), -9)
                                    elif barracks_count == 1:
                                        target = self.transformDistance(round(self.cc_x.mean()), 15, round(self.cc_y.mean()), 12)

                                    # build barracks in different location than supply, using same method
                                    return actions.FunctionCall(_BUILD_BARRACKS, [_NOT_QUEUED, target])

                        elif smart_action == ACTION_BUILD_MARINE:
                            if _TRAIN_MARINE in obs.observation['available_actions']:
                                # que the command so that barracks can train several marines
                                return actions.FunctionCall(_TRAIN_MARINE, [_QUEUED])

                        elif smart_action == ACTION_ATTACK:
                            do_it = True

                            # make sure scvs are not selected
                            if len(obs.observation['single_select']) > 0 and obs.observation['single_select'][0][0] == _TERRAN_SCV:
                                do_it = False
                            # same but for multi_select
                            if len(obs.observation['multi_select']) > 0 and obs.observation['multi_select'][0][0] == TERRAN_SCV:
                                do_it = False
                            # randomly choose area near a quadrant center
                            if do_it and _ATTACK_MINIMAP in obs.observation["available_actions"]:
                                x_offset = random.randint(-1, 1)
                                y_offset = random.randint(-1, 1)

                                # attack the chosen coordinates
                                return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, self.transformLocation(int(x) + (x_offset * 8), int(y) + (y_offset * 8))])

                    # remaining move is to send scvs back to a mineral patch, or
                    # if not scv, just do nothing
                    elif self.move_number == 2:
                        self.move_number = 0

                        smart_action, x, y = self.splitAction(self.previous_action)

                        if smart_action == ACTION_BUILD_BARRACKS or smart_action == ACTION_BUILD_SUPPLY_DEPOT:
                            if _HARVEST_GATHER in obs.observation['avaiable_actions']:
                                unit_y, unit_x = (unit_type == _NEUTRAL_MINERAL_FIELD).nonzero()

                                if unit_y.any():
                                    i = random.randint(0, len(unit_y) - 1)

                                    m_x = unit_x[i]
                                    m_y = unit_y[i]

                                    target = [int(m_x), int(m_y)]

                                    return actions.FunctionCall(_HARVEST_GATHER, [_QUEUED, target])

            return actions.FunctionCall(_NO_OP, [])
