# This is a basic protoss proxy stargate bot.
# The base of this code is originally from skjb's pysc2 tutorial.
# https://github.com/skjb/pysc2-tutorial
# The code is specifically at https://github.com/skjb/pysc2-tutorial/blob/master/Building%20a%20Basic%20Agent/simple_agent.py
# The blogpost is https://chatbotslife.com/building-a-basic-pysc2-agent-b109cde1477c

# implement agent from its directory
# shell command to start agent is:
# python -m pysc2.bin.agent --map Simple64 --agent protoss_cheese_a.SimpleAgent --agent_race P


# ------------------------------------------------------------------------------
# http://liquipedia.net/starcraft2/2_Gate_Zealot_Rush
# 8 Pylon[1]
# 8 Gateway
# 8 Gateway
# 8 Probe
# 9 Probe [2]
# 10 Zealot[3]
# 12 Zealot[4]
# 14 Zealot[5]
# 16 Zealot[6]
# Continue Zealot production until the attack is over. Build extra Pylons where
# necessary; note that sometimes even if your supply is capped it is best not
# to build a new Pylon if you have barely 100 minerals and one of your Zealots
# is about to die.
# 1 ↑ If you wish to proxy your Gateways, send either your first produced probe,
# one of your starting probes, or allow your probes to mine one cycle and then
# send the first probe back out to proxy location. When to send the probe will
# depend on which map and proxy location you choose.
# 2 ↑ Depending on map, you may cut this probe out; assuming your scout/proxy
# probe dies, this will leave you with an even supply to build Zealots with.
# 3 ↑ Chrono boost this Zealot.
# 4 ↑ Chrono boost this Zealot.
# 5 ↑ Chrono boost this Zealot.
# 6 ↑ Chrono boost this Zealot.
# ------------------------------------------------------------------------

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

import time

# Functions
_BUILD_PYLON = actions.FUNCTIONS.Build_Pylon_screen.id
_BUILD_GATEWAY = actions.FUNCTIONS.Build_Gateway_screen.id
_NOOP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id

# Features
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index

# Unit IDs
_PROTOSS_NEXUS = 59
_PROTOSS_PROBE = 84
_PROTOSS_GATEWAY = 62

# Parameters
_PLAYER_SELF = 1
_NOT_QUEUED = [0]
_QUEUED = [1]



class SimpleAgent(base_agent.BaseAgent):
    base_top_left = None
    pylon_built = False
    probe_selected = False

    # projects coordinates, relative to base top or bottom
    def transformLocation(self, x, x_distance, y, y_distance):
        if not self.base_top_left:
            return [x - x_distance, y - y_distance]

        return [x + x_distance, y + y_distance]

    # function for taking actions at each step
    def step(self, obs):
        super(SimpleAgent, self).step(obs)

        time.sleep(0.10)

        # check where base is
        if self.base_top_left is None:
            player_y, player_x = (obs.observation["minimap"][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
            self.base_top_left = player_y.mean() <= 31

        # select probe
        if not self.pylon_built:
            if not self.probe_selected:
                unit_type = obs.observation["screen"][_UNIT_TYPE]
                unit_y, unit_x = (unit_type == _PROTOSS_PROBE).nonzero()
                print('CHECK HERE FOR ERROR',unit_x,unit_y)
                target = [unit_x[0], unit_y[0]]

                self.probe_selected = True

                return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])

            # build first pylon
            elif _BUILD_PYLON in obs.observation["available_actions"]:
                unit_type = obs.observation["screen"][_UNIT_TYPE]
                unit_y, unit_x = (unit_type == _PROTOSS_NEXUS).nonzero()

                target = self.transformLocation(int(unit_x.mean()), 30, int(unit_y.mean()), 0)

                self.pylon_built = True

                return actions.FunctionCall(_BUILD_PYLON, [_NOT_QUEUED, target])

        return actions.FunctionCall(_NOOP, [])
