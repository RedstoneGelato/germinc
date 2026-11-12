"""Starts the striker robot. On the Pi:

    python3 main_attack.py          RCJA field (strategy.py)
    python3 main_attack.py --int    international field (strategy_int.py)
"""
import sys

if "--int" in sys.argv:      # must come first: the imports below then get the international field and strategy
    import field_int
    field_int.install_international()

import robot_config as cfg  # noqa: E402
from common import Brain, run_robot  # noqa: E402
from strategy import StrikerStrategy  # noqa: E402


class StrikerBrain(Brain):
    idle_state = {"bot active": 0, "command": 1}

    def __init__(self, bot, log=print):
        super().__init__(bot, StrikerStrategy(), log)

    def extra_state(self):
        return {"command": self.strategy.command}


if __name__ == "__main__":
    run_robot(cfg.AttackRobot, StrikerBrain)
