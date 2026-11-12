"""Starts the goalie robot. On the Pi:

    python3 main_defence.py          RCJA field (strategy.py)
    python3 main_defence.py --int    international field (strategy_int.py)
"""
import sys

if "--int" in sys.argv:      # must come first: the imports below then get the international field and strategy
    import field_int
    field_int.install_international()

import robot_config as cfg  # noqa: E402
from common import Brain, run_robot  # noqa: E402
from strategy import GoalieStrategy  # noqa: E402


class GoalieBrain(Brain):
    def __init__(self, bot, log=print):
        super().__init__(bot, GoalieStrategy(has_dribbler=bot.motors.has_dribbler), log)

    def extra_state(self):
        return {"clear": getattr(self.strategy, "clear_plan", None)}


if __name__ == "__main__":
    run_robot(cfg.DefenseRobot, GoalieBrain)
