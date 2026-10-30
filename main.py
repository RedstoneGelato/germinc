from common import Brain
from strategy import GoalieStrategy


class GoalieBrain(Brain):
    def __init__(self, bot, log=print):
        super().__init__(bot, GoalieStrategy(has_dribbler=bot.motors.has_dribbler), log)

    def extra_state(self):
        return {"clear": getattr(self.strategy, "clear_plan", None)}


def main():
    import hardware_defense as hw
    from common import run_robot
    run_robot(hw, GoalieBrain)


if __name__ == "__main__":
    main()
