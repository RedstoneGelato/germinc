from common import Brain
from strategy import GoalieStrategy, StrikerStrategy


class GoalieBrain(Brain):
    def __init__(self, bot, log=print):
        strategies = {"defend": GoalieStrategy(has_dribbler=bot.motors.has_dribbler), "attack": StrikerStrategy()}
        super().__init__(bot, strategies, "defend", log)


def main():
    import hardware_defense as hw
    from common import run_robot
    run_robot(hw, GoalieBrain)


if __name__ == "__main__":
    main()
