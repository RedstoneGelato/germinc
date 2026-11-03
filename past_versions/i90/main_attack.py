from common import Brain
from strategy import GoalieStrategy, StrikerStrategy


class StrikerBrain(Brain):
    def __init__(self, bot, log=print):
        strategies = {"defend": GoalieStrategy(has_dribbler=bot.motors.has_dribbler), "attack": StrikerStrategy()}
        super().__init__(bot, strategies, "attack", log)


def main():
    import hardware_attack as hw
    from common import run_robot
    run_robot(hw, StrikerBrain)


if __name__ == "__main__":
    main()
