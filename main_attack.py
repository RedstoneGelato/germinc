from common import Brain
from strategy import StrikerStrategy


class StrikerBrain(Brain):
    idle_state = {"bot active": 0, "command": 1}

    def __init__(self, bot, log=print):
        super().__init__(bot, StrikerStrategy(), log)

    def extra_state(self):
        return {"command": self.strategy.command}


def main():
    import hardware_attack as hw
    from common import run_robot
    run_robot(hw, StrikerBrain)


if __name__ == "__main__":
    main()
