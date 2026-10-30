#!/usr/bin/env python3
from stop_robot import run


def load_hardware():
    import hardware_attack
    return hardware_attack


if __name__ == "__main__":
    run(load_hardware)
