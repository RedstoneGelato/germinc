"""
main.py - DEFENSE (goalie) robot. main_attack.py is the striker.

GoalieBrain.tick() is one pass of the 100 Hz loop: everything the robot decides happens there.
simulator.py runs the very same GoalieBrain on a simulated field, so test logic changes there first.

The work lives in:
    robot_config.py        numbers shared by both robots + coordinate conventions
    hardware_defense.py    this robot's motor addresses / calibration, robot id
    common.py              start-up, standby (paused) calibration, shutdown
    vision.py, detection.py, localisation.py, perception.py, lines.py, motion.py, comms.py
    strategy.py            GoalieStrategy = what this robot decides to do

Before running: tune the camera in test_camera.py ON THIS ROBOT and press "Generate config file"
(robot_vision_config.json is per robot), then check detections + localisation in test_localisation.py.
"""
import time
import traceback

import robot_config as cfg
from common import keep_rate
from strategy import GoalieStrategy


class GoalieBrain:
    def __init__(self, bot, log=print):
        self.bot = bot #common.Robot (or simulator.SimBot)
        self.log = log
        self.strategy = GoalieStrategy(has_dribbler=bot.motors.has_dribbler)
        self.robot_active = None
        self.status = ""

    def tick(self, paused):
        bot = self.bot
        strategy = self.strategy
        colours = bot.pcb.snapshot() #pull variables from sensors
        frame_id, det = bot.vision.frame_id, bot.vision.latest()

#----------------------------------------------------------------------
#            pause and unpause bot (switch on = paused)
#----------------------------------------------------------------------
        if paused: #paused bot
            if self.robot_active or self.robot_active is None:
                self.log("Paused")
                self.robot_active = False
                strategy.reset()
                bot.on_pause()
            bot.comms.my_state.update({"bot active": 0}) # comm say bot off
            bot.standby(colours, det)
            return
        if not self.robot_active:
            self.log(f"running, attacking {bot.attack}")
            self.robot_active = True #running bot

#----------------------------------------------------------------------
#            sensors -> world
#----------------------------------------------------------------------
        compass = bot.imu.compass() #bot heading
        pose = bot.loc.get()
        mate = bot.comms.teammate() #other bot's latest message, None if the bots aren't connected
        world = bot.world
        world.update(frame_id, det, pose, bot.attack, mate) #uses the teammate's ball when we can't see it
        line = bot.lines.update(colours, compass, world.line_pts, pose) #LDR ring + camera + pose

#----------------------------------------------------------------------
#            strategy -> motors, dribbler, kicker
#----------------------------------------------------------------------
        desired_pos, desired_heading, dribbler_on = strategy.update(world, compass, mate, line.touches)

        speeds = bot.mover.step(desired_pos, desired_heading, compass, line, pose, **strategy.speed,
                                dribbler_on=dribbler_on) #slower the more sideways/backwards it moves while dribbling
        bot.actuate(speeds, dribbler_on, strategy.kick)

#----------------------------------------------------------------------
#            comms to other bot
#----------------------------------------------------------------------
        bot.comms.my_state.update({
            "bot active": 1,
            "chasing": strategy.chasing, #going for the ball
            "pos": [round(pose.x), round(pose.y)] if pose.confident else None,
            "ball": [round(v) for v in world.ball_own_field] if world.ball_own_field else None, #own camera only
            "opps": [[round(v) for v in o] for o in world.opponents_own], #opponent robots our camera sees (centres)
        })

        self.desired_pos, self.line, self.dribbler_on = desired_pos, line, dribbler_on #kept for the simulator display
        self.status = (f"botstate={strategy.botstate} {strategy.phase} line={'ON ' + line.source if line.on_line else 'off'} "
                       f"pos=({pose.x:.0f},{pose.y:.0f})+-{pose.std:.0f} ball={world.ball}({world.ball_source}) goals={world.goal_source}")

#==========================================================================================#
#                                                                                          #
#                                   START OF ACTUAL CODE                                   #
#                                                                                          #
#==========================================================================================#

def main():
    from gpiozero import DigitalInputDevice
    import hardware_defense as hw
    from common import Robot

    script_activate_pin = DigitalInputDevice(cfg.SWITCH_PIN, pull_up = True) #gpio pin for on/off switch
    bot = Robot(hw)
    brain = GoalieBrain(bot)
    bot.wait_ready()

    print("Waiting for signal")
    try:
        next_loop = time.monotonic()
        while True:
            paused = script_activate_pin.is_active
            brain.tick(paused)
            if paused:
                time.sleep(0.02)
                next_loop = time.monotonic()
                continue
            print(brain.status)
            next_loop = keep_rate(next_loop) # Maintain a fixed 100 Hz loop

    except KeyboardInterrupt:
        print("User stopped.")
    except Exception as e:
        print(f"Unexpected error: {e!r}")
        traceback.print_exc()
    finally:
        bot.shutdown()


if __name__ == "__main__":
    main()
