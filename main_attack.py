"""
main_attack.py - ATTACK (striker) robot. main.py is the goalie.

StrikerBrain.tick() is one pass of the 100 Hz loop: everything the robot decides happens there.
simulator.py runs the very same StrikerBrain on a simulated field, so test logic changes there first.

The work lives in:
    robot_config.py        numbers shared by both robots + coordinate conventions
    hardware_attack.py     this robot's motor addresses / calibration, robot id
    common.py              start-up, standby (paused) calibration, shutdown
    vision.py, detection.py, localisation.py, perception.py, lines.py, motion.py, comms.py
    strategy.py            StrikerStrategy = what this robot decides to do

Before running: tune the camera in test_camera.py ON THIS ROBOT and press "Generate config file"
(robot_vision_config.json is per robot: each camera has its own lens, mounting and robot centre).
"""
from common import CameraWatch
from utils import wrap_pi
from strategy import StrikerStrategy


class StrikerBrain:
    def __init__(self, bot, log=print):
        self.bot = bot #common.Robot (or simulator.SimBot)
        self.log = log
        self.strategy = StrikerStrategy()
        self.robot_active = None
        self.status = ""
        self.camera = CameraWatch()

    def tick(self, paused):
        bot = self.bot
        strategy = self.strategy
        colours = bot.pcb.snapshot() #pull variables from sensors
        frame_id, det = bot.vision.frame_id, bot.vision.latest()
        blind = self.camera.blind(frame_id)

#----------------------------------------------------------------------
#            pause and unpause bot (switch on = paused)
#----------------------------------------------------------------------
        if paused: #paused bot
            if self.robot_active or self.robot_active is None:
                self.log("Paused")
                self.robot_active = False
                strategy.reset()
                bot.on_pause()
            # bot off (likely damage or 30 sec penalty) -> tell the goalie to get the ball
            bot.comms.my_state.update({"bot active": 0, "command": 1})
            bot.standby(colours, det)
            return
        if not self.robot_active:
            self.log(f"running, attacking {bot.attack}")
            self.robot_active = True #running bot
        if blind: # camera frozen / restarting: everything we know is from the last frame, so stand still
            bot.actuate((0, 0, 0, 0), getattr(self, "dribbler_on", False), False) # (keep holding a ball we have)
            bot.comms.my_state.update({"bot active": 0, "command": 1}) # to the teammate we're as good as paused
            self.status = "camera blind - standing still"
            return

#----------------------------------------------------------------------
#            sensors -> world
#----------------------------------------------------------------------
        pose = bot.loc.get()
        compass = wrap_pi(bot.imu.compass() + pose.dh) #bot heading, with localisation's correction from the lines
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
            "command": strategy.command, #1 = goalie go get the ball, 0 = goalie stay in goal
            "chasing": strategy.chasing, #going for the ball
            "has ball": world.ball_in_capture, #ball in our dribbler
            "pos": [round(pose.x), round(pose.y)] if pose.confident else None,
            "ball": [round(v) for v in world.ball_own_field] if world.ball_own_field else None, #own camera only
            "opps": [[round(v) for v in o] for o in world.opponents_own], #opponent robots our camera sees (centres)
        })

        self.desired_pos, self.line, self.dribbler_on = desired_pos, line, dribbler_on #kept for the simulator display
        self.status = (f"botstate={strategy.botstate} {strategy.phase} ball:{strategy.possession or '-'} line={'ON ' + line.source if line.on_line else 'off'} "
                       f"pos=({pose.x:.0f},{pose.y:.0f})+-{pose.std:.0f} ball={world.ball}({world.ball_source}) goals={world.goal_source}")

#==========================================================================================#
#                                                                                          #
#                                   START OF ACTUAL CODE                                   #
#                                                                                          #
#==========================================================================================#

def main():
    import hardware_attack as hw
    from common import run_robot
    run_robot(hw, StrikerBrain) # keeps going if tick() crashes, restarts a dead camera / comms thread (see common.py)


if __name__ == "__main__":
    main()
