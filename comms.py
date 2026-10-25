"""
comms.py - UDP broadcast link between our two robots.
"""
import json
import socket
import threading
import time

import robot_config as cfg


class TeammateLinkThread(threading.Thread): #comms between bots
    def __init__(self, robot_id, send_interval=0.05):
        super().__init__()
        self.robot_id = robot_id #1 = striker, 2 = goalie
        self.daemon = True
        self.running = True
        self.enabled = True  # set False to disable comms (referee request)

        self.send_interval = send_interval

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.sock.bind(("", cfg.COMMS_PORT))
        self.sock.settimeout(0.02)

        self.teammate_state = {} #most recent info from teammate
        self.teammate_last_seen = 0
        self.last_loop_t = time.monotonic() #watchdog (common.Robot): restarted if this stops moving
        self.my_state = {"bot active": 0} #what robot wants to tell its teammate

    def run(self):
        last_send = 0
        recv_errors = 0
        while self.running:
            now = time.monotonic()
            self.last_loop_t = now

            if self.enabled and now - last_send >= self.send_interval:
                try:
                    msg = {"team": cfg.TEAM_ID, "robot": self.robot_id, **self.my_state}
                    self.sock.sendto(json.dumps(msg).encode("utf-8"), ("255.255.255.255", cfg.COMMS_PORT)) #send message
                except (OSError, TypeError, ValueError) as e: #network down, or something in my_state json can't send
                    print(f"Comms send error: {e!r}")
                last_send = now

            try:
                data, _ = self.sock.recvfrom(1024)
                msg = json.loads(data.decode("utf-8"))
                if (self.enabled and isinstance(msg, dict)
                        and msg.get("team") == cfg.TEAM_ID #message from bot of the same team
                        and msg.get("robot") != self.robot_id): #ignore a possible echo of own broadcast
                    self.teammate_state = msg #receive message
                    self.teammate_last_seen = now
                recv_errors = 0
            except socket.timeout:
                recv_errors = 0
            except ValueError: #bad json / utf-8 from someone else on the network
                pass
            except OSError as e:
                if not self.running:
                    break
                recv_errors += 1
                if recv_errors >= 50: #socket broken: end the thread, the watchdog (common.Robot) starts a new one
                    print(f"Comms receive keeps failing ({e!r}) - stopping the comms thread")
                    break
                time.sleep(0.02)

    def teammate(self, max_age=0.5):
        """Teammate's last message if it's fresh, else None."""
        if isinstance(self.teammate_state, dict) and time.monotonic() - self.teammate_last_seen < max_age:
            return self.teammate_state
        return None

    def stop(self):
        self.running = False
        self.sock.close()
