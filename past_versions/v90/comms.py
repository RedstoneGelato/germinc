import json
import socket
import threading
import time

import robot_config as cfg

BROADCAST = ("255.255.255.255", cfg.COMMS_PORT)
MAX_MESSAGE_BYTES = 1024
RECEIVE_TIMEOUT = 0.02
MAX_RECEIVE_ERRORS = 50


class TeammateLinkThread(threading.Thread):
    def __init__(self, robot_id, send_interval=0.05):
        super().__init__()
        self.robot_id = robot_id
        self.daemon = True
        self.running = True
        self.enabled = True
        self.send_interval = send_interval

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.sock.bind(("", cfg.COMMS_PORT))
        self.sock.settimeout(RECEIVE_TIMEOUT)

        self.teammate_state = {}
        self.teammate_last_seen = 0
        self.last_loop_t = time.monotonic()
        self.my_state = {"bot active": 0}

    def run(self):
        last_send = 0
        recv_errors = 0
        while self.running:
            now = time.monotonic()
            self.last_loop_t = now

            if self.enabled and now - last_send >= self.send_interval:
                try:
                    msg = {"team": cfg.TEAM_ID, "robot": self.robot_id, **self.my_state}
                    self.sock.sendto(json.dumps(msg).encode("utf-8"), BROADCAST)
                except (OSError, TypeError, ValueError) as e:
                    print(f"Comms send error: {e!r}")
                last_send = now

            try:
                data, _ = self.sock.recvfrom(MAX_MESSAGE_BYTES)
                msg = json.loads(data.decode("utf-8"))
                if (self.enabled and isinstance(msg, dict)
                        and msg.get("team") == cfg.TEAM_ID
                        and msg.get("robot") != self.robot_id):
                    self.teammate_state = msg
                    self.teammate_last_seen = now
                recv_errors = 0
            except socket.timeout:
                recv_errors = 0
            except ValueError:
                pass
            except OSError as e:
                if not self.running:
                    break
                recv_errors += 1
                if recv_errors >= MAX_RECEIVE_ERRORS:
                    print(f"Comms receive keeps failing ({e!r}) - stopping the comms thread")
                    break
                time.sleep(RECEIVE_TIMEOUT)

    def teammate(self, max_age=0.5):
        if isinstance(self.teammate_state, dict) and time.monotonic() - self.teammate_last_seen < max_age:
            return self.teammate_state
        return None

    def stop(self):
        self.running = False
        self.sock.close()
