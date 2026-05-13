import select
import sys

def read_input():
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.readline().strip()
    return None

while True:
    user_input = read_input()
    if user_input is not None:
        print(user_input)