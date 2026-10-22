"""
main_int.py - the goalie (main.py) on the INTERNATIONAL rules (RCJ Singapore): field_int.py + strategy_int.py instead
of field.py + strategy.py. Run this instead of main.py there (deploy/germinc.service runs main.py: point it here).
"""
import sys

import field_int

field_int.install()                          # first: every module's `import field` gets field_int
import strategy_int  # noqa: E402

sys.modules["strategy"] = strategy_int       # main.py's `from strategy import GoalieStrategy` gets strategy_int
import main  # noqa: E402

if __name__ == "__main__":
    main.main()
