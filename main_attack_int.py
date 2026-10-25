"""
main_attack_int.py - the striker (main_attack.py) on the INTERNATIONAL rules: field_int.py +
strategy_int.py instead of field.py + strategy.py. Run this instead of main_attack.py there (deploy/germinc.service
runs main_attack.py: point it here).
"""
import sys

import field_int

field_int.install()                          # first: every module's `import field` gets field_int
import strategy_int  # noqa: E402

sys.modules["strategy"] = strategy_int       # main_attack.py's `from strategy import StrikerStrategy` gets strategy_int
import main_attack  # noqa: E402

if __name__ == "__main__":
    main_attack.main()
