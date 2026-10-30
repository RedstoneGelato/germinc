import sys

import field_int

field_int.install()
import strategy_int  # noqa: E402

sys.modules["strategy"] = strategy_int
import main_attack  # noqa: E402

if __name__ == "__main__":
    main_attack.main()
