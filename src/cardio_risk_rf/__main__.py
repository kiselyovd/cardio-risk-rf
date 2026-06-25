"""CLI entrypoint: python -m cardio_risk_rf"""

from __future__ import annotations

import sys


def main() -> int:
    print("cardio-risk-rf - use make train / make evaluate / make serve")
    return 0


if __name__ == "__main__":
    sys.exit(main())
