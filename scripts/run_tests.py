"""Run all unit tests via unittest discovery."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    import argparse

    p = argparse.ArgumentParser(description="Run unit tests")
    p.add_argument("--start-dir", default="tests", help="Directory to start discovery")
    p.add_argument("--pattern", default="test*.py", help="Test file pattern")
    p.add_argument("-v", "--verbosity", type=int, default=2, help="Verbosity level")
    p.add_argument("-f", "--failfast", action="store_true", help="Stop on first failure")
    args = p.parse_args()

    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(ROOT / args.start_dir), pattern=args.pattern, top_level_dir=str(ROOT))
    runner = unittest.TextTestRunner(verbosity=args.verbosity, failfast=args.failfast)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()

