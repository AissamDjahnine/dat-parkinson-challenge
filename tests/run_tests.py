"""Zero-dependency test runner.

`pytest` is not required and will not exist in the competition
runtime either, so the suite is written as plain `test_*` functions with bare
asserts and this runner discovers them. If pytest is ever installed it collects
the same files unchanged -- the two are compatible, not alternatives.

    python tests/run_tests.py            # everything
    python tests/run_tests.py aug nets    # only matching modules

Tests that need the real dataset or a GPU skip themselves by raising
`SkipTest`, so the suite stays green on a bare checkout.
"""

from __future__ import annotations

import importlib.util
import sys
import time
import traceback
from pathlib import Path

TESTS = Path(__file__).resolve().parent
ROOT = TESTS.parent
sys.path.insert(0, str(ROOT))


class SkipTest(Exception):
    """Raise to skip a test that needs data/GPU that is not present."""


# Test modules do `from run_tests import SkipTest`. When this file runs as
# __main__ the name "run_tests" is NOT in sys.modules, so that import would load a
# SECOND copy with a different SkipTest class and skips would be caught as
# failures. Registering ourselves under both names keeps the class identity single.
sys.modules.setdefault("run_tests", sys.modules[__name__])
sys.modules.setdefault("_testkit", sys.modules[__name__])


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = mod
    spec.loader.exec_module(mod)
    return mod


def main(argv: list[str]) -> int:
    files = sorted(TESTS.glob("test_*.py"))
    if argv:
        files = [f for f in files if any(a in f.stem for a in argv)]
    if not files:
        print("no test modules matched")
        return 1

    passed = failed = skipped = 0
    failures: list[tuple[str, str]] = []
    t_all = time.time()

    for f in files:
        print(f"\n\033[1m{f.stem}\033[0m")
        try:
            mod = _load(f)
        except Exception:
            print("  !! import failed")
            failures.append((f.stem, traceback.format_exc()))
            failed += 1
            continue
        for name in sorted(n for n in dir(mod) if n.startswith("test_")):
            fn = getattr(mod, name)
            if not callable(fn):
                continue
            t0 = time.time()
            try:
                fn()
            except SkipTest as e:
                print(f"  \033[33mSKIP\033[0m {name}  ({e})")
                skipped += 1
            except Exception:
                dt = time.time() - t0
                print(f"  \033[31mFAIL\033[0m {name}  ({dt:.2f}s)")
                failures.append((f"{f.stem}::{name}", traceback.format_exc()))
                failed += 1
            else:
                print(f"  \033[32mok\033[0m   {name}  ({time.time() - t0:.2f}s)")
                passed += 1

    for name, tb in failures:
        print(f"\n{'=' * 70}\nFAILED {name}\n{'=' * 70}\n{tb}")

    print(f"\n{passed} passed, {failed} failed, {skipped} skipped "
          f"in {time.time() - t_all:.1f}s")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
