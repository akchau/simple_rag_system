import logging
import os
import sys
from pathlib import Path
from typing import Callable

import pytest

# from src.logger import init_logging

sys.path.append(os.path.join(Path(__file__).resolve().parent, "src"))
sys.path.append(os.path.join(Path(__file__).resolve().parent, "tests"))


# def log():
#     init_logging()

DOUBLE_DASH_ARGS: dict[str, Callable] = {}
# DOUBLE_DASH_ARGS = {"log": log}

if __name__ == "__main__":
    raw_args = sys.argv
    args = []

    if "--log" not in raw_args:
        logging.disable()

    for arg in raw_args:
        if arg.startswith("--"):
            func = DOUBLE_DASH_ARGS.get(arg.replace("--", ""))

            if func is not None:
                func()
        else:
            args.append(arg)

    if len(args) == 1:
        exit_code = pytest.main(["-s", "-v"])
        sys.exit(exit_code)
    elif len(args) == 2:
        if args[1].startswith("-"):
            marker = f'not {args[1].replace("-", "")}'
        else:
            marker = args[1]
        exit_code = pytest.main(["-s", "-v", "-k", marker])
        sys.exit(exit_code)
    else:
        raise pytest.UsageError("Некорректное количество аргументов, можно передать два аргумента")
