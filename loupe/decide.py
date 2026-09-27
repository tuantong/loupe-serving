import argparse
import json
import os
from functools import cache
from pathlib import Path

from loupe.api import DecideRequest
from loupe.engine import Engine
from loupe.serve import build_engine


@cache
def default_engine() -> Engine:
    return build_engine(os.environ.get("LOUPE_BACKEND", "fake"))


def decide(state, questions: dict, policy: dict | None = None, engine: Engine | None = None) -> dict:
    request = DecideRequest(state=state, questions=questions, policy=policy or {})
    return (engine or default_engine()).decide(request).model_dump()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="loupe.decide")
    parser.add_argument("request", help="path to a JSON file with state, questions, and optional policy")
    args = parser.parse_args(argv)
    body = json.loads(Path(args.request).read_text())
    out = decide(body["state"], body["questions"], body.get("policy"))
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
