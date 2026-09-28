from __future__ import annotations

import pytest

from b77.cli import COMMANDS, parser


def test_every_subcommand_parses_and_has_a_handler() -> None:
    p = parser()
    for argv in (
        ["data"],
        ["splits", "--check"],
        ["embed"],
        ["baselines"],
        ["timing", "--steps", "5"],
        ["smoke", "--model", "qwen3-0.6b"],
        ["sweep"],
        ["train-final", "--seed", "1"],
        ["train-modernbert"],
        ["learning-curve", "--k", "5", "10"],
        ["prompt", "--arm", "sol-fewshot", "--live", "--cap", "20"],
        ["estimate"],
        ["report"],
    ):
        args = p.parse_args(argv)
        assert args.command in COMMANDS


def test_live_and_fake_are_exclusive() -> None:
    with pytest.raises(SystemExit):
        parser().parse_args(["prompt", "--arm", "luna-zeroshot", "--live", "--fake"])
