from __future__ import annotations

import sys

from scripts.train_ml_tickers import parse_args


def test_cli_parses_new_algorithm_arguments(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_ml_tickers.py",
            "--tickers",
            "SSI",
            "--algorithms",
            "cart,lstm,bilstm",
            "--primary-algorithm",
            "lstm",
            "--sequence-length",
            "20",
            "--epochs",
            "50",
            "--batch-size",
            "32",
        ],
    )

    args = parse_args()

    assert args.algorithms == "cart,lstm,bilstm"
    assert args.primary_algorithm == "lstm"
    assert args.sequence_length == 20
    assert args.epochs == 50
    assert args.batch_size == 32
