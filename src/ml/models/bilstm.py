"""Bidirectional LSTM model."""

from __future__ import annotations

from .lstm import LstmModel


class BiLstmModel(LstmModel):
    """True bidirectional LSTM implementation."""

    algorithm_name = "bilstm"
    bidirectional = True
