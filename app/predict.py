"""Inference core for the Bosch failure-prediction service.

Loads the artifacts produced by the notebook and turns a raw part record (a
sparse map of `L#_S#_F#` / `_D#` measurements) into a failure prediction. The
record is expanded to the canonical column universe and run through the *same*
`src.features.build_features` used in training — that is the train/serve parity
guarantee.
"""
from __future__ import annotations

import json
from functools import lru_cache

import joblib
import pandas as pd

from src import config, features as fe


class Predictor:
    def __init__(self):
        self.model = joblib.load(config.MODEL_PKL_PATH)
        self.feature_list: list[str] = json.loads(config.FEATURE_LIST_PATH.read_text())
        self.raw_columns: dict[str, list[str]] = json.loads(config.RAW_COLUMNS_PATH.read_text())
        self.threshold: float = float(json.loads(config.THRESHOLD_PATH.read_text())["threshold"])
        # fast membership lookups for routing incoming keys to the right block
        self._sets = {k: set(v) for k, v in self.raw_columns.items()}

    def _record_to_blocks(self, record: dict, part_id: int):
        """Split one sparse record into canonical numeric/date/categorical frames."""
        routed = {"numeric": {}, "date": {}, "categorical": {}}
        for col, val in record.items():
            for block, members in self._sets.items():
                if col in members:
                    routed[block][col] = val
                    break  # a column belongs to exactly one block
        idx = pd.Index([part_id], name=config.ID_COL)
        blocks = {}
        for block in ("numeric", "date", "categorical"):
            df = pd.DataFrame([routed[block]], index=idx)
            df = fe.reindex_to_canonical(df, self.raw_columns[block])
            # Match the training loader's dtype exactly: numeric/date are read as
            # float32 there. Trees split on thresholds, so a float32-vs-float64
            # gap could flip a split — cast so inference features are identical.
            if block in ("numeric", "date"):
                df = df.astype("float32")
            blocks[block] = df
        return blocks

    def predict_one(self, record: dict, part_id: int = 0) -> dict:
        blocks = self._record_to_blocks(record, part_id)
        feat = fe.build_features(blocks["numeric"], blocks["date"], blocks["categorical"])
        X = fe.align_to_feature_list(feat, self.feature_list)
        proba = float(self.model.predict_proba(X)[:, 1][0])
        return {
            "id": part_id,
            "response": int(proba >= self.threshold),
            "probability": proba,
            "threshold": self.threshold,
        }


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    """Load artifacts once and reuse across requests."""
    return Predictor()
