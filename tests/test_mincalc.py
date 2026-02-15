from __future__ import annotations

import numpy as np
import pandas as pd

from zeta.core import (
    build_georoc_table,
    build_ternary_table,
    match_oxide_column,
    process_dataframe,
)


def test_match_oxide_aliases() -> None:
    assert match_oxide_column("SiO2 wt%") == "SiO2"
    assert match_oxide_column("FeO(T)") == "FeO"
    assert match_oxide_column("FeOT") == "FeO"


def test_auto_detects_garnet() -> None:
    frame = pd.DataFrame(
        [
            {
                "sample": "GRT-1",
                "SiO2": 37.3,
                "Al2O3": 21.1,
                "FeOT": 30.4,
                "MgO": 5.3,
                "CaO": 4.7,
                "MnO": 1.1,
            }
        ]
    )

    processed = process_dataframe(frame, mineral_mode="auto", sample_id_column="sample")

    assert processed.loc[0, "mineral"] == "garnet"
    assert processed.loc[0, "apfu_Si"] > 2.5
    assert processed.loc[0, "apfu_Si"] < 3.4


def test_manual_pyroxene_ternary_and_georoc_totals() -> None:
    frame = pd.DataFrame(
        [
            {
                "id": "CPX-1",
                "SiO2": 51.0,
                "TiO2": 0.4,
                "Al2O3": 1.3,
                "FeO": 9.2,
                "MgO": 16.2,
                "CaO": 22.7,
                "Na2O": 0.4,
            }
        ]
    )

    processed = process_dataframe(
        frame,
        mineral_mode="manual",
        manual_mineral="pyroxene",
        sample_id_column="id",
    )

    ternary = build_ternary_table(processed)
    georoc = build_georoc_table(processed)

    assert len(ternary) == 1
    total = float(ternary.loc[0, "wo"] + ternary.loc[0, "en"] + ternary.loc[0, "fs"])
    assert np.isclose(total, 1.0, atol=1e-6)

    expected_total = georoc[["SiO2", "TiO2", "Al2O3", "FeO", "MgO", "CaO", "Na2O"]].sum(axis=1).iloc[0]
    assert np.isclose(georoc.loc[0, "TOTAL"], expected_total, atol=1e-6)
