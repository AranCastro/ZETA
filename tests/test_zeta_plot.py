from __future__ import annotations

import numpy as np
import pandas as pd

from zeta.plot import (
    prepare_amphibole_leake_data,
    prepare_feldspar_ternary_data,
    prepare_garnet_composition_data,
    prepare_pyroxene_quad_data,
)


def test_prepare_pyroxene_quad_data_from_apfu() -> None:
    frame = pd.DataFrame(
        [
            {"sample_id": "PX-1", "mineral": "pyroxene", "apfu_Ca": 0.90, "apfu_Mg": 0.80, "apfu_Fe2": 0.30}
        ]
    )
    out = prepare_pyroxene_quad_data(frame)

    assert len(out) == 1
    total = out[["wo_pct", "en_pct", "fs_pct"]].sum(axis=1).iloc[0]
    assert np.isclose(total, 100.0, atol=1e-6)
    assert 0.0 <= float(out.loc[0, "quad_x_fs_in_en_fs"]) <= 100.0


def test_prepare_amphibole_leake_groups() -> None:
    frame = pd.DataFrame(
        [
            {"sample_id": "AMP-C", "apfu_Na": 0.2, "apfu_Ca": 1.8, "apfu_Si": 7.5},
            {"sample_id": "AMP-S", "apfu_Na": 1.8, "apfu_Ca": 0.2, "apfu_Si": 7.8},
        ]
    )
    out = prepare_amphibole_leake_data(frame)

    groups = set(out["amphibole_group"].tolist())
    assert "Calcic" in groups
    assert "Sodic" in groups


def test_prepare_feldspar_ternary_data_sums_to_100() -> None:
    frame = pd.DataFrame(
        [
            {"sample_id": "FSP-1", "apfu_Ca": 0.25, "apfu_Na": 0.65, "apfu_K": 0.10},
        ]
    )
    out = prepare_feldspar_ternary_data(frame)

    total = out[["an_pct", "ab_pct", "or_pct"]].sum(axis=1).iloc[0]
    assert np.isclose(total, 100.0, atol=1e-6)


def test_prepare_garnet_composition_identifies_dominant_endmember() -> None:
    frame = pd.DataFrame(
        [
            {"sample_id": "GRT-1", "apfu_Mg": 0.40, "apfu_Fe2": 1.70, "apfu_Ca": 0.60, "apfu_Mn": 0.30}
        ]
    )
    out = prepare_garnet_composition_data(frame)

    assert out.loc[0, "dominant_endmember"] == "Almandine"
    total = out[["prp_pct", "alm_pct", "grs_pct", "sps_pct"]].sum(axis=1).iloc[0]
    assert np.isclose(total, 100.0, atol=1e-6)
