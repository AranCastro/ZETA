from __future__ import annotations

import numpy as np
import pandas as pd

from zeta.power_tools import (
    analyze_zoning_profile,
    classify_nomenclature,
    compute_thermobarometry,
    estimate_fe3_by_charge_balance,
    normalize_apfu,
    project_endmembers,
    renormalize_oxides,
)


def test_classify_nomenclature_generates_specific_names() -> None:
    frame = pd.DataFrame(
        [
            {
                "sample_id": "AMP-1",
                "mineral": "amphibole",
                "apfu_Si": 7.1,
                "apfu_Ca": 1.8,
                "apfu_Na": 0.2,
                "apfu_Mg": 2.2,
                "apfu_Fe2": 1.0,
            },
            {
                "sample_id": "PX-1",
                "mineral": "pyroxene",
                "apfu_Ca": 0.45,
                "apfu_Mg": 0.50,
                "apfu_Fe2": 0.05,
            },
        ]
    )

    out = classify_nomenclature(frame)

    amp_name = out.loc[out["sample_id"] == "AMP-1", "amphibole_name"].iloc[0]
    pyx_name = out.loc[out["sample_id"] == "PX-1", "pyroxene_name"].iloc[0]

    assert amp_name in {"Magnesiohornblende", "Ferrohornblende", "Tschermakite", "Ferrotschermakite"}
    assert pyx_name != ""


def test_analyze_zoning_profile_detects_oscillatory_pattern() -> None:
    frame = pd.DataFrame(
        {
            "distance_um": [0, 10, 20, 30, 40, 50],
            "Fe": [10, 13, 9, 14, 8, 15],
            "Mg": [12, 11, 10, 9, 8, 7],
        }
    )

    summary, profiles = analyze_zoning_profile(
        traverse_frame=frame,
        distance_col="distance_um",
        element_cols=["Fe", "Mg"],
        normal_direction="decreasing",
    )

    assert "distance" in profiles.columns
    fe_type = summary.loc[summary["element"] == "Fe", "zoning_type"].iloc[0]
    mg_type = summary.loc[summary["element"] == "Mg", "zoning_type"].iloc[0]
    assert fe_type == "oscillatory"
    assert mg_type in {"normal", "reverse"}


def test_fe3_recalc_and_normalizations() -> None:
    frame = pd.DataFrame(
        [
            {
                "sample_id": "GRT-1",
                "oxygen_basis": 12.0,
                "apfu_Si": 3.0,
                "apfu_Al": 2.0,
                "apfu_Fe2": 1.8,
                "apfu_Mg": 0.7,
                "apfu_Ca": 0.4,
                "wt_SiO2": 37.0,
                "wt_Al2O3": 21.0,
                "wt_FeO": 30.0,
                "wt_MgO": 6.0,
                "wt_CaO": 5.0,
            }
        ]
    )

    fe_out = estimate_fe3_by_charge_balance(frame)
    assert "apfu_Fe3_recalc" in fe_out.columns
    assert fe_out.loc[0, "apfu_Fe3_recalc"] >= 0.0

    oxide_out = renormalize_oxides(frame, target_total=100.0, include_hydrous=False)
    assert np.isclose(float(oxide_out.loc[0, "norm_total"]), 100.0, atol=1e-6)

    apfu_out = normalize_apfu(frame, target_cations=8.0)
    assert np.isclose(float(apfu_out.loc[0, "norm_cation_sum"]), 8.0, atol=1e-6)


def test_endmembers_and_thermobarometry_outputs() -> None:
    frame = pd.DataFrame(
        [
            {
                "sample_id": "PT-1",
                "apfu_Ca": 0.5,
                "apfu_Mg": 0.8,
                "apfu_Fe2": 0.7,
                "apfu_Na": 0.2,
                "apfu_K": 0.1,
                "apfu_Mn": 0.05,
                "apfu_Ti": 0.2,
                "apfu_Al": 1.6,
                "zr_ppm": 120,
                "ti_ppm": 15,
            }
        ]
    )

    endm = project_endmembers(frame)
    assert np.isclose(float(endm.loc[0, ["pyx_wo", "pyx_en", "pyx_fs"]].sum()), 100.0, atol=1e-6)
    assert np.isclose(float(endm.loc[0, ["fsp_an", "fsp_ab", "fsp_or"]].sum()), 100.0, atol=1e-6)

    thermo = compute_thermobarometry(frame)
    assert "al_in_hornblende_pressure_kbar" in thermo.columns
    assert np.isfinite(float(thermo.loc[0, "al_in_hornblende_pressure_kbar"]))
