from __future__ import annotations

import pandas as pd

from zeta import process_dataframe
from zeta.plot import (
    prepare_amphibole_leake_data,
    prepare_feldspar_ternary_data,
    prepare_garnet_composition_data,
    prepare_pyroxene_quad_data,
)
from zeta.power_tools import analyze_zoning_profile
from zeta.samples import sample_formula_data, sample_microprobe_data, sample_traverse_data


def test_sample_microprobe_runs_end_to_end() -> None:
    microprobe = sample_microprobe_data()
    assert not microprobe.empty
    processed = process_dataframe(microprobe, mineral_mode="auto", sample_id_column="sample_id")

    assert len(processed) == len(microprobe)
    assert processed["mineral"].nunique() >= 2
    assert "qc_flags" in processed.columns


def test_sample_formula_supports_all_zeta_plot_diagrams() -> None:
    formulas = sample_formula_data()

    pyx = prepare_pyroxene_quad_data(formulas)
    amp = prepare_amphibole_leake_data(formulas)
    fsp = prepare_feldspar_ternary_data(formulas)
    grt = prepare_garnet_composition_data(formulas)

    assert len(pyx) > 0
    assert len(amp) > 0
    assert len(fsp) > 0
    assert len(grt) > 0


def test_sample_traverse_supports_zoning_analysis() -> None:
    traverse = sample_traverse_data()
    summary, profile = analyze_zoning_profile(
        traverse_frame=traverse,
        distance_col="distance_um",
        element_cols=["Fe", "Mg", "Mn"],
        normal_direction="decreasing",
        sector_col="sector",
    )

    assert isinstance(summary, pd.DataFrame)
    assert isinstance(profile, pd.DataFrame)
    assert len(summary) >= 1
    assert "distance" in profile.columns
