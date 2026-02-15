from __future__ import annotations

import pandas as pd


def sample_microprobe_data() -> pd.DataFrame:
    rows = [
        # Garnet-like analyses
        {"sample_id": "GRT-01", "SiO2": 37.2, "TiO2": 0.1, "Al2O3": 21.0, "Cr2O3": 0.0, "FeOT": 30.2, "MnO": 1.2, "MgO": 5.6, "CaO": 4.8, "Na2O": 0.0, "K2O": 0.0},
        {"sample_id": "GRT-02", "SiO2": 37.8, "TiO2": 0.1, "Al2O3": 20.4, "Cr2O3": 0.2, "FeOT": 27.9, "MnO": 1.0, "MgO": 7.2, "CaO": 5.1, "Na2O": 0.0, "K2O": 0.0},
        {"sample_id": "GRT-03", "SiO2": 36.9, "TiO2": 0.1, "Al2O3": 21.6, "Cr2O3": 0.0, "FeOT": 31.8, "MnO": 0.9, "MgO": 4.3, "CaO": 4.3, "Na2O": 0.0, "K2O": 0.0},
        {"sample_id": "GRT-04", "SiO2": 37.4, "TiO2": 0.0, "Al2O3": 20.9, "Cr2O3": 0.1, "FeOT": 29.0, "MnO": 1.5, "MgO": 5.8, "CaO": 5.4, "Na2O": 0.0, "K2O": 0.0},
        # Pyroxene-like analyses
        {"sample_id": "CPX-01", "SiO2": 51.5, "TiO2": 0.4, "Al2O3": 2.1, "Cr2O3": 0.0, "FeO": 8.9, "MnO": 0.3, "MgO": 15.9, "CaO": 21.7, "Na2O": 0.5, "K2O": 0.0},
        {"sample_id": "CPX-02", "SiO2": 50.7, "TiO2": 0.6, "Al2O3": 3.2, "Cr2O3": 0.1, "FeO": 10.3, "MnO": 0.2, "MgO": 14.2, "CaO": 20.9, "Na2O": 0.8, "K2O": 0.1},
        {"sample_id": "CPX-03", "SiO2": 52.2, "TiO2": 0.2, "Al2O3": 1.6, "Cr2O3": 0.0, "FeO": 7.8, "MnO": 0.2, "MgO": 16.8, "CaO": 22.4, "Na2O": 0.3, "K2O": 0.0},
        {"sample_id": "CPX-04", "SiO2": 49.8, "TiO2": 0.5, "Al2O3": 2.6, "Cr2O3": 0.1, "FeO": 11.1, "MnO": 0.3, "MgO": 13.3, "CaO": 21.2, "Na2O": 0.7, "K2O": 0.1},
        # Amphibole-like analyses
        {"sample_id": "AMP-01", "SiO2": 45.1, "TiO2": 1.2, "Al2O3": 11.3, "Cr2O3": 0.2, "FeO": 14.6, "MnO": 0.2, "MgO": 12.1, "CaO": 10.5, "Na2O": 2.0, "K2O": 0.9},
        {"sample_id": "AMP-02", "SiO2": 43.9, "TiO2": 1.6, "Al2O3": 12.4, "Cr2O3": 0.1, "FeO": 15.8, "MnO": 0.3, "MgO": 10.7, "CaO": 10.1, "Na2O": 2.4, "K2O": 1.1},
        {"sample_id": "AMP-03", "SiO2": 46.0, "TiO2": 0.8, "Al2O3": 9.7, "Cr2O3": 0.1, "FeO": 13.1, "MnO": 0.2, "MgO": 13.6, "CaO": 10.8, "Na2O": 1.6, "K2O": 0.6},
        {"sample_id": "AMP-04", "SiO2": 44.2, "TiO2": 1.0, "Al2O3": 10.8, "Cr2O3": 0.2, "FeO": 16.4, "MnO": 0.3, "MgO": 11.0, "CaO": 9.7, "Na2O": 2.5, "K2O": 1.2},
    ]
    return pd.DataFrame(rows)


def sample_formula_data() -> pd.DataFrame:
    rows = [
        # Pyroxene
        {"sample_id": "PX-F01", "mineral": "pyroxene", "apfu_Si": 1.99, "apfu_Ti": 0.02, "apfu_Al": 0.08, "apfu_Fe2": 0.28, "apfu_Fe3": 0.00, "apfu_Mg": 0.92, "apfu_Ca": 0.80, "apfu_Na": 0.03, "apfu_K": 0.00, "apfu_Mn": 0.01, "oxygen_basis": 6},
        {"sample_id": "PX-F02", "mineral": "pyroxene", "apfu_Si": 1.97, "apfu_Ti": 0.03, "apfu_Al": 0.10, "apfu_Fe2": 0.40, "apfu_Fe3": 0.02, "apfu_Mg": 0.75, "apfu_Ca": 0.78, "apfu_Na": 0.04, "apfu_K": 0.00, "apfu_Mn": 0.01, "oxygen_basis": 6},
        # Amphibole
        {"sample_id": "AMP-F01", "mineral": "amphibole", "apfu_Si": 7.10, "apfu_Ti": 0.12, "apfu_Al": 1.85, "apfu_Fe2": 1.35, "apfu_Fe3": 0.10, "apfu_Mg": 2.05, "apfu_Ca": 1.75, "apfu_Na": 0.35, "apfu_K": 0.20, "apfu_Mn": 0.03, "oxygen_basis": 23},
        {"sample_id": "AMP-F02", "mineral": "amphibole", "apfu_Si": 7.35, "apfu_Ti": 0.08, "apfu_Al": 1.60, "apfu_Fe2": 1.10, "apfu_Fe3": 0.06, "apfu_Mg": 2.30, "apfu_Ca": 1.80, "apfu_Na": 0.25, "apfu_K": 0.15, "apfu_Mn": 0.02, "oxygen_basis": 23},
        # Feldspar
        {"sample_id": "FSP-F01", "mineral": "feldspar", "apfu_Si": 2.65, "apfu_Al": 1.35, "apfu_Ca": 0.20, "apfu_Na": 0.72, "apfu_K": 0.08, "oxygen_basis": 8},
        {"sample_id": "FSP-F02", "mineral": "feldspar", "apfu_Si": 2.75, "apfu_Al": 1.25, "apfu_Ca": 0.62, "apfu_Na": 0.34, "apfu_K": 0.04, "oxygen_basis": 8},
        # Garnet
        {"sample_id": "GRT-F01", "mineral": "garnet", "apfu_Si": 3.01, "apfu_Al": 1.98, "apfu_Fe2": 1.45, "apfu_Fe3": 0.02, "apfu_Mg": 0.90, "apfu_Ca": 0.50, "apfu_Mn": 0.15, "oxygen_basis": 12},
        {"sample_id": "GRT-F02", "mineral": "garnet", "apfu_Si": 2.99, "apfu_Al": 2.02, "apfu_Fe2": 1.70, "apfu_Fe3": 0.03, "apfu_Mg": 0.55, "apfu_Ca": 0.55, "apfu_Mn": 0.20, "oxygen_basis": 12},
        # Tourmaline
        {"sample_id": "TRM-F01", "mineral": "tourmaline", "apfu_Si": 6.00, "apfu_Al": 6.10, "apfu_Fe2": 1.20, "apfu_Mg": 1.80, "apfu_Ca": 0.12, "apfu_Na": 0.70, "apfu_K": 0.02, "oxygen_basis": 31},
        # Spinel
        {"sample_id": "SPN-F01", "mineral": "spinel", "apfu_Al": 1.60, "apfu_Cr": 0.90, "apfu_Fe2": 0.55, "apfu_Mg": 0.95, "oxygen_basis": 4},
        # Biotite + titanite + quartz style values for thermobarometry proxy columns
        {"sample_id": "BT-TN-QZ-01", "mineral": "biotite", "apfu_Ti": 0.22, "apfu_Mg": 1.55, "apfu_Fe2": 1.25, "apfu_Al": 2.10, "zr_ppm": 130, "ti_ppm": 25, "oxygen_basis": 11},
    ]
    return pd.DataFrame(rows).fillna(0.0)


def sample_traverse_data() -> pd.DataFrame:
    rows = [
        {"distance_um": 0, "Fe": 15.5, "Mg": 10.8, "Mn": 0.42, "sector": "A"},
        {"distance_um": 20, "Fe": 15.1, "Mg": 11.0, "Mn": 0.38, "sector": "A"},
        {"distance_um": 40, "Fe": 15.8, "Mg": 10.7, "Mn": 0.44, "sector": "B"},
        {"distance_um": 60, "Fe": 14.9, "Mg": 11.2, "Mn": 0.36, "sector": "B"},
        {"distance_um": 80, "Fe": 16.0, "Mg": 10.5, "Mn": 0.48, "sector": "A"},
        {"distance_um": 100, "Fe": 14.6, "Mg": 11.4, "Mn": 0.33, "sector": "B"},
    ]
    return pd.DataFrame(rows)
