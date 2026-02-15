from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd


APFU_CHARGES = {
    "apfu_Si": 4,
    "apfu_Ti": 4,
    "apfu_Al": 3,
    "apfu_Cr": 3,
    "apfu_Fe2": 2,
    "apfu_Fe3": 3,
    "apfu_Mn": 2,
    "apfu_Mg": 2,
    "apfu_Ca": 2,
    "apfu_Na": 1,
    "apfu_K": 1,
    "apfu_Ni": 2,
    "apfu_Zn": 2,
    "apfu_P": 5,
}


def _lookup_columns(frame: pd.DataFrame) -> dict[str, str]:
    return {str(column).strip().lower(): str(column) for column in frame.columns}


def _resolve_column(frame: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    lookup = _lookup_columns(frame)
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
        found = lookup.get(str(candidate).lower())
        if found is not None:
            return found
    return None


def _numeric_series(frame: pd.DataFrame, candidates: list[str], default: float = 0.0) -> pd.Series:
    column = _resolve_column(frame, candidates)
    if column is None:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default).astype(float)


def _sample_ids(frame: pd.DataFrame) -> pd.Series:
    sample_column = _resolve_column(frame, ["sample_id", "sample", "id", "analysis", "name"])
    if sample_column is None:
        return pd.Series(frame.index, index=frame.index).map(str)
    return frame[sample_column].astype(str)


def _phase(frame: pd.DataFrame) -> pd.Series:
    phase_column = _resolve_column(frame, ["mineral", "phase", "mineral_type"])
    if phase_column is None:
        return pd.Series("unknown", index=frame.index, dtype=object)
    return frame[phase_column].astype(str).str.lower()


def _normalize_components(data: pd.DataFrame) -> pd.DataFrame:
    comps = data.fillna(0.0).astype(float).clip(lower=0.0)
    total = comps.sum(axis=1)
    out = comps.copy()
    valid = total > 0
    out.loc[valid] = comps.loc[valid].div(total.loc[valid], axis=0)
    out.loc[~valid] = 0.0
    return out


def _sign_changes(values: np.ndarray, tolerance: float) -> int:
    if values.size < 3:
        return 0
    diffs = np.diff(values)
    signs = np.where(diffs > tolerance, 1, np.where(diffs < -tolerance, -1, 0))
    nonzero = signs[signs != 0]
    if nonzero.size < 2:
        return 0
    return int(np.sum(nonzero[1:] != nonzero[:-1]))


def classify_nomenclature(formula_frame: pd.DataFrame) -> pd.DataFrame:
    frame = formula_frame.copy()
    phase = _phase(frame)

    mg = _numeric_series(frame, ["apfu_Mg", "mg"]).clip(lower=0)
    fe2 = _numeric_series(frame, ["apfu_Fe2", "apfu_Fe", "fe2"]).clip(lower=0)
    fe3 = _numeric_series(frame, ["apfu_Fe3", "fe3"]).clip(lower=0)
    ca = _numeric_series(frame, ["apfu_Ca", "ca"]).clip(lower=0)
    na = _numeric_series(frame, ["apfu_Na", "na"]).clip(lower=0)
    k = _numeric_series(frame, ["apfu_K", "k"]).clip(lower=0)
    si = _numeric_series(frame, ["apfu_Si", "si"]).clip(lower=0)
    al = _numeric_series(frame, ["apfu_Al", "al"]).clip(lower=0)
    cr = _numeric_series(frame, ["apfu_Cr", "cr"]).clip(lower=0)
    mn = _numeric_series(frame, ["apfu_Mn", "mn"]).clip(lower=0)

    out = pd.DataFrame({"sample_id": _sample_ids(frame), "phase_input": phase})

    pyrox_components = _normalize_components(pd.DataFrame({"wo": ca, "en": mg, "fs": fe2}))
    wo = pyrox_components["wo"] * 100.0
    en = pyrox_components["en"] * 100.0
    fs = pyrox_components["fs"] * 100.0
    pyrox_name = np.select(
        [
            (wo < 5.0) & (fs < 50.0),
            (wo < 5.0) & (fs >= 50.0),
            (wo >= 5.0) & (wo < 20.0),
            (wo >= 20.0) & (fs < 50.0),
            (wo >= 20.0) & (fs >= 50.0),
        ],
        [
            "Enstatite",
            "Ferrosilite",
            "Pigeonite",
            "Diopside/Augite",
            "Hedenbergite/Augite",
        ],
        default="Unclassified pyroxene",
    )
    out["pyroxene_name"] = np.where(phase.str.contains("pyrox|cpx|opx"), pyrox_name, "")

    mg_number = np.where((mg + fe2) > 0.0, mg / (mg + fe2), np.nan)
    ca_b = ca.clip(upper=2.0)
    na_b = na.clip(upper=2.0)
    amphi_group = np.select(
        [
            (ca_b >= 1.5) & (na_b < 0.5),
            (na_b >= 1.5) & (ca_b < 0.5),
            (ca_b >= 0.5) & (na_b >= 0.5),
        ],
        ["Calcic", "Sodic", "Sodic-Calcic"],
        default="Unclassified",
    )
    amphi_name = np.select(
        [
            (amphi_group == "Calcic") & (si >= 6.5) & (mg_number >= 0.5),
            (amphi_group == "Calcic") & (si >= 6.5) & (mg_number < 0.5),
            (amphi_group == "Calcic") & (si < 6.5) & (mg_number >= 0.5),
            (amphi_group == "Calcic") & (si < 6.5) & (mg_number < 0.5),
            (amphi_group == "Sodic") & (mg_number >= 0.5),
            (amphi_group == "Sodic") & (mg_number < 0.5),
            (amphi_group == "Sodic-Calcic") & (mg_number >= 0.5),
            (amphi_group == "Sodic-Calcic") & (mg_number < 0.5),
        ],
        [
            "Magnesiohornblende",
            "Ferrohornblende",
            "Tschermakite",
            "Ferrotschermakite",
            "Glaucophane",
            "Riebeckite",
            "Winchite",
            "Richterite",
        ],
        default="Unclassified amphibole",
    )
    out["amphibole_name"] = np.where(phase.str.contains("amph|hornblende|actin|riebeck|glauc"), amphi_name, "")
    out["amphibole_group"] = np.where(
        phase.str.contains("amph|hornblende|actin|riebeck|glauc"), amphi_group, ""
    )

    tour_x_site = np.select(
        [ca >= 0.5, na >= 0.5],
        ["Calcic", "Alkali"],
        default="X-vacant",
    )
    tourmaline_name = np.select(
        [
            (tour_x_site == "Alkali") & (mg_number >= 0.65),
            (tour_x_site == "Alkali") & (mg_number <= 0.35),
            (tour_x_site == "X-vacant") & (mg_number <= 0.35),
            (tour_x_site == "Calcic") & (mg_number >= 0.5),
            (tour_x_site == "Calcic") & (mg_number < 0.5),
        ],
        [
            "Dravite",
            "Schorl",
            "Foitite",
            "Uvite",
            "Feruvite",
        ],
        default="Unclassified tourmaline",
    )
    out["tourmaline_name"] = np.where(phase.str.contains("tourm"), tourmaline_name, "")
    out["tourmaline_subgroup"] = np.where(phase.str.contains("tourm"), tour_x_site, "")

    garnet_components = _normalize_components(pd.DataFrame({"prp": mg, "alm": fe2, "grs": ca, "sps": mn}))
    prp = garnet_components["prp"] * 100.0
    alm = garnet_components["alm"] * 100.0
    grs = garnet_components["grs"] * 100.0
    sps = garnet_components["sps"] * 100.0

    andradite_proxy = _normalize_components(pd.DataFrame({"and": ca * fe3, "sum": pd.Series(1.0, index=frame.index)}))["and"]
    dominant = pd.DataFrame({"Pyrope": prp, "Almandine": alm, "Grossular": grs, "Spessartine": sps}).idxmax(axis=1)
    variety = np.where(pd.DataFrame({"prp": prp, "alm": alm, "grs": grs, "sps": sps}).max(axis=1) >= 60.0, dominant, "Mixed Garnet")
    variety = np.where(andradite_proxy > 0.25, "Andradite-rich Garnet", variety)
    out["garnet_variety"] = np.where(phase.str.contains("garnet|grt"), variety, "")

    cr_number = np.where((cr + al) > 0.0, cr / (cr + al), np.nan)
    spinel_name = np.select(
        [
            (cr_number >= 0.6) & (mg_number >= 0.5),
            (cr_number >= 0.6) & (mg_number < 0.5),
            (cr_number < 0.6) & (mg_number >= 0.5),
            (cr_number < 0.6) & (mg_number < 0.5),
        ],
        ["Magnesiochromite", "Chromite", "Spinel", "Hercynite"],
        default="Unclassified spinel",
    )
    out["spinel_group_name"] = np.where(phase.str.contains("spinel|chr"), spinel_name, "")

    out["classification_note"] = (
        "Proxy rules aligned to requested nomenclature families; validate with full site-allocation workflows for publication."
    )
    return out


def analyze_zoning_profile(
    traverse_frame: pd.DataFrame,
    distance_col: str,
    element_cols: list[str],
    normal_direction: str = "decreasing",
    sector_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if distance_col not in traverse_frame.columns:
        raise ValueError(f"Distance column not found: {distance_col}")
    if not element_cols:
        raise ValueError("Select at least one element column for zoning analysis.")

    frame = traverse_frame.copy()
    frame[distance_col] = pd.to_numeric(frame[distance_col], errors="coerce")
    frame = frame.dropna(subset=[distance_col]).sort_values(distance_col).reset_index(drop=True)
    if frame.empty:
        raise ValueError("No valid traverse rows after parsing distance column.")

    profiles = pd.DataFrame({"distance": frame[distance_col]})
    rows: list[dict[str, object]] = []

    sector_variance_weight = 0.0
    has_sector = sector_col is not None and sector_col in frame.columns

    for element in element_cols:
        if element not in frame.columns:
            continue
        values = pd.to_numeric(frame[element], errors="coerce")
        valid = values.notna()
        x = frame.loc[valid, distance_col].to_numpy(dtype=float)
        y = values.loc[valid].to_numpy(dtype=float)
        if y.size < 3:
            continue

        slope = float(np.polyfit(x, y, 1)[0])
        amplitude = float(np.nanmax(y) - np.nanmin(y))
        tolerance = max(1e-9, 0.02 * amplitude)
        sign_changes = _sign_changes(y, tolerance=tolerance)

        if has_sector:
            grouped = frame.loc[valid].groupby(sector_col)[element].mean(numeric_only=True)
            sector_variance = float(grouped.var()) if len(grouped) > 1 else 0.0
            line_variance = float(np.var(y - np.polyval(np.polyfit(x, y, 1), x)))
            sector_variance_weight = sector_variance / max(line_variance, 1e-12)
        else:
            sector_variance_weight = 0.0

        if has_sector and sector_variance_weight > 1.5:
            zoning = "sector zoning"
        elif sign_changes >= 3 and amplitude > tolerance:
            zoning = "oscillatory"
        else:
            if normal_direction == "increasing":
                zoning = "normal" if slope > 0 else "reverse"
            else:
                zoning = "normal" if slope < 0 else "reverse"

        rows.append(
            {
                "element": element,
                "slope": slope,
                "amplitude": amplitude,
                "sign_changes": sign_changes,
                "sector_index": sector_variance_weight,
                "zoning_type": zoning,
            }
        )
        profiles[element] = pd.to_numeric(frame[element], errors="coerce")

    if not rows:
        raise ValueError("No analyzable element columns were found in the traverse data.")

    summary = pd.DataFrame(rows)
    return summary, profiles


def estimate_diffusion_and_closure(
    traverse_frame: pd.DataFrame,
    distance_col: str,
    element_col: str,
    duration_myr: float,
    d0_m2_s: float,
    activation_energy_kj_mol: float,
) -> dict[str, float]:
    if duration_myr <= 0:
        raise ValueError("Duration must be greater than zero.")

    frame = traverse_frame.copy()
    distance = pd.to_numeric(frame[distance_col], errors="coerce")
    values = pd.to_numeric(frame[element_col], errors="coerce")
    valid = distance.notna() & values.notna()

    x = distance.loc[valid].to_numpy(dtype=float)
    y = values.loc[valid].to_numpy(dtype=float)
    if y.size < 3:
        raise ValueError("Need at least three valid points to estimate diffusion profile.")

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    y_min = float(np.min(y))
    y_max = float(np.max(y))
    if math.isclose(y_min, y_max):
        raise ValueError("Element profile is flat; diffusion length cannot be estimated.")

    y_norm = (y - y_min) / (y_max - y_min)
    x10 = float(np.interp(0.1, y_norm, x))
    x90 = float(np.interp(0.9, y_norm, x))
    diffusion_length_m = abs(x90 - x10) * 1e-6

    duration_s = duration_myr * 1e6 * 365.25 * 24.0 * 3600.0
    d_eff = (diffusion_length_m**2) / max(4.0 * duration_s, 1e-30)

    closure_temperature_c = float("nan")
    if d_eff > 0 and d0_m2_s > d_eff > 0 and activation_energy_kj_mol > 0:
        r = 8.314
        ea_j_mol = activation_energy_kj_mol * 1000.0
        closure_temperature_k = ea_j_mol / (r * math.log(d0_m2_s / d_eff))
        closure_temperature_c = closure_temperature_k - 273.15

    return {
        "diffusion_length_um": diffusion_length_m * 1e6,
        "effective_diffusivity_m2_s": d_eff,
        "closure_temperature_c": closure_temperature_c,
    }


def estimate_fe3_by_charge_balance(formula_frame: pd.DataFrame) -> pd.DataFrame:
    frame = formula_frame.copy()
    fe2_col = _resolve_column(frame, ["apfu_Fe2", "apfu_Fe", "fe2"])
    if fe2_col is None:
        raise ValueError("Need APFU Fe2 (or APFU Fe) column for Fe3+ estimation.")

    fe3_col = _resolve_column(frame, ["apfu_Fe3", "fe3"])
    oxygen_col = _resolve_column(frame, ["oxygen_basis", "oxygen", "o_basis"])

    fe2 = pd.to_numeric(frame[fe2_col], errors="coerce").fillna(0.0).astype(float).clip(lower=0.0)
    fe3 = (
        pd.to_numeric(frame[fe3_col], errors="coerce").fillna(0.0).astype(float).clip(lower=0.0)
        if fe3_col is not None
        else pd.Series(0.0, index=frame.index, dtype=float)
    )
    oxygen_basis = (
        pd.to_numeric(frame[oxygen_col], errors="coerce").fillna(12.0).astype(float)
        if oxygen_col is not None
        else pd.Series(12.0, index=frame.index, dtype=float)
    )

    charge_sum = pd.Series(0.0, index=frame.index, dtype=float)
    for column, charge in APFU_CHARGES.items():
        if column in frame.columns:
            charge_sum += pd.to_numeric(frame[column], errors="coerce").fillna(0.0).astype(float) * charge

    expected_charge = 2.0 * oxygen_basis
    deficit = expected_charge - charge_sum

    fe_total = fe2 + fe3
    fe3_new = (fe3 + deficit).clip(lower=0.0)
    fe3_new = np.minimum(fe3_new, fe_total)
    fe2_new = fe_total - fe3_new

    out = frame.copy()
    out["apfu_Fe2_recalc"] = fe2_new.round(6)
    out["apfu_Fe3_recalc"] = pd.Series(fe3_new, index=frame.index).round(6)
    out["fe3_fraction_recalc"] = (pd.Series(fe3_new, index=frame.index) / fe_total.replace(0, np.nan)).fillna(0.0).round(6)
    return out


def renormalize_oxides(
    table: pd.DataFrame,
    target_total: float = 100.0,
    include_hydrous: bool = False,
    water_column_name: str = "wt_H2O",
) -> pd.DataFrame:
    frame = table.copy()
    oxide_cols = [column for column in frame.columns if str(column).startswith("wt_")]
    if not oxide_cols:
        raise ValueError("No wt_ oxide columns found for renormalization.")

    selected = oxide_cols.copy()
    if not include_hydrous and water_column_name in selected:
        selected.remove(water_column_name)

    total = frame[selected].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1)
    factor = target_total / total.replace(0.0, np.nan)

    out = frame.copy()
    for col in selected:
        out[f"norm_{col}"] = (
            pd.to_numeric(frame[col], errors="coerce").fillna(0.0).astype(float) * factor
        ).fillna(0.0)
    out["norm_total"] = out[[f"norm_{col}" for col in selected]].sum(axis=1)
    out["renorm_mode"] = "hydrous" if include_hydrous else "anhydrous"
    return out


def normalize_apfu(
    table: pd.DataFrame,
    target_cations: float,
) -> pd.DataFrame:
    frame = table.copy()
    apfu_cols = [column for column in frame.columns if str(column).startswith("apfu_")]
    if not apfu_cols:
        raise ValueError("No APFU columns found for cation normalization.")

    apfu = frame[apfu_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    csum = apfu.sum(axis=1)
    factor = target_cations / csum.replace(0.0, np.nan)

    out = frame.copy()
    for col in apfu_cols:
        out[f"norm_{col}"] = (apfu[col] * factor).fillna(0.0)
    out["norm_cation_sum"] = out[[f"norm_{col}" for col in apfu_cols]].sum(axis=1)
    return out


def project_endmembers(table: pd.DataFrame) -> pd.DataFrame:
    frame = table.copy()
    ca = _numeric_series(frame, ["apfu_Ca", "ca"])
    mg = _numeric_series(frame, ["apfu_Mg", "mg"])
    fe2 = _numeric_series(frame, ["apfu_Fe2", "apfu_Fe", "fe2"])
    mn = _numeric_series(frame, ["apfu_Mn", "mn"])
    na = _numeric_series(frame, ["apfu_Na", "na"])
    k = _numeric_series(frame, ["apfu_K", "k"])

    pyx = _normalize_components(pd.DataFrame({"wo": ca, "en": mg, "fs": fe2}))
    fsp = _normalize_components(pd.DataFrame({"an": ca, "ab": na, "or": k}))
    grt = _normalize_components(pd.DataFrame({"prp": mg, "alm": fe2, "grs": ca, "sps": mn}))

    out = pd.DataFrame({"sample_id": _sample_ids(frame)})
    out["pyx_wo"] = (pyx["wo"] * 100.0).round(4)
    out["pyx_en"] = (pyx["en"] * 100.0).round(4)
    out["pyx_fs"] = (pyx["fs"] * 100.0).round(4)

    out["fsp_an"] = (fsp["an"] * 100.0).round(4)
    out["fsp_ab"] = (fsp["ab"] * 100.0).round(4)
    out["fsp_or"] = (fsp["or"] * 100.0).round(4)

    out["grt_prp"] = (grt["prp"] * 100.0).round(4)
    out["grt_alm"] = (grt["alm"] * 100.0).round(4)
    out["grt_grs"] = (grt["grs"] * 100.0).round(4)
    out["grt_sps"] = (grt["sps"] * 100.0).round(4)
    return out


def compute_thermobarometry(table: pd.DataFrame) -> pd.DataFrame:
    frame = table.copy()

    ti_apfu = _numeric_series(frame, ["apfu_Ti", "ti_apfu"])
    mg = _numeric_series(frame, ["apfu_Mg", "mg"])
    fe2 = _numeric_series(frame, ["apfu_Fe2", "apfu_Fe", "fe2"])
    al_tot = _numeric_series(frame, ["apfu_Al", "al_tot", "al_total"])
    zr_ppm = _numeric_series(frame, ["zr_ppm", "Zr_ppm"])
    ti_ppm_qtz = _numeric_series(frame, ["ti_ppm", "Ti_ppm", "ti_quartz_ppm"])

    mg_number = np.where((mg + fe2) > 0, mg / (mg + fe2), np.nan)

    # Proxy implementation requiring user calibration checks.
    ti_biotite_t = 480.0 + 180.0 * ti_apfu + 120.0 * np.nan_to_num(mg_number, nan=0.0)

    al_hbl_p = -3.92 + 5.03 * al_tot

    zr_safe = zr_ppm.replace(0, np.nan)
    zr_ttn_t = (7708.0 / (10.52 - np.log10(zr_safe))) - 273.15

    ti_safe = ti_ppm_qtz.replace(0, np.nan)
    titaniq_t = (3765.0 / (5.69 - np.log10(ti_safe))) - 273.15

    out = pd.DataFrame({"sample_id": _sample_ids(frame)})
    out["ti_in_biotite_temp_c_proxy"] = pd.Series(ti_biotite_t, index=frame.index).round(2)
    out["al_in_hornblende_pressure_kbar"] = pd.Series(al_hbl_p, index=frame.index).round(3)
    out["zr_in_titanite_temp_c_proxy"] = pd.Series(zr_ttn_t, index=frame.index).round(2)
    out["titaniq_temp_c"] = pd.Series(titaniq_t, index=frame.index).round(2)
    out["thermobarometry_note"] = (
        "Check calibration assumptions and activities before publication use."
    )
    return out


def fe_mg_equilibrium_filter(
    table: pd.DataFrame,
    phase_a: str,
    phase_b: str,
    sample_id_col: str = "sample_id",
    phase_col: str = "mineral",
    kd_min: float = 0.5,
    kd_max: float = 5.0,
) -> pd.DataFrame:
    if sample_id_col not in table.columns:
        raise ValueError(f"Sample column not found: {sample_id_col}")
    if phase_col not in table.columns:
        raise ValueError(f"Phase column not found: {phase_col}")

    frame = table.copy()
    frame[phase_col] = frame[phase_col].astype(str).str.lower()

    fe = _numeric_series(frame, ["apfu_Fe2", "apfu_Fe", "fe2"]).replace(0, np.nan)
    mg = _numeric_series(frame, ["apfu_Mg", "mg"]).replace(0, np.nan)
    ratio = fe / mg

    reduced = pd.DataFrame(
        {
            "sample_id": frame[sample_id_col].astype(str),
            "phase": frame[phase_col],
            "fe_mg_ratio": ratio,
        }
    )
    reduced = reduced[reduced["phase"].isin([phase_a.lower(), phase_b.lower()])]
    pivot = reduced.pivot_table(index="sample_id", columns="phase", values="fe_mg_ratio", aggfunc="mean")

    if phase_a.lower() not in pivot.columns or phase_b.lower() not in pivot.columns:
        raise ValueError("Could not compute Kd: one or both phases are missing in the selected data.")

    kd = pivot[phase_a.lower()] / pivot[phase_b.lower()]
    out = pd.DataFrame(
        {
            "sample_id": pivot.index.astype(str),
            f"fe_mg_{phase_a.lower()}": pivot[phase_a.lower()],
            f"fe_mg_{phase_b.lower()}": pivot[phase_b.lower()],
            "kd_fe_mg": kd,
        }
    ).reset_index(drop=True)
    out["equilibrium_pass"] = out["kd_fe_mg"].between(kd_min, kd_max, inclusive="both")
    return out


def tag_core_rim(
    table: pd.DataFrame,
    distance_col: str,
    core_fraction: float = 0.25,
    rim_fraction: float = 0.25,
) -> pd.DataFrame:
    if distance_col not in table.columns:
        raise ValueError(f"Distance column not found: {distance_col}")
    if core_fraction <= 0 or rim_fraction <= 0 or core_fraction + rim_fraction >= 1.0:
        raise ValueError("core_fraction and rim_fraction must be > 0 and sum to < 1.")

    frame = table.copy()
    dist = pd.to_numeric(frame[distance_col], errors="coerce")
    dmin = float(dist.min())
    dmax = float(dist.max())
    span = dmax - dmin

    core_cut = dmin + core_fraction * span
    rim_cut = dmax - rim_fraction * span

    zones = np.where(
        dist <= core_cut,
        "core",
        np.where(dist >= rim_cut, "rim", "interior"),
    )

    out = frame.copy()
    out["zone"] = zones
    return out
