from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OxideSpec:
    oxide: str
    cation: str
    molar_mass: float
    cations_per_oxide: int
    oxygens_per_oxide: int
    charge: int


@dataclass(frozen=True)
class MineralSpec:
    name: str
    oxygen_basis: float
    expected_si: float
    si_tolerance: float
    expected_cations: float
    cation_tolerance: float
    oxide_total_min: float
    oxide_total_max: float


@dataclass(frozen=True)
class FormulaComputation:
    oxygen_basis: float
    apfu: dict[str, float]
    oxygen_sum: float
    cation_sum: float
    charge_sum: float
    charge_imbalance: float


OXIDE_SPECS: dict[str, OxideSpec] = {
    "SiO2": OxideSpec("SiO2", "Si", 60.0843, 1, 2, 4),
    "TiO2": OxideSpec("TiO2", "Ti", 79.866, 1, 2, 4),
    "Al2O3": OxideSpec("Al2O3", "Al", 101.9613, 2, 3, 3),
    "Cr2O3": OxideSpec("Cr2O3", "Cr", 151.9904, 2, 3, 3),
    "FeO": OxideSpec("FeO", "Fe2", 71.844, 1, 1, 2),
    "Fe2O3": OxideSpec("Fe2O3", "Fe3", 159.6882, 2, 3, 3),
    "MnO": OxideSpec("MnO", "Mn", 70.9374, 1, 1, 2),
    "MgO": OxideSpec("MgO", "Mg", 40.3044, 1, 1, 2),
    "CaO": OxideSpec("CaO", "Ca", 56.0774, 1, 1, 2),
    "Na2O": OxideSpec("Na2O", "Na", 61.9789, 2, 1, 1),
    "K2O": OxideSpec("K2O", "K", 94.196, 2, 1, 1),
    "NiO": OxideSpec("NiO", "Ni", 74.6928, 1, 1, 2),
    "ZnO": OxideSpec("ZnO", "Zn", 81.379, 1, 1, 2),
    "P2O5": OxideSpec("P2O5", "P", 141.9445, 2, 5, 5),
}

CATION_ORDER = ["Si", "Ti", "Al", "Cr", "Fe2", "Fe3", "Mn", "Mg", "Ca", "Na", "K", "Ni", "Zn", "P"]
OXIDE_ORDER = list(OXIDE_SPECS.keys())

MINERAL_SPECS: dict[str, MineralSpec] = {
    "garnet": MineralSpec("garnet", 12.0, 3.0, 0.35, 8.0, 0.6, 94.0, 102.5),
    "pyroxene": MineralSpec("pyroxene", 6.0, 2.0, 0.35, 4.0, 0.45, 94.0, 102.5),
    "amphibole": MineralSpec("amphibole", 23.0, 7.5, 1.2, 15.0, 2.0, 90.0, 103.5),
}

_SUFFIXES = ("WTPERCENT", "WTPCT", "PERCENT", "PCT", "WT", "OXIDE", "OX")

_RAW_ALIASES = {
    "FEOT": "FeO",
    "FEO*": "FeO",
    "FETOT": "FeO",
    "FETOTAL": "FeO",
    "FEO(T)": "FeO",
    "FEO_TOTAL": "FeO",
    "SIO2": "SiO2",
    "TIO2": "TiO2",
    "AL2O3": "Al2O3",
    "CR2O3": "Cr2O3",
    "FEO": "FeO",
    "FE2O3": "Fe2O3",
    "MNO": "MnO",
    "MGO": "MgO",
    "CAO": "CaO",
    "NA2O": "Na2O",
    "K2O": "K2O",
    "NIO": "NiO",
    "ZNO": "ZnO",
    "P2O5": "P2O5",
}


def _clean_token(token: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", token).upper()


ALIASES = {_clean_token(key): value for key, value in _RAW_ALIASES.items()}
for oxide in OXIDE_ORDER:
    ALIASES[_clean_token(oxide)] = oxide


def match_oxide_column(name: str) -> str | None:
    token = _clean_token(name)
    if not token:
        return None
    if token in ALIASES:
        return ALIASES[token]

    reduced = token
    changed = True
    while changed:
        changed = False
        for suffix in _SUFFIXES:
            if reduced.endswith(suffix) and len(reduced) > len(suffix):
                reduced = reduced[: -len(suffix)]
                changed = True
    return ALIASES.get(reduced)


def build_oxide_column_map(columns: Iterable[str]) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    for column in columns:
        oxide = match_oxide_column(str(column))
        if oxide is None:
            continue
        mapping.setdefault(oxide, []).append(str(column))
    return mapping


def _first_numeric_value(row: pd.Series, columns: list[str]) -> float:
    for column in columns:
        value = pd.to_numeric(row.get(column), errors="coerce")
        if pd.notna(value):
            numeric = float(value)
            return numeric if numeric > 0 else 0.0
    return 0.0


def extract_oxide_wt(row: pd.Series, oxide_columns: dict[str, list[str]]) -> dict[str, float]:
    data: dict[str, float] = {oxide: 0.0 for oxide in OXIDE_ORDER}
    for oxide, columns in oxide_columns.items():
        data[oxide] = _first_numeric_value(row, columns)
    return data


def calculate_structural_formula(oxide_wt: dict[str, float], spec: MineralSpec) -> FormulaComputation:
    cation_moles: dict[str, float] = {}
    oxygen_sum = 0.0

    for oxide, wt in oxide_wt.items():
        if wt <= 0:
            continue
        params = OXIDE_SPECS.get(oxide)
        if params is None:
            continue
        moles_oxide = wt / params.molar_mass
        cation_moles[params.cation] = cation_moles.get(params.cation, 0.0) + moles_oxide * params.cations_per_oxide
        oxygen_sum += moles_oxide * params.oxygens_per_oxide

    if oxygen_sum <= 0:
        return FormulaComputation(
            oxygen_basis=spec.oxygen_basis,
            apfu={},
            oxygen_sum=0.0,
            cation_sum=0.0,
            charge_sum=0.0,
            charge_imbalance=float("nan"),
        )

    factor = spec.oxygen_basis / oxygen_sum
    apfu = {cation: moles * factor for cation, moles in cation_moles.items()}
    cation_sum = float(sum(apfu.values()))

    charge_sum = 0.0
    for cation, value in apfu.items():
        charge = 0
        for oxide_spec in OXIDE_SPECS.values():
            if oxide_spec.cation == cation:
                charge = oxide_spec.charge
                break
        charge_sum += value * charge

    expected_charge = 2.0 * spec.oxygen_basis
    charge_imbalance = charge_sum - expected_charge

    return FormulaComputation(
        oxygen_basis=spec.oxygen_basis,
        apfu=apfu,
        oxygen_sum=oxygen_sum,
        cation_sum=cation_sum,
        charge_sum=charge_sum,
        charge_imbalance=charge_imbalance,
    )


def detect_mineral(oxide_wt: dict[str, float]) -> tuple[str, float, dict[str, float]]:
    oxide_total = float(sum(oxide_wt.values()))
    scores: dict[str, float] = {}

    for mineral, spec in MINERAL_SPECS.items():
        calc = calculate_structural_formula(oxide_wt, spec)
        si = calc.apfu.get("Si", 0.0)

        score = 0.0
        score += abs(si - spec.expected_si) / max(spec.si_tolerance, 0.01)
        score += abs(calc.cation_sum - spec.expected_cations) / max(spec.cation_tolerance, 0.01)
        if math.isfinite(calc.charge_imbalance):
            score += abs(calc.charge_imbalance) / 0.5
        else:
            score += 10.0

        if oxide_total < spec.oxide_total_min or oxide_total > spec.oxide_total_max:
            score += 1.0
        scores[mineral] = score

    ranked = sorted(scores.items(), key=lambda item: item[1])
    best_name, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else ranked[0][1] + 1.0
    margin = max(0.0, second_score - best_score)
    confidence = 1.0 - math.exp(-margin)
    return best_name, confidence, scores


def _evaluate_flags(
    oxide_total: float,
    calc: FormulaComputation,
    spec: MineralSpec,
    has_fe2o3: bool,
    has_feo: bool,
) -> list[str]:
    flags: list[str] = []

    if oxide_total < spec.oxide_total_min or oxide_total > spec.oxide_total_max:
        flags.append("oxide_total_out_of_range")

    if not math.isfinite(calc.charge_imbalance) or abs(calc.charge_imbalance) > 0.35:
        flags.append("charge_imbalance")

    si = calc.apfu.get("Si", 0.0)
    si_out = not (spec.expected_si - spec.si_tolerance <= si <= spec.expected_si + spec.si_tolerance)
    csum_out = not (
        spec.expected_cations - spec.cation_tolerance
        <= calc.cation_sum
        <= spec.expected_cations + spec.cation_tolerance
    )
    if si_out or csum_out:
        flags.append("impossible_stoichiometry")

    if spec.name == "amphibole" and has_feo and not has_fe2o3:
        flags.append("fe3_estimation_needed")

    return flags


def _robust_z(values: pd.Series) -> np.ndarray:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    median = float(np.nanmedian(arr))
    mad = float(np.nanmedian(np.abs(arr - median)))
    if mad == 0.0 or math.isnan(mad):
        return np.zeros(arr.shape, dtype=float)
    return 0.6745 * (arr - median) / mad


def _append_flag(flag_string: str, new_flag: str) -> str:
    if not flag_string:
        return new_flag
    current = set(part.strip() for part in flag_string.split(";") if part.strip())
    current.add(new_flag)
    return ";".join(sorted(current))


def process_dataframe(
    frame: pd.DataFrame,
    mineral_mode: str = "auto",
    manual_mineral: str = "garnet",
    sample_id_column: str | None = None,
) -> pd.DataFrame:
    oxide_columns = build_oxide_column_map(frame.columns)
    if not oxide_columns:
        raise ValueError("No recognizable oxide columns found.")

    rows: list[dict[str, object]] = []
    for idx, row in frame.iterrows():
        oxide_wt = extract_oxide_wt(row, oxide_columns)
        oxide_total = float(sum(oxide_wt.values()))

        if mineral_mode == "manual":
            mineral = manual_mineral
            confidence = 1.0
            scores = {name: math.nan for name in MINERAL_SPECS}
        else:
            mineral, confidence, scores = detect_mineral(oxide_wt)

        spec = MINERAL_SPECS[mineral]
        calc = calculate_structural_formula(oxide_wt, spec)
        flags = _evaluate_flags(
            oxide_total=oxide_total,
            calc=calc,
            spec=spec,
            has_fe2o3=oxide_wt.get("Fe2O3", 0.0) > 0.0,
            has_feo=oxide_wt.get("FeO", 0.0) > 0.0,
        )

        sample_id: object
        if sample_id_column and sample_id_column in frame.columns:
            sample_id = row.get(sample_id_column)
        else:
            sample_id = idx

        result: dict[str, object] = {
            "sample_id": sample_id,
            "mineral": mineral,
            "detection_confidence": round(confidence, 4),
            "oxide_total": round(oxide_total, 4),
            "oxygen_basis": spec.oxygen_basis,
            "cation_sum": round(calc.cation_sum, 4),
            "charge_sum": round(calc.charge_sum, 4),
            "charge_imbalance": round(calc.charge_imbalance, 4) if math.isfinite(calc.charge_imbalance) else math.nan,
            "qc_flags": ";".join(flags),
            "qc_pass": len(flags) == 0,
        }

        for mineral_name in MINERAL_SPECS:
            result[f"detect_score_{mineral_name}"] = round(scores.get(mineral_name, math.nan), 4)

        for oxide in OXIDE_ORDER:
            result[f"wt_{oxide}"] = round(oxide_wt.get(oxide, 0.0), 4)

        for cation in CATION_ORDER:
            result[f"apfu_{cation}"] = round(calc.apfu.get(cation, 0.0), 4)

        rows.append(result)

    output = pd.DataFrame(rows)
    z_total = _robust_z(output["oxide_total"])
    z_charge = _robust_z(output["charge_imbalance"])
    outlier = (np.abs(z_total) > 3.5) | (np.abs(z_charge) > 3.5)
    output["qc_outlier"] = outlier

    if outlier.any():
        output.loc[outlier, "qc_flags"] = output.loc[outlier, "qc_flags"].map(
            lambda value: _append_flag(str(value), "batch_outlier")
        )
        output.loc[outlier, "qc_pass"] = False

    return output


def build_publication_table(processed: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "sample_id",
        "mineral",
        "oxide_total",
        "cation_sum",
        "charge_imbalance",
        "qc_pass",
        "qc_flags",
    ] + [f"apfu_{cation}" for cation in CATION_ORDER]
    keep = [column for column in preferred if column in processed.columns]
    return processed[keep].copy()


def build_georoc_table(processed: pd.DataFrame) -> pd.DataFrame:
    columns = ["sample_id", "mineral"] + [f"wt_{oxide}" for oxide in OXIDE_ORDER]
    available = [column for column in columns if column in processed.columns]
    georoc = processed[available].copy()
    rename_map = {f"wt_{oxide}": oxide for oxide in OXIDE_ORDER}
    georoc = georoc.rename(columns=rename_map)
    oxide_columns = [oxide for oxide in OXIDE_ORDER if oxide in georoc.columns]
    georoc["TOTAL"] = georoc[oxide_columns].sum(axis=1)
    return georoc


def build_ternary_table(processed: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for _, row in processed.iterrows():
        mineral = str(row.get("mineral", ""))
        sample_id = row.get("sample_id")

        if mineral == "pyroxene":
            wo = float(row.get("apfu_Ca", 0.0))
            en = float(row.get("apfu_Mg", 0.0))
            fs = float(row.get("apfu_Fe2", 0.0))
            total = wo + en + fs
            if total > 0:
                rows.append(
                    {
                        "sample_id": sample_id,
                        "ternary_type": "pyroxene_wo_en_fs",
                        "wo": wo / total,
                        "en": en / total,
                        "fs": fs / total,
                    }
                )
        elif mineral == "garnet":
            prp = float(row.get("apfu_Mg", 0.0))
            alm = float(row.get("apfu_Fe2", 0.0))
            grs = float(row.get("apfu_Ca", 0.0))
            sps = float(row.get("apfu_Mn", 0.0))
            ternary_total = prp + alm + grs
            quadrilateral_total = prp + alm + grs + sps
            if ternary_total > 0:
                rows.append(
                    {
                        "sample_id": sample_id,
                        "ternary_type": "garnet_prp_alm_grs",
                        "prp": prp / ternary_total,
                        "alm": alm / ternary_total,
                        "grs": grs / ternary_total,
                        "sps_fraction": sps / quadrilateral_total if quadrilateral_total > 0 else 0.0,
                    }
                )

    return pd.DataFrame(rows)


def summarize_quality(processed: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        processed.groupby("mineral", dropna=False)
        .agg(
            analyses=("sample_id", "count"),
            qc_pass_rate=("qc_pass", "mean"),
            mean_charge_imbalance=("charge_imbalance", "mean"),
            outliers=("qc_outlier", "sum"),
        )
        .reset_index()
    )
    grouped["qc_pass_rate"] = grouped["qc_pass_rate"].round(4)
    grouped["mean_charge_imbalance"] = grouped["mean_charge_imbalance"].round(4)
    return grouped
