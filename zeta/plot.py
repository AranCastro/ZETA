from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


DEFAULT_COLORS = {
    "background": "#f8f9fa",
    "primary": "#8b3a62",
    "accent": "#d4af37",
    "text": "#1a1a1a",
    "success": "#059669",
    "warning": "#ea580c",
}


def _column_lookup(frame: pd.DataFrame) -> dict[str, str]:
    return {str(column).strip().lower(): str(column) for column in frame.columns}


def _resolve_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    lookup = _column_lookup(frame)
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
        found = lookup.get(candidate.lower())
        if found is not None:
            return found
    return None


def _series_from_candidates(frame: pd.DataFrame, candidates: list[str]) -> tuple[pd.Series, str | None]:
    column = _resolve_column(frame, candidates)
    if column is None:
        return pd.Series(0.0, index=frame.index, dtype=float), None
    values = pd.to_numeric(frame[column], errors="coerce").fillna(0.0).astype(float)
    return values.clip(lower=0.0), column


def _sample_ids(frame: pd.DataFrame) -> pd.Series:
    sample_col = _resolve_column(frame, ["sample_id", "sample", "id", "analysis", "name"])
    if sample_col is None:
        return pd.Series(frame.index, index=frame.index, dtype=object).map(str)
    return frame[sample_col].astype(str)


def _filter_by_mineral(frame: pd.DataFrame, keywords: list[str]) -> pd.DataFrame:
    mineral_col = _resolve_column(frame, ["mineral", "phase", "mineral_type"])
    if mineral_col is None:
        return frame
    mineral = frame[mineral_col].astype(str).str.lower()
    pattern = "|".join(keywords)
    mask = mineral.str.contains(pattern, na=False, regex=True)
    if mask.any():
        return frame.loc[mask].copy()
    return frame


def _normalize_100(components: dict[str, pd.Series]) -> pd.DataFrame:
    data = pd.DataFrame(components).fillna(0.0).astype(float)
    total = data.sum(axis=1)
    mask = total > 0
    if not mask.any():
        raise ValueError("No non-zero compositional components found for this diagram.")
    scaled = data.loc[mask].div(total.loc[mask], axis=0) * 100.0
    return scaled


def _with_common_columns(frame: pd.DataFrame, normalized: pd.DataFrame) -> pd.DataFrame:
    selected = frame.loc[normalized.index].copy()
    return pd.DataFrame(
        {
            "sample_id": _sample_ids(selected),
            **{column: normalized[column] for column in normalized.columns},
        },
        index=normalized.index,
    ).reset_index(drop=True)


def prepare_pyroxene_quad_data(formula_frame: pd.DataFrame) -> pd.DataFrame:
    frame = _filter_by_mineral(formula_frame, ["pyrox", "cpx", "opx"])

    wo, wo_col = _series_from_candidates(frame, ["wo", "wo_pct", "apfu_Ca", "ca"])
    en, en_col = _series_from_candidates(frame, ["en", "en_pct", "apfu_Mg", "mg"])
    fs, fs_col = _series_from_candidates(frame, ["fs", "fs_pct", "apfu_Fe2", "apfu_Fe", "fe2"])

    if wo_col is None and en_col is None and fs_col is None:
        raise ValueError("Pyroxene diagram needs Wo/En/Fs or APFU Ca/Mg/Fe2 columns.")

    normalized = _normalize_100({"wo_pct": wo, "en_pct": en, "fs_pct": fs})
    result = _with_common_columns(frame, normalized)

    en_fs = result["en_pct"] + result["fs_pct"]
    result["quad_x_fs_in_en_fs"] = np.where(en_fs > 0.0, 100.0 * result["fs_pct"] / en_fs, np.nan)
    result["quad_y_wo"] = result["wo_pct"]

    conditions = [
        result["wo_pct"] < 5.0,
        (result["wo_pct"] >= 5.0) & (result["wo_pct"] < 20.0),
        (result["wo_pct"] >= 20.0) & (result["quad_x_fs_in_en_fs"] < 50.0),
        (result["wo_pct"] >= 20.0) & (result["quad_x_fs_in_en_fs"] >= 50.0),
    ]
    choices = [
        "Orthopyroxene",
        "Pigeonite",
        "Calcic Clinopyroxene",
        "Ferroan Clinopyroxene",
    ]
    result["pyroxene_field"] = np.select(conditions, choices, default="High-Ca Pyroxene")
    return result


def prepare_amphibole_leake_data(formula_frame: pd.DataFrame) -> pd.DataFrame:
    frame = _filter_by_mineral(formula_frame, ["amph", "hornblende", "actinolite", "riebeckite"])

    na, na_col = _series_from_candidates(frame, ["apfu_Na", "na"])
    ca, ca_col = _series_from_candidates(frame, ["apfu_Ca", "ca"])
    si, si_col = _series_from_candidates(frame, ["apfu_Si", "si"])
    k, _ = _series_from_candidates(frame, ["apfu_K", "k"])

    if na_col is None and ca_col is None:
        raise ValueError("Amphibole diagram needs APFU Na and Ca columns.")

    na_b_proxy = na.clip(upper=2.0)
    ca_b_proxy = ca.clip(upper=2.0)

    valid = (na_b_proxy + ca_b_proxy + si + k) > 0
    if not valid.any():
        raise ValueError("No amphibole-like compositions available for Leake classification.")

    filtered = frame.loc[valid].copy()
    result = pd.DataFrame({"sample_id": _sample_ids(filtered).astype(str)})
    result["na_b_proxy"] = na_b_proxy.loc[valid].to_numpy(dtype=float)
    result["ca_b_proxy"] = ca_b_proxy.loc[valid].to_numpy(dtype=float)
    result["si_apfu"] = si.loc[valid].to_numpy(dtype=float)
    result["a_site_na_k_proxy"] = (na + k - na_b_proxy).loc[valid].clip(lower=0.0).to_numpy(dtype=float)

    conditions = [
        (result["ca_b_proxy"] >= 1.5) & (result["na_b_proxy"] < 0.5),
        (result["na_b_proxy"] >= 1.5) & (result["ca_b_proxy"] < 0.5),
        (result["ca_b_proxy"] >= 0.5) & (result["na_b_proxy"] >= 0.5),
    ]
    choices = ["Calcic", "Sodic", "Sodic-Calcic"]
    result["amphibole_group"] = np.select(conditions, choices, default="Low-B/Unclassified")
    return result.reset_index(drop=True)


def prepare_feldspar_ternary_data(formula_frame: pd.DataFrame) -> pd.DataFrame:
    frame = _filter_by_mineral(formula_frame, ["feld", "plag", "sanidine", "orthoclase", "albite", "anorthite"])

    an, an_col = _series_from_candidates(frame, ["an", "an_pct", "apfu_Ca", "ca"])
    ab, ab_col = _series_from_candidates(frame, ["ab", "ab_pct", "apfu_Na", "na"])
    or_, or_col = _series_from_candidates(frame, ["or", "or_pct", "apfu_K", "k"])

    if an_col is None and ab_col is None and or_col is None:
        raise ValueError("Feldspar ternary needs An/Ab/Or or APFU Ca/Na/K columns.")

    normalized = _normalize_100({"an_pct": an, "ab_pct": ab, "or_pct": or_})
    result = _with_common_columns(frame, normalized)

    conditions = [
        result["or_pct"] >= 35.0,
        result["an_pct"] >= 50.0,
        result["an_pct"] >= 10.0,
    ]
    choices = ["Alkali Feldspar", "Calcic Plagioclase", "Intermediate Plagioclase"]
    result["feldspar_field"] = np.select(conditions, choices, default="Albite-rich")
    return result


def prepare_garnet_composition_data(formula_frame: pd.DataFrame) -> pd.DataFrame:
    frame = _filter_by_mineral(formula_frame, ["garnet", "grt"])

    prp, prp_col = _series_from_candidates(frame, ["prp", "prp_pct", "apfu_Mg", "mg"])
    alm, alm_col = _series_from_candidates(frame, ["alm", "alm_pct", "apfu_Fe2", "apfu_Fe", "fe2"])
    grs, grs_col = _series_from_candidates(frame, ["grs", "grs_pct", "apfu_Ca", "ca"])
    sps, sps_col = _series_from_candidates(frame, ["sps", "sps_pct", "apfu_Mn", "mn"])

    if prp_col is None and alm_col is None and grs_col is None and sps_col is None:
        raise ValueError("Garnet plot needs PRP/ALM/GRS/SPS or APFU Mg/Fe2/Ca/Mn columns.")

    normalized = _normalize_100({"prp_pct": prp, "alm_pct": alm, "grs_pct": grs, "sps_pct": sps})
    result = _with_common_columns(frame, normalized)

    dominant_idx = result[["prp_pct", "alm_pct", "grs_pct", "sps_pct"]].idxmax(axis=1)
    label_map = {
        "prp_pct": "Pyrope",
        "alm_pct": "Almandine",
        "grs_pct": "Grossular",
        "sps_pct": "Spessartine",
    }
    result["dominant_endmember"] = dominant_idx.map(label_map)
    return result


def _apply_plot_style(fig: go.Figure, colors: dict[str, str] | None = None) -> go.Figure:
    palette = DEFAULT_COLORS.copy()
    if colors:
        palette.update(colors)

    fig.update_layout(
        paper_bgcolor=palette["background"],
        plot_bgcolor="#ffffff",
        font_color=palette["text"],
        legend_title_text="",
    )
    return fig


def build_pyroxene_quad_figure(data: pd.DataFrame, colors: dict[str, str] | None = None) -> go.Figure:
    palette = DEFAULT_COLORS.copy()
    if colors:
        palette.update(colors)

    color_map = {
        "Orthopyroxene": palette["primary"],
        "Pigeonite": palette["accent"],
        "Calcic Clinopyroxene": palette["success"],
        "Ferroan Clinopyroxene": palette["warning"],
        "High-Ca Pyroxene": "#6b7280",
    }

    fig = px.scatter(
        data,
        x="quad_x_fs_in_en_fs",
        y="quad_y_wo",
        color="pyroxene_field",
        color_discrete_map=color_map,
        hover_data=["sample_id", "wo_pct", "en_pct", "fs_pct"],
        title="Pyroxene Quadrilateral (Wo-En-Fs)",
    )
    fig.update_xaxes(title="Fs in (En+Fs) [mol%]", range=[0, 100], gridcolor="#dddddd")
    fig.update_yaxes(title="Wo [mol%]", range=[0, 55], gridcolor="#dddddd")
    fig.add_hline(y=5, line_dash="dash", line_color="#7f7f7f")
    fig.add_hline(y=20, line_dash="dash", line_color="#7f7f7f")
    fig.add_annotation(x=98, y=3.5, text="Opx", showarrow=False, xanchor="right")
    fig.add_annotation(x=98, y=12.0, text="Pigeonite", showarrow=False, xanchor="right")
    fig.add_annotation(x=98, y=30.0, text="Cpx", showarrow=False, xanchor="right")
    return _apply_plot_style(fig, colors)


def build_amphibole_leake_figure(data: pd.DataFrame, colors: dict[str, str] | None = None) -> go.Figure:
    palette = DEFAULT_COLORS.copy()
    if colors:
        palette.update(colors)

    color_map = {
        "Calcic": palette["success"],
        "Sodic": palette["warning"],
        "Sodic-Calcic": palette["primary"],
        "Low-B/Unclassified": "#6b7280",
    }

    fig = px.scatter(
        data,
        x="na_b_proxy",
        y="ca_b_proxy",
        color="amphibole_group",
        color_discrete_map=color_map,
        hover_data=["sample_id", "si_apfu", "a_site_na_k_proxy"],
        title="Amphibole Classification (Leake-Style Proxy)",
    )

    fig.add_vline(x=0.5, line_dash="dot", line_color="#7f7f7f")
    fig.add_vline(x=1.5, line_dash="dot", line_color="#7f7f7f")
    fig.add_hline(y=0.5, line_dash="dot", line_color="#7f7f7f")
    fig.add_hline(y=1.5, line_dash="dot", line_color="#7f7f7f")

    fig.add_annotation(x=0.25, y=1.9, text="Calcic", showarrow=False)
    fig.add_annotation(x=1.0, y=1.0, text="Sodic-Calcic", showarrow=False)
    fig.add_annotation(x=1.85, y=0.2, text="Sodic", showarrow=False)

    fig.update_xaxes(title="B-site Na proxy (apfu)", range=[0, 2.1], gridcolor="#dddddd")
    fig.update_yaxes(title="B-site Ca proxy (apfu)", range=[0, 2.1], gridcolor="#dddddd")
    return _apply_plot_style(fig, colors)


def build_feldspar_ternary_figure(data: pd.DataFrame, colors: dict[str, str] | None = None) -> go.Figure:
    palette = DEFAULT_COLORS.copy()
    if colors:
        palette.update(colors)

    color_map = {
        "Alkali Feldspar": palette["warning"],
        "Calcic Plagioclase": palette["success"],
        "Intermediate Plagioclase": palette["primary"],
        "Albite-rich": palette["accent"],
    }

    fig = px.scatter_ternary(
        data,
        a="an_pct",
        b="ab_pct",
        c="or_pct",
        color="feldspar_field",
        color_discrete_map=color_map,
        hover_data=["sample_id", "an_pct", "ab_pct", "or_pct"],
        title="Feldspar Ternary (An-Ab-Or)",
    )
    fig.update_traces(marker={"size": 10, "line": {"width": 0.6, "color": "#ffffff"}})
    fig.update_layout(
        ternary={
            "aaxis": {"title": "An"},
            "baxis": {"title": "Ab"},
            "caxis": {"title": "Or"},
        }
    )
    return _apply_plot_style(fig, colors)


def build_garnet_ternary_figure(data: pd.DataFrame, colors: dict[str, str] | None = None) -> go.Figure:
    fig = px.scatter_ternary(
        data,
        a="prp_pct",
        b="alm_pct",
        c="grs_pct",
        color="sps_pct",
        color_continuous_scale=["#fee2e2", "#8b3a62"],
        hover_data=["sample_id", "dominant_endmember", "sps_pct"],
        title="Garnet Composition (Prp-Alm-Grs; color = Sps)",
    )
    fig.update_traces(marker={"size": 10, "line": {"width": 0.6, "color": "#ffffff"}})
    fig.update_layout(
        ternary={
            "aaxis": {"title": "Prp"},
            "baxis": {"title": "Alm"},
            "caxis": {"title": "Grs"},
        }
    )
    return _apply_plot_style(fig, colors)


def build_garnet_xy_figure(data: pd.DataFrame, colors: dict[str, str] | None = None) -> go.Figure:
    palette = DEFAULT_COLORS.copy()
    if colors:
        palette.update(colors)

    fig = px.scatter(
        data,
        x="prp_pct",
        y="alm_pct",
        color="dominant_endmember",
        color_discrete_sequence=[palette["primary"], palette["accent"], palette["success"], palette["warning"]],
        hover_data=["sample_id", "grs_pct", "sps_pct"],
        title="Garnet Composition Cross-Plot (Prp vs Alm)",
    )
    fig.update_xaxes(title="Prp [mol%]", gridcolor="#dddddd")
    fig.update_yaxes(title="Alm [mol%]", gridcolor="#dddddd")
    return _apply_plot_style(fig, colors)
