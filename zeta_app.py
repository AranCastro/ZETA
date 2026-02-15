from __future__ import annotations

import html
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zeta import (
    MINERAL_SPECS,
    build_georoc_table,
    build_publication_table,
    build_ternary_table,
    process_dataframe,
    summarize_quality,
)
from zeta.io import read_uploaded_table
from zeta.plot import (
    build_amphibole_leake_figure,
    build_feldspar_ternary_figure,
    build_garnet_ternary_figure,
    build_garnet_xy_figure,
    build_pyroxene_quad_figure,
    prepare_amphibole_leake_data,
    prepare_feldspar_ternary_data,
    prepare_garnet_composition_data,
    prepare_pyroxene_quad_data,
)
from zeta.power_tools import (
    analyze_zoning_profile,
    classify_nomenclature,
    compute_thermobarometry,
    estimate_diffusion_and_closure,
    estimate_fe3_by_charge_balance,
    fe_mg_equilibrium_filter,
    normalize_apfu,
    project_endmembers,
    renormalize_oxides,
    tag_core_rim,
)
from zeta.samples import sample_formula_data, sample_microprobe_data, sample_traverse_data

LOGO_PATH = ROOT / "zeta" / "assets" / "zeta_logo_hex.svg"

PALETTE = {
    "background": "#f8f9fa",
    "sidebar": "#2d1b2e",
    "primary": "#8b3a62",
    "accent": "#d4af37",
    "text": "#1a1a1a",
    "success": "#059669",
    "warning": "#ea580c",
}


def _to_csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8")


def _render_sample_box() -> None:
    with st.expander("Sample Data Box", expanded=False):
        st.caption("Run ZETA even without your own files. Built-in samples are available for processing and plotting.")
        sample_microprobe = sample_microprobe_data()
        sample_formula = sample_formula_data()
        sample_traverse = sample_traverse_data()

        st.download_button(
            "Download sample microprobe table",
            data=_to_csv_bytes(sample_microprobe),
            file_name="zeta_sample_microprobe.csv",
            mime="text/csv",
            key="dl_sample_microprobe",
        )
        st.download_button(
            "Download sample formula table",
            data=_to_csv_bytes(sample_formula),
            file_name="zeta_sample_formulas.csv",
            mime="text/csv",
            key="dl_sample_formulas",
        )
        st.download_button(
            "Download sample traverse table",
            data=_to_csv_bytes(sample_traverse),
            file_name="zeta_sample_traverse.csv",
            mime="text/csv",
            key="dl_sample_traverse",
        )


def _apply_theme() -> None:
    st.markdown(
        f"""
        <style>
        :root {{
            --zeta-bg: {PALETTE["background"]};
            --zeta-sidebar: {PALETTE["sidebar"]};
            --zeta-primary: {PALETTE["primary"]};
            --zeta-accent: {PALETTE["accent"]};
            --zeta-text: {PALETTE["text"]};
            --zeta-success: {PALETTE["success"]};
            --zeta-warning: {PALETTE["warning"]};
        }}

        [data-testid="stAppViewContainer"] {{
            background-color: var(--zeta-bg);
        }}

        [data-testid="stHeader"] {{
            background: transparent;
        }}

        .main * {{
            color: var(--zeta-text);
        }}

        [data-testid="stSidebar"] {{
            background-color: var(--zeta-sidebar);
        }}

        [data-testid="stSidebar"] * {{
            color: #f8f9fa;
        }}

        [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div {{
            background-color: rgba(255, 255, 255, 0.12);
            border: 1px solid rgba(255, 255, 255, 0.25);
        }}

        [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] svg {{
            fill: #f8f9fa;
        }}

        div.stButton > button,
        div.stDownloadButton > button {{
            background-color: var(--zeta-primary);
            color: #f8f9fa;
            border: 1px solid var(--zeta-primary);
            border-radius: 8px;
        }}

        div.stButton > button:hover,
        div.stDownloadButton > button:hover {{
            background-color: var(--zeta-accent);
            border-color: var(--zeta-accent);
            color: var(--zeta-text);
        }}

        div.stButton > button:focus,
        div.stDownloadButton > button:focus {{
            box-shadow: 0 0 0 0.2rem rgba(212, 175, 55, 0.35);
        }}

        [data-testid="stMetric"] {{
            background: #ffffff;
            border: 1px solid #e6e6e6;
            border-radius: 10px;
            padding: 0.75rem;
        }}

        .zeta-alert {{
            border-radius: 8px;
            padding: 0.7rem 0.9rem;
            margin: 0.25rem 0 0.75rem 0;
            font-weight: 600;
        }}

        .zeta-alert.success {{
            background-color: rgba(5, 150, 105, 0.12);
            border-left: 0.4rem solid var(--zeta-success);
            color: var(--zeta-text);
        }}

        .zeta-alert.warning {{
            background-color: rgba(234, 88, 12, 0.12);
            border-left: 0.4rem solid var(--zeta-warning);
            color: var(--zeta-text);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_status(kind: str, message: str) -> None:
    if kind not in {"success", "warning"}:
        raise ValueError(f"Unsupported status kind: {kind}")
    safe = html.escape(message)
    st.markdown(f"<div class='zeta-alert {kind}'>{safe}</div>", unsafe_allow_html=True)


def _init_state() -> None:
    if "zeta_processed" not in st.session_state:
        st.session_state["zeta_processed"] = None
    if "zeta_source_name" not in st.session_state:
        st.session_state["zeta_source_name"] = None


def main() -> None:
    st.set_page_config(page_title="ZETA", layout="wide")
    _init_state()
    _apply_theme()

    logo_col, title_col = st.columns([1, 5])
    with logo_col:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=140)
    with title_col:
        st.title("ZETA")
        st.caption("Smart Mineral Formula Calculator for EPMA batch data")

    st.markdown(
        """
        Upload CSV/Excel microprobe data, run structural formula calculations,
        auto-detect mineral classes, and export QC-ready tables.
        """
    )

    _render_sample_box()
    input_source = st.radio(
        "EPMA input source",
        options=["Upload EPMA table", "Use built-in sample microprobe data"],
        horizontal=True,
        key="zeta_epma_input_source",
    )

    raw_frame: pd.DataFrame | None = None
    if input_source == "Use built-in sample microprobe data":
        raw_frame = sample_microprobe_data()
        st.session_state["zeta_source_name"] = "built-in-sample-microprobe"
        _render_status("success", f"Loaded built-in sample microprobe dataset ({len(raw_frame)} analyses).")
        st.dataframe(raw_frame.head(20), use_container_width=True, hide_index=True)
    else:
        uploaded = st.file_uploader("Upload EPMA table", type=["csv", "xlsx", "xls"])
        if uploaded is not None:
            try:
                raw_frame = read_uploaded_table(uploaded.name, uploaded.getvalue())
            except Exception as exc:
                st.error(f"Failed to read file: {exc}")
            else:
                st.session_state["zeta_source_name"] = uploaded.name
                _render_status("success", f"Loaded {uploaded.name} with {len(raw_frame)} analyses.")
                st.dataframe(raw_frame.head(20), use_container_width=True, hide_index=True)

    if raw_frame is not None:
        st.subheader("Processing Settings")
        sample_options = ["(row index)"] + [str(column) for column in raw_frame.columns]
        sample_col = st.selectbox("Sample ID column", options=sample_options, index=0)

        mode = st.radio(
            "Mineral assignment",
            options=["Auto detect per analysis", "Manual single mineral"],
            horizontal=True,
        )

        manual_mineral = "garnet"
        if mode == "Manual single mineral":
            manual_mineral = st.selectbox("Mineral", options=list(MINERAL_SPECS.keys()), index=0)

        if st.button("Run ZETA", type="primary"):
            mineral_mode = "manual" if mode == "Manual single mineral" else "auto"
            sample_id_col = None if sample_col == "(row index)" else sample_col
            try:
                processed = process_dataframe(
                    raw_frame,
                    mineral_mode=mineral_mode,
                    manual_mineral=manual_mineral,
                    sample_id_column=sample_id_col,
                )
            except Exception as exc:
                st.error(f"Processing failed: {exc}")
            else:
                st.session_state["zeta_processed"] = processed

    processed = st.session_state.get("zeta_processed")
    has_processed = isinstance(processed, pd.DataFrame) and not processed.empty

    if has_processed:
        st.subheader("QC Summary")
        metric_a, metric_b, metric_c = st.columns(3)
        with metric_a:
            st.metric("Analyses", int(len(processed)))
        with metric_b:
            st.metric("QC Pass", f"{100.0 * processed['qc_pass'].mean():.1f}%")
        with metric_c:
            st.metric("Outliers", int(processed["qc_outlier"].sum()))

        summary = summarize_quality(processed)
        st.dataframe(summary, use_container_width=True, hide_index=True)

        chart_data = processed.copy()
        chart_data["qc_label"] = np.where(chart_data["qc_pass"], "pass", "flagged")

        st.subheader("Visual QC")
        fig = px.scatter(
            chart_data,
            x="oxide_total",
            y="charge_imbalance",
            color="qc_label",
            color_discrete_map={"pass": PALETTE["success"], "flagged": PALETTE["warning"]},
            symbol="mineral",
            hover_data=["sample_id", "qc_flags", "detection_confidence"],
            title="Oxide Total vs Charge Imbalance",
        )
        fig.update_layout(
            paper_bgcolor=PALETTE["background"],
            plot_bgcolor="#ffffff",
            font_color=PALETTE["text"],
        )
        fig.update_xaxes(gridcolor="#dddddd")
        fig.update_yaxes(gridcolor="#dddddd")
        fig.add_hline(y=0.0, line_dash="dot", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Processed Results")
        st.dataframe(processed, use_container_width=True, hide_index=True)

        publication = build_publication_table(processed)
        ternary = build_ternary_table(processed)
        georoc = build_georoc_table(processed)

        st.subheader("Exports")
        export_a, export_b, export_c = st.columns(3)
        with export_a:
            st.download_button(
                "Download publication table",
                data=_to_csv_bytes(publication),
                file_name="zeta_publication_table.csv",
                mime="text/csv",
            )
        with export_b:
            st.download_button(
                "Download ternary data",
                data=_to_csv_bytes(ternary),
                file_name="zeta_ternary_data.csv",
                mime="text/csv",
            )
        with export_c:
            st.download_button(
                "Download GeoRoc-style table",
                data=_to_csv_bytes(georoc),
                file_name="zeta_georoc.csv",
                mime="text/csv",
            )

        _render_status(
            "warning",
            "Amphibole handling uses simplified 23-oxygen normalization. "
            "Use dedicated amphibole routines for publication-grade site allocation.",
        )
    else:
        st.info("Upload data and run ZETA to generate formulas and QC flags.")

    st.divider()
    st.header("ZETA-Plot: Discrimination Diagram Generator")
    st.caption("Upload mineral formulas from ZETA and auto-generate classification diagrams.")

    source_options = (
        ["Use current ZETA results", "Use built-in sample formulas", "Upload formula table"]
        if has_processed
        else ["Use built-in sample formulas", "Upload formula table", "Use current ZETA results"]
    )
    plot_source_mode = st.radio(
        "Formula source",
        options=source_options,
        horizontal=True,
        key="zeta_plot_source_mode",
    )

    plot_frame: pd.DataFrame | None
    if plot_source_mode == "Use current ZETA results":
        if has_processed:
            plot_frame = processed.copy()
        else:
            plot_frame = None
            st.info("No current ZETA results found. Choose sample formulas or upload a formula table.")
    elif plot_source_mode == "Use built-in sample formulas":
        plot_frame = sample_formula_data()
        _render_status("success", f"Loaded built-in sample formula dataset ({len(plot_frame)} rows).")
        st.dataframe(plot_frame.head(20), use_container_width=True, hide_index=True)
    else:
        plot_upload = st.file_uploader(
            "Upload ZETA formula table (CSV/Excel)",
            type=["csv", "xlsx", "xls"],
            key="zeta_plot_upload",
        )
        if plot_upload is None:
            plot_frame = None
            st.info("Upload a formula table to enable ZETA-Plot diagrams.")
        else:
            try:
                plot_frame = read_uploaded_table(plot_upload.name, plot_upload.getvalue())
            except Exception as exc:
                plot_frame = None
                st.error(f"Failed to read formula table: {exc}")
            else:
                _render_status("success", f"Loaded diagram source: {plot_upload.name} ({len(plot_frame)} rows).")

    if plot_frame is None:
        return

    diagram_tabs = st.tabs(
        [
            "Pyroxene Quad (Wo-En-Fs)",
            "Amphibole (Leake Proxy)",
            "Feldspar Ternary (An-Ab-Or)",
            "Garnet Composition",
        ]
    )

    with diagram_tabs[0]:
        try:
            pyroxene_data = prepare_pyroxene_quad_data(plot_frame)
        except ValueError as exc:
            st.info(str(exc))
        else:
            pyroxene_fig = build_pyroxene_quad_figure(pyroxene_data, colors=PALETTE)
            st.plotly_chart(pyroxene_fig, use_container_width=True)
            st.dataframe(pyroxene_data, use_container_width=True, hide_index=True)
            st.download_button(
                "Download pyroxene diagram data",
                data=_to_csv_bytes(pyroxene_data),
                file_name="zeta_plot_pyroxene_quad.csv",
                mime="text/csv",
                key="dl_zeta_plot_pyroxene",
            )

    with diagram_tabs[1]:
        try:
            amphibole_data = prepare_amphibole_leake_data(plot_frame)
        except ValueError as exc:
            st.info(str(exc))
        else:
            amphibole_fig = build_amphibole_leake_figure(amphibole_data, colors=PALETTE)
            st.plotly_chart(amphibole_fig, use_container_width=True)
            st.dataframe(amphibole_data, use_container_width=True, hide_index=True)
            st.download_button(
                "Download amphibole diagram data",
                data=_to_csv_bytes(amphibole_data),
                file_name="zeta_plot_amphibole_leake_proxy.csv",
                mime="text/csv",
                key="dl_zeta_plot_amphibole",
            )

    with diagram_tabs[2]:
        try:
            feldspar_data = prepare_feldspar_ternary_data(plot_frame)
        except ValueError as exc:
            st.info(str(exc))
        else:
            feldspar_fig = build_feldspar_ternary_figure(feldspar_data, colors=PALETTE)
            st.plotly_chart(feldspar_fig, use_container_width=True)
            st.dataframe(feldspar_data, use_container_width=True, hide_index=True)
            st.download_button(
                "Download feldspar diagram data",
                data=_to_csv_bytes(feldspar_data),
                file_name="zeta_plot_feldspar_ternary.csv",
                mime="text/csv",
                key="dl_zeta_plot_feldspar",
            )

    with diagram_tabs[3]:
        try:
            garnet_data = prepare_garnet_composition_data(plot_frame)
        except ValueError as exc:
            st.info(str(exc))
        else:
            garnet_ternary = build_garnet_ternary_figure(garnet_data, colors=PALETTE)
            st.plotly_chart(garnet_ternary, use_container_width=True)

            garnet_xy = build_garnet_xy_figure(garnet_data, colors=PALETTE)
            st.plotly_chart(garnet_xy, use_container_width=True)

            st.dataframe(garnet_data, use_container_width=True, hide_index=True)
            st.download_button(
                "Download garnet diagram data",
                data=_to_csv_bytes(garnet_data),
                file_name="zeta_plot_garnet_composition.csv",
                mime="text/csv",
                key="dl_zeta_plot_garnet",
            )

    st.divider()
    st.header("Advanced Features (Power User Tools)")

    power_tabs = st.tabs(
        [
            "Classification & Naming",
            "Zoning Profile Analysis",
            "Recalculation Options",
            "Thermobarometry & Filters",
        ]
    )

    with power_tabs[0]:
        st.caption(
            "Auto-classification for amphibole, pyroxene, tourmaline, garnet, and spinel "
            "with direct mineral-name outputs."
        )
        classed = classify_nomenclature(plot_frame)
        st.dataframe(classed, use_container_width=True, hide_index=True)
        st.download_button(
            "Download classification table",
            data=_to_csv_bytes(classed),
            file_name="zeta_power_classification.csv",
            mime="text/csv",
            key="dl_zeta_power_classification",
        )

    with power_tabs[1]:
        st.caption("Import core-to-rim traverse data, detect zoning style, and estimate diffusion/closure terms.")
        zoning_mode = st.radio(
            "Traverse source",
            options=["Use current formula table", "Use built-in sample traverse", "Upload traverse file"],
            horizontal=True,
            key="zeta_zoning_source_mode",
        )

        traverse_frame: pd.DataFrame | None
        if zoning_mode == "Use current formula table":
            traverse_frame = plot_frame.copy()
        elif zoning_mode == "Use built-in sample traverse":
            traverse_frame = sample_traverse_data()
            _render_status("success", f"Loaded built-in sample traverse dataset ({len(traverse_frame)} rows).")
            st.dataframe(traverse_frame, use_container_width=True, hide_index=True)
        else:
            traverse_upload = st.file_uploader(
                "Upload traverse table (CSV/Excel)",
                type=["csv", "xlsx", "xls"],
                key="zeta_traverse_upload",
            )
            if traverse_upload is None:
                traverse_frame = None
                st.info("Upload a traverse file to run zoning-profile tools.")
            else:
                try:
                    traverse_frame = read_uploaded_table(traverse_upload.name, traverse_upload.getvalue())
                except Exception as exc:
                    traverse_frame = None
                    st.error(f"Failed to read traverse table: {exc}")
                else:
                    _render_status(
                        "success",
                        f"Loaded traverse source: {traverse_upload.name} ({len(traverse_frame)} rows).",
                    )

        if traverse_frame is not None and not traverse_frame.empty:
            numeric_columns = [
                column
                for column in traverse_frame.columns
                if pd.api.types.is_numeric_dtype(traverse_frame[column])
            ]
            if not numeric_columns:
                st.info("Traverse table needs numeric columns for distance and element values.")
            else:
                default_distance = numeric_columns[0]
                distance_col = st.selectbox(
                    "Distance column (core -> rim)", options=numeric_columns, index=0, key="zeta_dist_col"
                )
                element_options = [column for column in numeric_columns if column != distance_col]
                selected_elements = st.multiselect(
                    "Element/parameter columns",
                    options=element_options,
                    default=element_options[: min(3, len(element_options))],
                    key="zeta_element_cols",
                )
                normal_direction = st.selectbox(
                    "Assumed normal zoning direction",
                    options=["decreasing", "increasing"],
                    index=0,
                    key="zeta_normal_direction",
                )
                sector_candidates = ["(none)"] + [str(col) for col in traverse_frame.columns if col != distance_col]
                sector_choice = st.selectbox(
                    "Sector column (optional)", options=sector_candidates, index=0, key="zeta_sector_col"
                )
                sector_col = None if sector_choice == "(none)" else sector_choice

                try:
                    zoning_summary, zoning_profiles = analyze_zoning_profile(
                        traverse_frame=traverse_frame,
                        distance_col=distance_col,
                        element_cols=selected_elements,
                        normal_direction=normal_direction,
                        sector_col=sector_col,
                    )
                except ValueError as exc:
                    st.info(str(exc))
                else:
                    st.subheader("Zoning Classification")
                    st.dataframe(zoning_summary, use_container_width=True, hide_index=True)

                    profile_long = zoning_profiles.melt(
                        id_vars=["distance"], var_name="element", value_name="value"
                    ).dropna(subset=["value"])
                    profile_fig = px.line(
                        profile_long,
                        x="distance",
                        y="value",
                        color="element",
                        markers=True,
                        title="Traverse Profiles (Core -> Rim)",
                    )
                    profile_fig.update_layout(
                        paper_bgcolor=PALETTE["background"],
                        plot_bgcolor="#ffffff",
                        font_color=PALETTE["text"],
                    )
                    profile_fig.update_xaxes(gridcolor="#dddddd")
                    profile_fig.update_yaxes(gridcolor="#dddddd")
                    st.plotly_chart(profile_fig, use_container_width=True)

                    st.download_button(
                        "Download zoning summary",
                        data=_to_csv_bytes(zoning_summary),
                        file_name="zeta_power_zoning_summary.csv",
                        mime="text/csv",
                        key="dl_zeta_power_zoning",
                    )

                    if selected_elements:
                        st.subheader("Diffusion / Closure Estimate")
                        diffusion_element = st.selectbox(
                            "Element for diffusion estimate",
                            options=selected_elements,
                            index=0,
                            key="zeta_diff_element",
                        )
                        duration_myr = st.number_input(
                            "Duration (Myr)", min_value=0.001, value=1.0, step=0.1, key="zeta_diff_duration"
                        )
                        d0 = st.number_input(
                            "D0 (m2/s)", min_value=1e-30, value=1e-12, format="%.2e", key="zeta_diff_d0"
                        )
                        ea = st.number_input(
                            "Activation energy (kJ/mol)",
                            min_value=1.0,
                            value=200.0,
                            step=5.0,
                            key="zeta_diff_ea",
                        )
                        try:
                            diffusion_result = estimate_diffusion_and_closure(
                                traverse_frame=traverse_frame,
                                distance_col=distance_col,
                                element_col=diffusion_element,
                                duration_myr=float(duration_myr),
                                d0_m2_s=float(d0),
                                activation_energy_kj_mol=float(ea),
                            )
                        except ValueError as exc:
                            st.info(str(exc))
                        else:
                            dcol1, dcol2, dcol3 = st.columns(3)
                            with dcol1:
                                st.metric("Diffusion length (um)", f"{diffusion_result['diffusion_length_um']:.2f}")
                            with dcol2:
                                st.metric("D_eff (m2/s)", f"{diffusion_result['effective_diffusivity_m2_s']:.2e}")
                            with dcol3:
                                closure_value = diffusion_result["closure_temperature_c"]
                                display = f"{closure_value:.1f}" if np.isfinite(closure_value) else "n/a"
                                st.metric("Closure T (C)", display)

    with power_tabs[2]:
        st.caption(
            "Fe2+/Fe3+ estimation, anhydrous/hydrous renormalization, normalization to cation targets, "
            "and endmember projections."
        )

        recalc_source = plot_frame.copy()

        st.subheader("Fe Recalculation (Charge Balance)")
        try:
            fe_recalc = estimate_fe3_by_charge_balance(recalc_source)
        except ValueError as exc:
            st.info(str(exc))
        else:
            st.dataframe(
                fe_recalc[
                    [
                        col
                        for col in [
                            "sample_id",
                            "apfu_Fe2",
                            "apfu_Fe3",
                            "apfu_Fe2_recalc",
                            "apfu_Fe3_recalc",
                            "fe3_fraction_recalc",
                        ]
                        if col in fe_recalc.columns
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                "Download Fe recalculation",
                data=_to_csv_bytes(fe_recalc),
                file_name="zeta_power_fe_recalc.csv",
                mime="text/csv",
                key="dl_zeta_power_fe",
            )

        st.subheader("Oxide Renormalization")
        renorm_target = st.number_input(
            "Target oxide total", min_value=50.0, max_value=110.0, value=100.0, step=0.5, key="zeta_renorm_target"
        )
        include_hydrous = st.checkbox("Include hydrous basis (wt_H2O)", value=False, key="zeta_include_hydrous")
        try:
            oxide_renorm = renormalize_oxides(
                recalc_source,
                target_total=float(renorm_target),
                include_hydrous=bool(include_hydrous),
            )
        except ValueError as exc:
            st.info(str(exc))
        else:
            st.dataframe(oxide_renorm.head(30), use_container_width=True, hide_index=True)
            st.download_button(
                "Download oxide renormalization",
                data=_to_csv_bytes(oxide_renorm),
                file_name="zeta_power_oxide_renormalized.csv",
                mime="text/csv",
                key="dl_zeta_power_oxide_renorm",
            )

        st.subheader("APFU Cation Normalization")
        target_cations = st.number_input(
            "Target cation sum", min_value=1.0, max_value=30.0, value=8.0, step=0.1, key="zeta_target_cations"
        )
        try:
            apfu_norm = normalize_apfu(recalc_source, target_cations=float(target_cations))
        except ValueError as exc:
            st.info(str(exc))
        else:
            st.dataframe(apfu_norm.head(30), use_container_width=True, hide_index=True)
            st.download_button(
                "Download APFU normalization",
                data=_to_csv_bytes(apfu_norm),
                file_name="zeta_power_apfu_normalized.csv",
                mime="text/csv",
                key="dl_zeta_power_apfu_norm",
            )

        st.subheader("Endmember Projection")
        endmember_frame = project_endmembers(recalc_source)
        st.dataframe(endmember_frame, use_container_width=True, hide_index=True)
        st.download_button(
            "Download endmember projection",
            data=_to_csv_bytes(endmember_frame),
            file_name="zeta_power_endmember_projection.csv",
            mime="text/csv",
            key="dl_zeta_power_endmembers",
        )

    with power_tabs[3]:
        st.caption(
            "Single-mineral T/P indicators, Fe-Mg equilibrium tests, and rim/core selection tools for "
            "thermobarometry workflows."
        )

        st.subheader("Single-Mineral Indicators")
        thermo = compute_thermobarometry(plot_frame)
        st.dataframe(thermo, use_container_width=True, hide_index=True)
        st.download_button(
            "Download thermobarometry table",
            data=_to_csv_bytes(thermo),
            file_name="zeta_power_thermobarometry.csv",
            mime="text/csv",
            key="dl_zeta_power_thermo",
        )

        st.subheader("Fe-Mg Equilibrium Filter")
        sample_candidates = [str(col) for col in plot_frame.columns]
        default_sample_idx = sample_candidates.index("sample_id") if "sample_id" in sample_candidates else 0
        sample_id_col = st.selectbox(
            "Sample ID column", options=sample_candidates, index=default_sample_idx, key="zeta_eq_sample_col"
        )

        phase_default_idx = sample_candidates.index("mineral") if "mineral" in sample_candidates else 0
        phase_col = st.selectbox("Phase column", options=sample_candidates, index=phase_default_idx, key="zeta_eq_phase_col")
        phase_a = st.text_input("Phase A", value="garnet", key="zeta_eq_phase_a")
        phase_b = st.text_input("Phase B", value="biotite", key="zeta_eq_phase_b")
        kd_min = st.number_input("Kd min", value=0.5, step=0.1, key="zeta_eq_kd_min")
        kd_max = st.number_input("Kd max", value=5.0, step=0.1, key="zeta_eq_kd_max")
        try:
            kd_table = fe_mg_equilibrium_filter(
                table=plot_frame,
                phase_a=phase_a,
                phase_b=phase_b,
                sample_id_col=sample_id_col,
                phase_col=phase_col,
                kd_min=float(kd_min),
                kd_max=float(kd_max),
            )
        except ValueError as exc:
            st.info(str(exc))
        else:
            st.dataframe(kd_table, use_container_width=True, hide_index=True)
            st.download_button(
                "Download Fe-Mg equilibrium filter",
                data=_to_csv_bytes(kd_table),
                file_name="zeta_power_equilibrium_filter.csv",
                mime="text/csv",
                key="dl_zeta_power_eq",
            )

        st.subheader("Core/Rim Selector")
        numeric_columns = [
            column for column in plot_frame.columns if pd.api.types.is_numeric_dtype(plot_frame[column])
        ]
        if numeric_columns:
            default_dist_col = "distance" if "distance" in numeric_columns else numeric_columns[0]
            dist_col = st.selectbox(
                "Distance column for zone tagging",
                options=numeric_columns,
                index=numeric_columns.index(default_dist_col),
                key="zeta_core_rim_dist_col",
            )
            core_frac = st.slider("Core fraction", min_value=0.05, max_value=0.45, value=0.25, step=0.05, key="zeta_core_frac")
            rim_frac = st.slider("Rim fraction", min_value=0.05, max_value=0.45, value=0.25, step=0.05, key="zeta_rim_frac")
            try:
                zones = tag_core_rim(
                    table=plot_frame,
                    distance_col=dist_col,
                    core_fraction=float(core_frac),
                    rim_fraction=float(rim_frac),
                )
            except ValueError as exc:
                st.info(str(exc))
            else:
                counts = zones["zone"].value_counts().rename_axis("zone").reset_index(name="count")
                ccol, dcol = st.columns([1, 3])
                with ccol:
                    st.dataframe(counts, use_container_width=True, hide_index=True)
                with dcol:
                    st.dataframe(zones, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download core/rim tagging",
                    data=_to_csv_bytes(zones),
                    file_name="zeta_power_core_rim_tagging.csv",
                    mime="text/csv",
                    key="dl_zeta_power_core_rim",
                )
        else:
            st.info("No numeric columns available for core/rim distance tagging.")


if __name__ == "__main__":
    main()
