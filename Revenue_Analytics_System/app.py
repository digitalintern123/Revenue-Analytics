"""
app.py — Revenue Performance Monitoring System (Fixed)

Fix: Data persists across page navigation by loading from database,
not just session_state. Session state stores the analysis date reference;
actual data is reloaded from SQLite on every page.
"""
from __future__ import annotations
import datetime as dt
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

_modules_dir = PROJECT_ROOT / "modules"
if not _modules_dir.is_dir() or not (_modules_dir / "__init__.py").exists():
    import streamlit as st
    st.set_page_config(page_title="Setup Error", page_icon="\u26a0\ufe0f")
    st.error("## \u26a0\ufe0f Folder structure problem")
    st.markdown(f"The **`modules/`** folder was not found next to `app.py`.\n\n"
                f"**Current path:** `{PROJECT_ROOT}`\n\n"
                "Run `python setup_project.py` first to create the project structure.")
    st.stop()

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from modules.database import init_db, load_all, load_by_date, get_available_dates, get_row_count, reset_database
from modules.data_processor import parse_file, process_and_save
from modules.revenue_analysis import (
    build_full_comparison, compare_dataframes, executive_summary,
    revenue_per_pax, volume_vs_spend_analysis, get_comparison_data,
    aggregate_revenue, classify_trend,
)
from modules.insights import generate_insights

logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title="Revenue Performance Monitor",
    page_icon="\U0001f4ca",
    layout="wide",
    initial_sidebar_state="expanded",
)

engine = init_db()

# ── Session state: store DATES, not DataFrames ──
# This is the key fix: dates survive page navigation reliably,
# and we reload data from the database on each page.
if "analysis_ready" not in st.session_state:
    st.session_state.analysis_ready = False
if "today_date" not in st.session_state:
    st.session_state.today_date = None
if "yesterday_date" not in st.session_state:
    st.session_state.yesterday_date = None
if "last_month_date" not in st.session_state:
    st.session_state.last_month_date = None
if "last_year_date" not in st.session_state:
    st.session_state.last_year_date = None

# ── Sidebar ──
st.sidebar.title("\U0001f4ca Revenue Monitor")
page = st.sidebar.radio(
    "Navigate",
    ["\U0001f4e4 Upload & Analyze",
     "\U0001f4c8 Executive Summary",
     "\U0001f504 Revenue Comparison",
     "\U0001f3ea Outlet Performance",
     "\U0001f916 Business Insights"],
)
st.sidebar.markdown("---")
row_count = get_row_count(engine)
st.sidebar.caption(f"\U0001f4c1 Database rows: **{row_count:,}**")
dates = get_available_dates(engine)
if dates:
    st.sidebar.caption(f"\U0001f4c5 History: **{len(dates)}** dates stored")
if st.session_state.analysis_ready:
    st.sidebar.success(f"\u2705 Analysis active: {st.session_state.today_date}")


# ── Helper: load data for a date from database ──
def load_date_data(target_date):
    """Load all rows for a specific date from the database."""
    if target_date is None:
        return None
    df = load_by_date(target_date, engine)
    if df.empty:
        return None
    return df


# ── Helper: get all 4 period DataFrames ──
def get_analysis_data():
    """Reload all period data from the database using stored dates."""
    today_df = load_date_data(st.session_state.today_date)
    yesterday_df = load_date_data(st.session_state.yesterday_date)
    last_month_df = load_date_data(st.session_state.last_month_date)
    last_year_df = load_date_data(st.session_state.last_year_date)
    return today_df, yesterday_df, last_month_df, last_year_df


# ── Styling helpers ──
def _color_pct(val):
    try:
        v = float(val)
    except (ValueError, TypeError):
        return ""
    if v > 0:
        return "color: #16a34a; font-weight: 600"
    elif v < 0:
        return "color: #dc2626; font-weight: 600"
    return ""


def _fmt_pct_val(v):
    try:
        s = f"{float(v):+.2f}".rstrip("0").rstrip(".")
        return s
    except (ValueError, TypeError):
        return v


def _fmt_metric_pct(v):
    if v is None:
        return None
    s = f"{v:+.1f}".rstrip("0").rstrip(".")
    return s + "%"


def _style_table(df):
    df = df.copy()
    for c in df.columns:
        if "%" in c:
            df[c] = pd.to_numeric(df[c], errors="coerce").round(2)
        elif "Revenue" in c or "Change" in c:
            df[c] = pd.to_numeric(df[c], errors="coerce").round(0).astype("Int64")
        elif "PAX" in c and "%" not in c and "Trend" not in c:
            df[c] = pd.to_numeric(df[c], errors="coerce").round(0).astype("Int64")
    pct_cols = [c for c in df.columns if "%" in c]
    rev_cols = [c for c in df.columns if "Revenue" in c]
    pax_cols = [c for c in df.columns if "PAX" in c and "%" not in c and "Trend" not in c]
    styler = df.style
    if pct_cols:
        styler = styler.map(_color_pct, subset=pct_cols)
        styler = styler.format({c: _fmt_pct_val for c in pct_cols}, na_rep="\\u2014")
    if rev_cols:
        styler = styler.format({c: "{:,.0f}" for c in rev_cols}, na_rep="\\u2014")
    if pax_cols:
        styler = styler.format({c: "{:,.0f}" for c in pax_cols}, na_rep="\\u2014")
    return styler

def _check_data():
    if not st.session_state.analysis_ready or st.session_state.today_date is None:
        st.warning("\U0001f4e4 Please upload today's report on the **Upload & Analyze** page first.")
        st.stop()


# ===================================================================
# PAGE 1 — Upload & Analyze
# ===================================================================
if page == "\U0001f4e4 Upload & Analyze":
    st.title("\U0001f4e4 Upload Revenue Reports")
    st.markdown(
        "Upload **today's report** (required) and optionally upload comparison reports. "
        "If you skip a comparison file, the system checks database history."
    )
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("\U0001f4cb Today's Report *")
        today_file = st.file_uploader("Today's revenue report (required)", type=["pdf", "xlsx", "xls"], key="upload_today")
    with col2:
        st.subheader("\U0001f4cb Yesterday's Report")
        yesterday_file = st.file_uploader("Yesterday's report (optional)", type=["pdf", "xlsx", "xls"], key="upload_yesterday")

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("\U0001f4cb Last Month's Report")
        last_month_file = st.file_uploader("Same date last month (optional)", type=["pdf", "xlsx", "xls"], key="upload_last_month")
    with col4:
        st.subheader("\U0001f4cb Last Year's Report")
        last_year_file = st.file_uploader("Same date last year (optional)", type=["pdf", "xlsx", "xls"], key="upload_last_year")

    st.markdown("---")

    if today_file is not None:
        if st.button("\U0001f680 Process & Analyze", type="primary", use_container_width=True):
            try:
                # ── Process today's report ──
                with st.spinner("Processing today's report..."):
                    today_df, summary = process_and_save(today_file, today_file.name, engine)

                st.success(
                    f"\u2705 Today's report: **{summary['inserted']}** rows inserted, "
                    f"**{summary['skipped_duplicates']}** duplicates skipped."
                )

                # Extract the report date and store it
                report_date = None
                if today_df["Date"].notna().any():
                    ref = today_df["Date"].dropna().iloc[0]
                    report_date = ref.date() if hasattr(ref, "date") else ref
                st.session_state.today_date = report_date

                # ── Process yesterday ──
                if yesterday_file:
                    with st.spinner("Processing yesterday's report..."):
                        ydf, ys = process_and_save(yesterday_file, yesterday_file.name, engine)
                    if ydf["Date"].notna().any():
                        ref = ydf["Date"].dropna().iloc[0]
                        st.session_state.yesterday_date = ref.date() if hasattr(ref, "date") else ref
                    st.success(f"\u2705 Yesterday's report: {ys['inserted']} rows")
                elif report_date:
                    yd = report_date - dt.timedelta(days=1)
                    fb = load_by_date(yd, engine)
                    if not fb.empty:
                        st.session_state.yesterday_date = yd
                        st.info("\U0001f4c2 Yesterday's data loaded from database history")
                    else:
                        st.session_state.yesterday_date = None
                        st.warning("\u26a0\ufe0f No yesterday data available")

                # ── Process last month ──
                if last_month_file:
                    with st.spinner("Processing last month's report..."):
                        mdf, ms = process_and_save(last_month_file, last_month_file.name, engine)
                    if mdf["Date"].notna().any():
                        ref = mdf["Date"].dropna().iloc[0]
                        st.session_state.last_month_date = ref.date() if hasattr(ref, "date") else ref
                    st.success(f"\u2705 Last month's report: {ms['inserted']} rows")
                elif report_date:
                    m = report_date.month - 1 if report_date.month > 1 else 12
                    y = report_date.year if report_date.month > 1 else report_date.year - 1
                    lm_date = dt.date(y, m, min(report_date.day, 28))
                    fb = load_by_date(lm_date, engine)
                    if not fb.empty:
                        st.session_state.last_month_date = lm_date
                        st.info("\U0001f4c2 Last month's data loaded from database history")
                    else:
                        st.session_state.last_month_date = None

                # ── Process last year ──
                if last_year_file:
                    with st.spinner("Processing last year's report..."):
                        lydf, lys = process_and_save(last_year_file, last_year_file.name, engine)
                    if lydf["Date"].notna().any():
                        ref = lydf["Date"].dropna().iloc[0]
                        st.session_state.last_year_date = ref.date() if hasattr(ref, "date") else ref
                    st.success(f"\u2705 Last year's report: {lys['inserted']} rows")
                elif report_date:
                    ly_date = dt.date(report_date.year - 1, report_date.month, min(report_date.day, 28))
                    fb = load_by_date(ly_date, engine)
                    if not fb.empty:
                        st.session_state.last_year_date = ly_date
                        st.info("\U0001f4c2 Last year's data loaded from database history")
                    else:
                        st.session_state.last_year_date = None

                # Mark analysis as ready
                st.session_state.analysis_ready = True

                # Preview
                st.markdown("---")
                st.subheader("\U0001f4cb Extracted Data Preview")
                st.dataframe(today_df.head(30), use_container_width=True)
                st.success("\u2705 **Analysis ready!** Navigate to other pages using the sidebar. Your data is saved and will persist across pages.")

            except ValueError as exc:
                st.error(f"\u26a0\ufe0f {exc}")
            except Exception as exc:
                logging.exception("Upload error")
                st.error(f"\u274c Unexpected error: {exc}")
    else:
        st.info("\U0001f446 Upload at least **today's report** to begin analysis.")

    # Database admin
    with st.expander("\U0001f5c4\ufe0f Database Management"):
        st.write(f"**Total rows stored:** {get_row_count(engine):,}")
        if dates:
            st.write(f"**Date range:** {min(dates)} \u2192 {max(dates)}")
        if st.button("\U0001f5d1\ufe0f Reset Database", type="secondary"):
            reset_database(engine)
            st.session_state.analysis_ready = False
            st.session_state.today_date = None
            st.success("Database cleared.")
            st.rerun()


# ===================================================================
# PAGE 2 — Executive Summary
# ===================================================================
elif page == "\U0001f4c8 Executive Summary":
    st.title("\U0001f4c8 Executive Summary")
    _check_data()

    today_df, yesterday_df, last_month_df, last_year_df = get_analysis_data()
    if today_df is None:
        st.error("Could not load today's data from database.")
        st.stop()

    summary = executive_summary(today_df, yesterday_df)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Revenue Today", f"\u20b9{summary['total_revenue_today']:,.0f}",
                   delta=_fmt_metric_pct(summary['growth_pct']))
    with c2:
        if summary["total_revenue_yesterday"]:
            st.metric("Yesterday Revenue", f"\u20b9{summary['total_revenue_yesterday']:,.0f}")
        else:
            st.metric("Yesterday Revenue", "No data")
    with c3:
        st.metric("PAX Today", f"{summary['total_pax_today']:,.0f}",
                   delta=_fmt_metric_pct(summary['pax_growth_pct']))
    with c4:
        g = summary["growth_pct"]
        st.metric("DoD Growth", _fmt_metric_pct(g) if g is not None else "N/A")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.success(f"\U0001f53a Top Growing Segment: **{summary['top_growing_segment']}**")
    with c2:
        st.error(f"\U0001f53b Top Declining Segment: **{summary['top_declining_segment']}**")

    st.subheader("Revenue by Segment")
    today_agg = aggregate_revenue(today_df)
    if not today_agg.empty:
        seg_pie = today_agg.groupby("Segment", as_index=False)["Revenue"].sum()
        fig = px.pie(seg_pie, names="Segment", values="Revenue", hole=0.4)
        fig.update_layout(height=380, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("PAX by Segment")
    if not today_agg.empty:
        pax_pie = today_agg.groupby("Segment", as_index=False)["Pax"].sum()
        fig_pax = px.pie(pax_pie, names="Segment", values="Pax", hole=0.4)
        fig_pax.update_layout(height=380, margin=dict(t=10, b=10))
        st.plotly_chart(fig_pax, use_container_width=True)

    st.subheader("Revenue by Location")
    if not today_agg.empty:
        loc_bar = today_agg.groupby("Location", as_index=False)["Revenue"].sum().sort_values("Revenue", ascending=True)
        fig2 = px.bar(loc_bar, x="Revenue", y="Location", orientation="h", labels={"Revenue": "Revenue (\u20b9)"})
        fig2.update_layout(height=350, margin=dict(t=10, l=10))
        st.plotly_chart(fig2, use_container_width=True)


# ===================================================================
# PAGE 3 — Revenue Comparison
# ===================================================================
elif page == "\U0001f504 Revenue Comparison":
    st.title("\U0001f504 Revenue Comparison")
    _check_data()

    today_df, yesterday_df, last_month_df, last_year_df = get_analysis_data()
    if today_df is None:
        st.error("Could not load today's data.")
        st.stop()

    s1, s2, s3 = st.columns(3)
    with s1:
        st.write("**DoD:**", "\u2705 Available" if yesterday_df is not None else "\u274c No data")
    with s2:
        st.write("**MoM:**", "\u2705 Available" if last_month_df is not None else "\u274c No data")
    with s3:
        st.write("**YoY:**", "\u2705 Available" if last_year_df is not None else "\u274c No data")

    st.markdown("---")
    tab_dod, tab_mom, tab_yoy, tab_full = st.tabs(["Day over Day", "Month over Month", "Year over Year", "Full Comparison"])

    with tab_dod:
        st.subheader("Today vs Yesterday")
        if yesterday_df is not None:
            dod = compare_dataframes(today_df, yesterday_df, "Today", "Yesterday")
            d = dod[["Segment", "Outlet", "Location", "Revenue_Today", "Revenue_Yesterday", "Pct_Change", "Trend", "Pax_Today", "Pax_Yesterday", "Pax_Pct_Change", "Pax_Trend"]].copy()
            d.rename(columns={"Revenue_Today": "Today Revenue", "Revenue_Yesterday": "Yesterday Revenue", "Pct_Change": "Rev DoD %", "Pax_Today": "Today PAX", "Pax_Yesterday": "Yesterday PAX", "Pax_Pct_Change": "PAX DoD %", "Pax_Trend": "PAX Trend"}, inplace=True)
            st.dataframe(_style_table(d), use_container_width=True, height=500)
        else:
            st.info("Upload yesterday's report for DoD comparison.")

    with tab_mom:
        st.subheader("Today vs Last Month")
        if last_month_df is not None:
            mom = compare_dataframes(today_df, last_month_df, "Today", "LastMonth")
            d = mom[["Segment", "Outlet", "Location", "Revenue_Today", "Revenue_LastMonth", "Pct_Change", "Trend", "Pax_Today", "Pax_LastMonth", "Pax_Pct_Change", "Pax_Trend"]].copy()
            d.rename(columns={"Revenue_Today": "Today Revenue", "Revenue_LastMonth": "Last Month Revenue", "Pct_Change": "Rev MoM %", "Pax_Today": "Today PAX", "Pax_LastMonth": "Last Month PAX", "Pax_Pct_Change": "PAX MoM %", "Pax_Trend": "PAX Trend"}, inplace=True)
            st.dataframe(_style_table(d), use_container_width=True, height=500)
        else:
            st.info("Upload last month's report for MoM comparison.")

    with tab_yoy:
        st.subheader("Today vs Last Year")
        if last_year_df is not None:
            yoy = compare_dataframes(today_df, last_year_df, "Today", "LastYear")
            d = yoy[["Segment", "Outlet", "Location", "Revenue_Today", "Revenue_LastYear", "Pct_Change", "Trend", "Pax_Today", "Pax_LastYear", "Pax_Pct_Change", "Pax_Trend"]].copy()
            d.rename(columns={"Revenue_Today": "Today Revenue", "Revenue_LastYear": "Last Year Revenue", "Pct_Change": "Rev YoY %", "Pax_Today": "Today PAX", "Pax_LastYear": "Last Year PAX", "Pax_Pct_Change": "PAX YoY %", "Pax_Trend": "PAX Trend"}, inplace=True)
            st.dataframe(_style_table(d), use_container_width=True, height=500)
        else:
            st.info("Upload last year's report for YoY comparison.")

    with tab_full:
        st.subheader("Full Comparison Table")
        full = build_full_comparison(today_df, yesterday_df, last_month_df, last_year_df)
        if not full.empty:
            st.dataframe(_style_table(full), use_container_width=True, height=500)
        else:
            st.info("Upload comparison reports to see the full table.")


# ===================================================================
# PAGE 4 — Outlet Performance
# ===================================================================
elif page == "\U0001f3ea Outlet Performance":
    st.title("\U0001f3ea Outlet Performance")
    _check_data()

    today_df, yesterday_df, _, _ = get_analysis_data()
    if today_df is None:
        st.error("Could not load today's data.")
        st.stop()

    today_agg = aggregate_revenue(today_df)
    today_agg["Label"] = today_agg["Outlet"] + " \u2014 " + today_agg["Location"]

    col_t, col_b = st.columns(2)
    with col_t:
        st.subheader("\U0001f51d Top 10 Outlets")
        top10 = today_agg.nlargest(10, "Revenue")
        fig_t = px.bar(top10, x="Revenue", y="Label", orientation="h", color="Revenue",
                       color_continuous_scale=["#fbbf24", "#16a34a"], labels={"Revenue": "Revenue (\u20b9)", "Label": ""})
        fig_t.update_layout(height=420, margin=dict(l=10, t=10), showlegend=False)
        st.plotly_chart(fig_t, use_container_width=True)

    with col_b:
        st.subheader("\u26a0\ufe0f Bottom 10 Outlets")
        bot10 = today_agg.nsmallest(10, "Revenue")
        fig_b = px.bar(bot10, x="Revenue", y="Label", orientation="h", color="Revenue",
                       color_continuous_scale=["#dc2626", "#fbbf24"], labels={"Revenue": "Revenue (\u20b9)", "Label": ""})
        fig_b.update_layout(height=420, margin=dict(l=10, t=10), showlegend=False)
        st.plotly_chart(fig_b, use_container_width=True)

    if yesterday_df is not None and not yesterday_df.empty:
        st.markdown("---")
        st.subheader("\U0001f4ca DoD Revenue Change by Outlet")
        dod = compare_dataframes(today_df, yesterday_df, "Today", "Yesterday").dropna(subset=["Pct_Change"])
        dod["Label"] = dod["Outlet"] + " \u2014 " + dod["Location"]
        dod = dod.sort_values("Pct_Change", ascending=True)
        fig_dod = px.bar(dod, x="Pct_Change", y="Label", orientation="h", color="Pct_Change",
                         color_continuous_scale=["#dc2626", "#f5f5f5", "#16a34a"], color_continuous_midpoint=0,
                         labels={"Pct_Change": "Change %", "Label": ""})
        fig_dod.update_layout(height=max(400, len(dod) * 25), margin=dict(l=10, t=10))
        st.plotly_chart(fig_dod, use_container_width=True)

    st.markdown("---")
    st.subheader("\U0001f4b0 Revenue per PAX")
    rpp = revenue_per_pax(today_df)
    if not rpp.empty:
        rpp_d = rpp.dropna(subset=["Rev_Per_Pax"]).sort_values("Rev_Per_Pax", ascending=False)
        rpp_d = rpp_d.rename(columns={"Rev_Per_Pax": "Rev/PAX (\u20b9)", "Revenue": "Revenue (\u20b9)"})
        st.dataframe(rpp_d.head(20), use_container_width=True)


# ===================================================================
# PAGE 5 — Business Insights
# ===================================================================
elif page == "\U0001f916 Business Insights":
    st.title("\U0001f916 Business Insights")
    _check_data()

    today_df, yesterday_df, last_month_df, last_year_df = get_analysis_data()
    if today_df is None:
        st.error("Could not load today's data.")
        st.stop()

    if st.button("\U0001f4dd Generate Management Summary", type="primary", use_container_width=True):
        with st.spinner("Analyzing..."):
            report = generate_insights(today_df, yesterday_df, last_month_df, last_year_df)
        st.markdown(report)
    else:
        st.info("Click the button above to generate the management summary.")

    if yesterday_df is not None and not yesterday_df.empty:
        st.markdown("---")
        st.subheader("\U0001f50d Volume vs Spend Driver Analysis")
        vs = volume_vs_spend_analysis(today_df, yesterday_df)
        if not vs.empty:
            d = vs[["Segment", "Outlet", "Location", "Pax_Chg%", "Rev_Chg%", "RPP_Chg%", "Driver"]].copy()
            d.rename(columns={"Pax_Chg%": "PAX \u0394%", "Rev_Chg%": "Revenue \u0394%", "RPP_Chg%": "Rev/Pax \u0394%"}, inplace=True)
            pct_vs_cols = [c for c in d.columns if "%" in c]
            vs_styler = d.style.format({c: _fmt_pct_val for c in pct_vs_cols}, na_rep="\\u2014")
            st.dataframe(vs_styler, use_container_width=True, height=400)
