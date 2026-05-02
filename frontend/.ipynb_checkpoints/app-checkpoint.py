# ── Dependencias ────────────────────────────────────────────────────────────
import io
import sys
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from great_tables import GT, style, loc
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.inference import (
    get_filter_options,
    get_predictions,
    get_model_evaluation,
    submit_feedback,
    get_feedback_summary,
    get_system_events,
    get_api_health_metrics,
    export_predictions,
)

# ── Configuración de página ───────────────────────────────────────────────────
st.set_page_config(
    page_title="1C Company · Forecast",
    layout="wide",
)

st.markdown("""
<style>
/* ── Sidebar ─────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: #1C1E26;
    border-right: 0.5px solid rgba(255,255,255,0.08);
}

/* ── Tabs: quita el fondo gris default y estiliza ────────────────── */
[data-testid="stTabs"] button {
    font-size: 13px;
    color: #8892A4;
    border-bottom: 2px solid transparent;
    padding-bottom: 10px;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #4F8EF7;
    border-bottom: 2px solid #4F8EF7;
    font-weight: 500;
}
[data-testid="stTabs"] button:hover {
    color: #E8EAF0;
    background: transparent;
}

/* ── Metric cards: borde sutil ───────────────────────────────────── */
[data-testid="stMetric"] {
    background-color: #1C1E26;
    border: 0.5px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 1rem 1.25rem;
}
[data-testid="stMetricLabel"] {
    font-size: 12px !important;
    color: #8892A4 !important;
}
[data-testid="stMetricValue"] {
    font-size: 24px !important;
    color: #E8EAF0 !important;
}

/* ── Dataframe: cabecera más oscura ──────────────────────────────── */
[data-testid="stDataFrame"] thead tr th {
    background-color: #16181F !important;
    color: #8892A4 !important;
    font-size: 11px !important;
    font-weight: 500 !important;
}

/* ── Plotly charts: fondo transparente para que case con el tema ─── */
[data-testid="stPlotlyChart"] {
    border: 0.5px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 1rem;
    background-color: #1C1E26;
}

/* ── Botones primarios ───────────────────────────────────────────── */
[data-testid="stDownloadButton"] button,
[data-testid="stButton"] button[kind="primary"] {
    background-color: #4F8EF7 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
}

/* ── Info / success / warning banners ───────────────────────────── */
[data-testid="stAlert"] {
    border-radius: 8px;
    border-left-width: 3px;
}

/* ── Inputs y multiselect ────────────────────────────────────────── */
[data-testid="stMultiSelect"] > div,
[data-testid="stSelectbox"] > div {
    background-color: #16181F;
    border-color: rgba(255,255,255,0.1) !important;
    border-radius: 8px;
}

/* ── Divider más sutil ───────────────────────────────────────────── */
hr {
    border-color: rgba(255,255,255,0.06) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar: filtros globales ─────────────────────────────────────────────────
with st.sidebar:
    st.title("Configuration")
    st.divider()
    st.subheader("Filters")

    # Cargar opciones una sola vez
    @st.cache_data(ttl=300)
    def load_filter_options():
        return get_filter_options()

    opts = load_filter_options()

    selected_regions = st.multiselect(
        "Region",
        options=opts["regions"],
        default=[],
        placeholder="All regions",
    )
    selected_shops = st.multiselect(
        "Store",
        options=opts["shops"],
        default=[],
        placeholder="All stores",
    )
    selected_categories = st.multiselect(
        "Category",
        options=opts["categories"],
        default=[],
        placeholder="All categories",
    )

    st.divider()
    st.caption("Data updates every 5 min")

# Guardar filtros en session_state para Tab 3
st.session_state["filters"] = {
    "regions":    selected_regions    or None,
    "shops":      selected_shops      or None,
    "categories": selected_categories or None,
}

# ── Cargar datos filtrados (compartido entre Tab 1 y Tab 3) ───────────────────
@st.cache_data(ttl=300)
def load_predictions(regions, shops, categories):
    return get_predictions(
        region_ids=regions,
        shop_ids=shops,
        category_ids=categories,
    )

f = st.session_state["filters"]
df_pred = load_predictions(
    tuple(f["regions"])    if f["regions"]    else None,
    tuple(f["shops"])      if f["shops"]      else None,
    tuple(f["categories"]) if f["categories"] else None,
)

# ── Tabs ──────────────────────────────────────────────────────────────────────
st.title("1C Company — Forecast App")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Forecasts",
    "Model Evaluation",
    "Export Data",
    "Feedback",
    "System Health",
])


# TAB 1 - Forecasts
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.header("Demand Forecast")

    if df_pred.empty:
        st.warning("No data found for the selected filters.")
    else:
        # KPIs
        total_units  = int(df_pred["value"].sum())        # ← ahora suma value
        stores_count = df_pred["shop_id"].nunique()
        avg_mape     = df_eval["mape"].mean() if not df_eval.empty else 0

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Total Forecasted Units", f"{total_units:,}",  border=True)
        col_b.metric("Average MAPE",           f"{avg_mape:.1f}%",  border=True)
        col_c.metric("Stores Analyzed",        stores_count,        border=True)

        st.divider()

        # ── Time Series: histórico + forecast + intervalo de confianza ────────
        df_ts = (
            df_pred
            .groupby("date")
            .agg(
                value=("value", "sum"),
                value_lower=("value_lower", "sum"),
                value_upper=("value_upper", "sum"),
            )
            .reset_index()
        )

        fig_ts = go.Figure()

        # Banda de confianza (ribbon)
        fig_ts.add_trace(go.Scatter(
            x=pd.concat([df_ts["date"], df_ts["date"][::-1]]),
            y=pd.concat([df_ts["value_upper"], df_ts["value_lower"][::-1]]),
            fill="toself",
            fillcolor="rgba(79, 142, 247, 0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip",
            name="Confidence interval",
        ))

        # Línea de forecast
        fig_ts.add_trace(go.Scatter(
            x=df_ts["date"],
            y=df_ts["value"],
            mode="lines",
            line=dict(color="#4F8EF7", width=2),
            name="Forecast",
        ))

        fig_ts.update_layout(
            title="Forecast Timeline",
            xaxis_title="Date",
            yaxis_title="Units",
            template="plotly_white",
            paper_bgcolor="rgba(0,0,0,0)",
    		plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=-0.2),
            hovermode="x unified",
        )

        st.plotly_chart(fig_ts, use_container_width=True)

        # ── Bar chart: unidades por categoría ────────────────────────────────
        df_cat = (
            df_pred
            .groupby("category_id")["value"]
            .sum()
            .reset_index()
            .sort_values("value", ascending=False)
        )

        fig_bar = px.bar(
            df_cat,
            x="category_id",
            y="value",
            title="Forecasted Units by Category",
            labels={"category_id": "Category", "value": "Units"},
            template="plotly_white",
            color="value",
            color_continuous_scale="Blues",
        )
        fig_bar.update_layout(coloraxis_showscale=False)

        st.plotly_chart(fig_bar, use_container_width=True)

        st.subheader("Detailed Forecasts")
        st.dataframe(df_pred, use_container_width=True, hide_index=True)

# TAB 2 - Model Evaluation
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.header("Model Performance")

    df_eval = get_model_evaluation()

    if df_eval.empty:
        st.warning("No evaluation data available.")
    else:
        avg_rmse = df_eval["rmse"].mean()
        avg_mae  = df_eval["mae"].mean()

        col_d, col_e, col_f = st.columns(3)
        col_d.metric("Average MAPE", f"{df_eval['mape'].mean():.1f}%", border=True)
        col_e.metric("Average RMSE", f"{avg_rmse:.2f}",               border=True)
        col_f.metric("Average MAE",  f"{avg_mae:.2f}",                border=True)

        st.divider()
        st.subheader("Metrics by Category")
        st.dataframe(
            df_eval[[
                "category_id", "mape", "rmse", "mae",
                "bias", "samples", "model_id"
            ]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "mape":    st.column_config.NumberColumn("MAPE (%)", format="%.1f"),
                "rmse":    st.column_config.NumberColumn("RMSE",     format="%.2f"),
                "mae":     st.column_config.NumberColumn("MAE",      format="%.2f"),
                "bias":    st.column_config.NumberColumn("Bias",     format="%.2f"),
                "samples": st.column_config.NumberColumn("Samples"),
            },
        )

        # Nota de metadata
        if "last_retrain_date" in df_eval.columns:
            last_retrain = df_eval["last_retrain_date"].max()
            train_start  = df_eval["training_start_date"].min()
            train_end    = df_eval["training_end_date"].max()
            test_start   = df_eval["test_start_date"].min()
            test_end     = df_eval["test_end_date"].max()

            st.info(
                f"**Training period:** {train_start:%Y-%m-%d} → {train_end:%Y-%m-%d} · "
                f"**Test period:** {test_start:%Y-%m-%d} → {test_end:%Y-%m-%d} · "
                f"**Last re-train:** {last_retrain:%Y-%m-%d}"
            )


# TAB 3 - Export
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.header("Export Forecasts")

    active = [
        f"Regions: {', '.join(map(str, f['regions']))}"       if f["regions"]    else None,
        f"Stores: {', '.join(map(str, f['shops']))}"          if f["shops"]      else None,
        f"Categories: {', '.join(map(str, f['categories']))}" if f["categories"] else None,
    ]
    active = [x for x in active if x]

    if active:
        st.success("Active filters: " + " · ".join(active))
    else:
        st.info("No filters active — exporting all data.")

    if df_pred.empty:
        st.warning("No data to export.")
    else:
        st.metric("Rows to export", f"{len(df_pred):,}")
        st.divider()

        # ── Formato de exportación ────────────────────────────────────────────
        fmt = st.radio(
            "Export format",
            ["CSV (.csv)", "Parquet (.parquet)"],
            horizontal=True,
        )

        if fmt == "CSV (.csv)":
            data     = export_predictions(df_pred, format="csv")
            filename = "1c_forecasts.csv"
            mime     = "text/csv"
        else:
            data     = export_predictions(df_pred, format="parquet")
            filename = "1c_forecasts.parquet"
            mime     = "application/octet-stream"

        st.download_button(
            label=f"Download {filename}",
            data=data,
            file_name=filename,
            mime=mime,
        )

        st.divider()

        # ── Preview con Great Tables ──────────────────────────────────────────
        from great_tables import GT, style, loc

        st.subheader("Preview (first 50 rows)")

        df_preview = (
            df_pred
            .head(50)
            .assign(
                date=df_pred["date"].dt.strftime("%Y-%m-%d"),
                run_date=df_pred["run_date"].dt.strftime("%Y-%m-%d"),
                value=df_pred["value"].round(0).astype(int),
            )
        )

        gt_table = (
            GT(df_preview)
            .tab_header(
                title="Forecast Preview",
                subtitle=f"{len(df_pred):,} total rows · showing first 50",
            )
            .tab_spanner(
                label="Identifiers",
                columns=["item_id", "shop_id", "category_id", "region_id"],
            )
            .tab_spanner(
                label="Forecast",
                columns=["date", "value"],
            )
            .cols_label(
                item_id="Item",
                shop_id="Store",
                category_id="Category",
                region_id="Region",
                date="Forecast Date",
                run_date="Run Date",
                value="Units",
            )
            .cols_hide(columns=["id"])
            .fmt_integer(columns=["value"])
            .tab_style(
                style=style.fill(color="#EFF6FF"),
                loc=loc.body(columns=["value"]),
            )
            .tab_style(
                style=style.text(weight="bold"),
                loc=loc.column_labels(),
            )
            .tab_options(
                table_width="100%",
                heading_background_color="#1E3A5F",
                heading_title_font_color="white",
                heading_subtitle_font_color="#CBD5E1",
            )
        )

        st.components.v1.html(
            gt_table.as_raw_html(),
            height=600,
            scrolling=True,
        )

# TAB 4 - Feedback
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.header("Feedback")

    opts = load_filter_options()

    # Resumen de issues existentes
    df_fb = get_feedback_summary()
    if not df_fb.empty:
        col_g, col_h = st.columns(2)
        col_g.metric("Total Reports",    int(df_fb["total"].sum()))
        col_h.metric("Issue Categories", df_fb["issue_type"].nunique())
        st.dataframe(df_fb, use_container_width=True, hide_index=True)
        st.divider()

    # Formulario
    st.subheader("Submit a Report")
    with st.form("feedback_form"):
        col1, col2 = st.columns(2)

        item_id     = col1.selectbox("Item ID",    options=[""] + list(df_pred["item_id"].unique()) if not df_pred.empty else [""])
        category_id = col2.selectbox("Category",   options=[""] + opts["categories"])
        region_id   = col1.selectbox("Region",     options=[""] + opts["regions"])
        issue_type  = col2.selectbox("Issue Type", options=["Overforecast", "Underforecast", "Missing Data", "Wrong Category", "Other"])
        severity    = st.select_slider(
            "Severity",
            options=["Low", "Medium", "High", "Critical"],
            value="Medium",
        )
        description = st.text_area("Description", placeholder="Describe the issue...")

        submitted = st.form_submit_button("Submit Feedback", type="primary")

        if submitted:
            if not description:
                st.error("Please add a description before submitting.")
            else:
                ok = submit_feedback(
                    item_id=str(item_id),
                    issue_type=issue_type,
                    severity=severity,
                    region_id=str(region_id),
                    category_id=str(category_id),
                    description=description,
                )
                if ok:
                    st.success("Feedback submitted successfully.")
                else:
                    st.error("Error submitting feedback. Please try again.")

# TAB 5 - System Health
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.header("System Health")

    hours = st.slider("Time window (hours)", min_value=1, max_value=72, value=24, step=1)

    kpis    = get_api_health_metrics(hours=hours)
    df_sys  = get_system_events(hours=hours)

    col_g, col_h, col_i = st.columns(3)
    col_g.metric("API Availability",  f"{kpis['availability']}%",      border=True)
    col_h.metric("Avg Latency",       f"{kpis['avg_latency_ms']} ms",  border=True)
    col_i.metric("Errors",            kpis["error_count"],             border=True)

    st.divider()

    if not df_sys.empty:
        # Barplot de llamadas por hora
        df_bar = (
            df_sys
            .set_index("datetime")
            .resample("1h")["event_id"]
            .count()
            .reset_index()
            .rename(columns={"event_id": "events", "datetime": "hour"})
        )
        fig_bar = px.bar(
            df_bar,
            x="hour",
            y="events",
            title=f"Events per Hour (last {hours}h)",
            template="plotly_white",
            color_discrete_sequence=["#4F8EF7"],
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # Tabla de eventos recientes
        st.subheader("Recent Events")
        st.dataframe(
            df_sys,
            use_container_width=True,
            hide_index=True,
            column_config={
                "status": st.column_config.TextColumn("Status"),
                "duration_ms": st.column_config.NumberColumn("Latency (ms)", format="%d ms"),
            },
        )
    else:
        st.info(f"No events in the last {hours} hours.")
