

import re
import calendar
from io import BytesIO
from datetime import datetime, date

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# =============================================================================
# APP CONFIG
# =============================================================================

st.set_page_config(
    page_title="Flowline Replacement - Forecast Dashboard",
    page_icon="📈",
    layout="wide",
)

st.markdown(
    """
    <style>
    div[data-testid="stMetric"] {
        background-color: #f8f9fb;
        padding: 8px 10px;
        border-radius: 12px;
        border: 1px solid #e6e8ef;
        min-height: 82px;
    }

    div[data-testid="stMetricLabel"] {
        font-size: 0.70rem !important;
        color: #4b5563;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.08rem !important;
        font-weight: 700;
        color: #111827;
    }

    div[data-testid="stMetricDelta"] {
        font-size: 0.70rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

DEFAULT_SHEET = "prod_wi_forecast"

BASE_ID_COLS = [
    "Pattern_Name",
    "Factor",
    "Well",
    "CPS",
    "Manifold",
    "Reservoir",
    "AREA",
]

STATIC_META_COLS = [
    "Bo",
    "VRR",
    "AA",
    "Average of Res P",
    "Average of P Sat",
]

REQUIRED_COLS = ["Pattern_Name", "Well", "RATES"]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def clean_column_name(col):
    if isinstance(col, (pd.Timestamp, datetime, date)):
        return pd.Timestamp(col).normalize()
    return str(col).strip()


def safe_divide(numerator, denominator):
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce")

    return np.where(
        (denominator.notna()) & (denominator != 0),
        numerator / denominator,
        np.nan,
    )


def normalize_rate_name(x):
    """
    Robustly normalize RATES column.
    Expected examples:
    - Oil Rate, Mstb/d
    - Water Rate, Mstb/d
    - WI Rate, Mstb/d
    """
    s = str(x).strip().lower()
    s_clean = re.sub(r"[^a-z0-9]+", " ", s)
    s_clean = re.sub(r"\s+", " ", s_clean).strip()

    if "oil" in s_clean and "rate" in s_clean:
        return "Oil Rate"

    if "water" in s_clean and "rate" in s_clean:
        return "Water Rate"

    if (
        re.search(r"\bwi\b", s_clean)
        or "injection" in s_clean
        or re.search(r"\binj\b", s_clean)
    ) and "rate" in s_clean:
        return "WI Rate"

    return "Unknown"


def find_date_columns(columns, dayfirst=False):
    static_cols = set(BASE_ID_COLS + STATIC_META_COLS + ["RATES"])
    date_cols = []
    date_map = {}

    for col in columns:
        if col in static_cols:
            continue

        dt = pd.NaT

        if isinstance(col, (pd.Timestamp, datetime, date)):
            dt = pd.Timestamp(col).normalize()
        else:
            s = str(col).strip()

            if not re.search(r"\d", s):
                continue

            if not any(symbol in s for symbol in ["/", "-", "."]):
                continue

            dt = pd.to_datetime(s, errors="coerce", dayfirst=dayfirst)

        if pd.notna(dt):
            dt = pd.Timestamp(dt).normalize()

            if 2000 <= dt.year <= 2100:
                date_cols.append(col)
                date_map[col] = dt

    return date_cols, date_map


def unique_join(series, max_items=8):
    vals = (
        series.dropna()
        .astype(str)
        .str.strip()
        .replace("", np.nan)
        .dropna()
        .unique()
        .tolist()
    )

    vals = sorted(vals)

    if len(vals) > max_items:
        return ", ".join(vals[:max_items]) + f" +{len(vals) - max_items} more"

    return ", ".join(vals)


def sorted_unique(series):
    vals = (
        series.dropna()
        .astype(str)
        .str.strip()
        .replace("", np.nan)
        .dropna()
        .unique()
        .tolist()
    )

    return sorted(vals)


def period_days_map(dates):
    dates = sorted(pd.to_datetime(pd.Series(dates).dropna().unique()))
    out = {}

    for i, d in enumerate(dates):
        d = pd.Timestamp(d).normalize()

        if i < len(dates) - 1:
            next_d = pd.Timestamp(dates[i + 1]).normalize()
            days = (next_d - d).days
        else:
            days = calendar.monthrange(d.year, d.month)[1]

        if days <= 0:
            days = calendar.monthrange(d.year, d.month)[1]

        out[d] = days

    return out


def row_component_list(row):
    used = []

    if bool(row.get("Pressure_Component_Used", False)):
        used.append("Pressure")

    if bool(row.get("WC_Component_Used", False)):
        used.append("Water Cut")

    if bool(row.get("VRR_Component_Used", False)):
        used.append("VRR")

    if not used:
        return "None"

    return " + ".join(used)


# =============================================================================
# LOAD DATA
# =============================================================================

@st.cache_data(show_spinner=False)
def get_excel_sheet_names(file_bytes):
    """Return workbook sheet names from an uploaded Excel file."""
    with pd.ExcelFile(BytesIO(file_bytes), engine="openpyxl") as workbook:
        return workbook.sheet_names


@st.cache_data(show_spinner=False)
def load_forecast_excel(file_bytes, sheet_name, dayfirst=False):
    """Read and reshape a selected worksheet from uploaded Excel bytes."""
    raw = pd.read_excel(
        BytesIO(file_bytes),
        sheet_name=sheet_name,
        engine="openpyxl",
    )
    raw.columns = [clean_column_name(c) for c in raw.columns]

    missing = [c for c in REQUIRED_COLS if c not in raw.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    date_cols, date_map = find_date_columns(raw.columns, dayfirst=dayfirst)

    if not date_cols:
        raise ValueError(
            "No forecast date columns detected. Expected headers like 01/01/2027, 02/01/2027, etc."
        )

    id_cols = [c for c in BASE_ID_COLS if c in raw.columns]

    df = raw.copy()

    for col in id_cols:
        df[col] = (
            df[col]
            .fillna("Blank")
            .astype(str)
            .str.strip()
            .replace("", "Blank")
        )

    df["RATES"] = df["RATES"].fillna("").astype(str).str.strip()
    df["Rate_Type"] = df["RATES"].apply(normalize_rate_name)

    df = df[df["Rate_Type"] != "Unknown"].copy()

    for col in date_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    rename_map = {
        "VRR": "VRR_Input",
        "Average of Res P": "Average_Res_P",
        "Average of P Sat": "Average_P_Sat",
    }

    df = df.rename(columns=rename_map)

    meta_cols = [
        "Bo",
        "VRR_Input",
        "AA",
        "Average_Res_P",
        "Average_P_Sat",
    ]

    for col in meta_cols:
        if col not in df.columns:
            df[col] = np.nan

        df[col] = pd.to_numeric(df[col], errors="coerce")

    long_df = df.melt(
        id_vars=id_cols + ["RATES", "Rate_Type"] + meta_cols,
        value_vars=date_cols,
        var_name="_Date_Column",
        value_name="Rate_Mstbd",
    )

    long_df["Date"] = long_df["_Date_Column"].map(date_map)
    long_df["Rate_Mstbd"] = pd.to_numeric(
        long_df["Rate_Mstbd"],
        errors="coerce",
    ).fillna(0.0)

    long_df["Oil Rate"] = np.where(
        long_df["Rate_Type"] == "Oil Rate",
        long_df["Rate_Mstbd"],
        0.0,
    )

    long_df["Water Rate"] = np.where(
        long_df["Rate_Type"] == "Water Rate",
        long_df["Rate_Mstbd"],
        0.0,
    )

    long_df["WI Rate"] = np.where(
        long_df["Rate_Type"] == "WI Rate",
        long_df["Rate_Mstbd"],
        0.0,
    )

    rate_types_found = sorted(df["Rate_Type"].unique().tolist())

    pressure_cols_found = {
        "Average of Res P": "Average_Res_P" in long_df.columns and long_df["Average_Res_P"].notna().any(),
        "Average of P Sat": "Average_P_Sat" in long_df.columns and long_df["Average_P_Sat"].notna().any(),
    }

    return raw, long_df, date_cols, rate_types_found, pressure_cols_found


def add_metrics(df, default_bo=1.0):
    out = df.copy()

    out["Bo_Calc"] = pd.to_numeric(out["Bo"], errors="coerce")
    out["Bo_Calc"] = out["Bo_Calc"].replace(0, np.nan).fillna(default_bo)

    out["Rate_Mstbd"] = pd.to_numeric(out["Rate_Mstbd"], errors="coerce").fillna(0.0)

    out["Oil Rate"] = np.where(
        out["Rate_Type"] == "Oil Rate",
        out["Rate_Mstbd"],
        0.0,
    )

    out["Water Rate"] = np.where(
        out["Rate_Type"] == "Water Rate",
        out["Rate_Mstbd"],
        0.0,
    )

    out["WI Rate"] = np.where(
        out["Rate_Type"] == "WI Rate",
        out["Rate_Mstbd"],
        0.0,
    )

    out["Average_Res_P"] = pd.to_numeric(out["Average_Res_P"], errors="coerce")
    out["Average_P_Sat"] = pd.to_numeric(out["Average_P_Sat"], errors="coerce")

    return out


def aggregate_time(df, group_cols=None):
    if group_cols is None:
        group_cols = []

    group_cols = list(group_cols)

    agg = (
        df
        .groupby(group_cols + ["Date"], dropna=False)
        .agg(
            Oil_Rate_Mstbd=("Oil Rate", "sum"),
            Water_Rate_Mstbd=("Water Rate", "sum"),
            WI_Rate_Mstbd=("WI Rate", "sum"),
            Average_Res_P=("Average_Res_P", "median"),
            Average_P_Sat=("Average_P_Sat", "median"),
            Bo_Calc=("Bo_Calc", "median"),
        )
        .reset_index()
    )

    agg["Bo_Calc"] = agg["Bo_Calc"].replace(0, np.nan).fillna(1.0)

    agg["Produced_Liquid_Mstbd"] = (
        agg["Oil_Rate_Mstbd"]
        + agg["Water_Rate_Mstbd"]
    )

    agg["Produced_Voidage_Mstbd"] = (
        agg["Oil_Rate_Mstbd"] * agg["Bo_Calc"]
        + agg["Water_Rate_Mstbd"]
    )

    agg["Water_Cut"] = safe_divide(
        agg["Water_Rate_Mstbd"],
        agg["Produced_Liquid_Mstbd"],
    )

    agg["Oil_Cut"] = safe_divide(
        agg["Oil_Rate_Mstbd"],
        agg["Produced_Liquid_Mstbd"],
    )

    agg["VRR"] = safe_divide(
        agg["WI_Rate_Mstbd"],
        agg["Produced_Voidage_Mstbd"],
    )

    agg["Pressure_Delta"] = agg["Average_Res_P"] - agg["Average_P_Sat"]

    agg["Pressure_Margin_Fraction"] = safe_divide(
        agg["Pressure_Delta"],
        agg["Average_P_Sat"],
    )

    days_lookup = period_days_map(agg["Date"])
    agg["Days"] = agg["Date"].map(days_lookup).fillna(30)

    agg["Oil_Volume_Mstb"] = agg["Oil_Rate_Mstbd"] * agg["Days"]
    agg["Water_Volume_Mstb"] = agg["Water_Rate_Mstbd"] * agg["Days"]
    agg["WI_Volume_Mstb"] = agg["WI_Rate_Mstbd"] * agg["Days"]

    agg["Produced_Voidage_Volume_Mstb"] = (
        agg["Produced_Voidage_Mstbd"] * agg["Days"]
    )

    return agg


def build_pattern_summary(
    df,
    target_vrr=1.0,
    vrr_tolerance=0.4,
    wc_healthy_limit=0.30,
    wc_unhealthy_limit=0.70,
    pressure_full_score_margin_psi=300.0,
    pressure_weight=40,
    wc_weight=30,
    vrr_weight=30,
    value_driver="Cumulative oil volume",
):
    pattern_time = aggregate_time(df, group_cols=["Pattern_Name"])

    summary = (
        pattern_time
        .groupby("Pattern_Name", dropna=False)
        .agg(
            Avg_Oil_Mstbd=("Oil_Rate_Mstbd", "mean"),
            Avg_Water_Mstbd=("Water_Rate_Mstbd", "mean"),
            Avg_WI_Mstbd=("WI_Rate_Mstbd", "mean"),
            Cum_Oil_Mstb=("Oil_Volume_Mstb", "sum"),
            Cum_Water_Mstb=("Water_Volume_Mstb", "sum"),
            Cum_WI_Mstb=("WI_Volume_Mstb", "sum"),
            Cum_Produced_Voidage_Mstb=("Produced_Voidage_Volume_Mstb", "sum"),
            Avg_Monthly_VRR=("VRR", "mean"),
            Min_VRR=("VRR", "min"),
            Max_VRR=("VRR", "max"),
            Avg_Water_Cut=("Water_Cut", "mean"),
            Max_Water_Cut=("Water_Cut", "max"),
            Avg_Oil_Cut=("Oil_Cut", "mean"),
            Average_Res_P=("Average_Res_P", "median"),
            Average_P_Sat=("Average_P_Sat", "median"),
        )
        .reset_index()
    )

    summary["Avg_VRR"] = safe_divide(
        summary["Cum_WI_Mstb"],
        summary["Cum_Produced_Voidage_Mstb"],
    )

    has_production = summary["Cum_Produced_Voidage_Mstb"].fillna(0) > 0
    has_injection = summary["Cum_WI_Mstb"].fillna(0) > 0

    summary["Calc_Status"] = np.select(
        [
            has_production & has_injection,
            ~has_production & has_injection,
            has_production & ~has_injection,
            ~has_production & ~has_injection,
        ],
        [
            "Active Prod + Inj",
            "Injection only / no production",
            "Production only / no injection",
            "No production / no injection",
        ],
        default="Check data",
    )

    summary.loc[~has_production, ["Avg_VRR", "Min_VRR", "Max_VRR"]] = np.nan

    # meta = (
    #     df
    #     .groupby("Pattern_Name", dropna=False)
    #     .agg(
    #         Wells=("Well", "nunique"),
    #         Well_List=("Well", unique_join),
    #         CPS=("CPS", unique_join),
    #         Manifold=("Manifold", unique_join),
    #         Reservoir=("Reservoir", unique_join),
    #         AREA=("AREA", unique_join),
    #         Input_VRR=("VRR_Input", "median"),
    #         AA=("AA", "median"),
    #         Rate_Types=("Rate_Type", unique_join),
    #     )
    #     .reset_index()
    # )
    
    meta_source = df.copy()

    meta_source["Producer_Well"] = np.where(
        meta_source["Rate_Type"].isin(["Oil Rate", "Water Rate"]),
        meta_source["Well"],
        np.nan,
    )
    
    meta_source["Water_Injector_Well"] = np.where(
        meta_source["Rate_Type"] == "WI Rate",
        meta_source["Well"],
        np.nan,
    )
    
    meta = (
        meta_source
        .groupby("Pattern_Name", dropna=False)
        .agg(
            Wells=("Well", "nunique"),
            Producer_Count=("Producer_Well", "nunique"),
            Water_Injector_Count=("Water_Injector_Well", "nunique"),
            Producers=("Producer_Well", unique_join),
            Water_Injectors=("Water_Injector_Well", unique_join),
            Well_List=("Well", unique_join),
            CPS=("CPS", unique_join),
            Manifold=("Manifold", unique_join),
            Reservoir=("Reservoir", unique_join),
            AREA=("AREA", unique_join),
            Input_VRR=("VRR_Input", "median"),
            AA=("AA", "median"),
            Rate_Types=("Rate_Type", unique_join),
        )
        .reset_index()
    )

    summary = summary.merge(meta, on="Pattern_Name", how="left")
    
    
    

    # =========================================================================
    # 1. PRESSURE SCORE
    # =========================================================================

    pressure_available = (
        summary["Average_Res_P"].notna()
        & summary["Average_P_Sat"].notna()
        & (summary["Average_Res_P"] > 0)
        & (summary["Average_P_Sat"] > 0)
    )

    summary["Pressure_Delta"] = summary["Average_Res_P"] - summary["Average_P_Sat"]

    summary["Pressure_Margin_Fraction"] = safe_divide(
        summary["Pressure_Delta"],
        summary["Average_P_Sat"],
    )

    full_margin = max(float(pressure_full_score_margin_psi), 1.0)

    summary["Pressure_Score"] = summary["Pressure_Delta"] / full_margin
    summary["Pressure_Score"] = summary["Pressure_Score"].clip(lower=0, upper=1)
    summary.loc[~pressure_available, "Pressure_Score"] = np.nan

    summary["Pressure_Above_Psat"] = np.select(
        [
            ~pressure_available,
            summary["Pressure_Delta"] > 0,
        ],
        [
            "Missing",
            "Yes",
        ],
        default="No",
    )

    summary["Pressure_Component_Used"] = pressure_available

    # =========================================================================
    # 2. WATER CUT SCORE
    # =========================================================================

    wc_available = summary["Avg_Water_Cut"].notna()

    wc_good = min(float(wc_healthy_limit), float(wc_unhealthy_limit))
    wc_bad = max(float(wc_healthy_limit), float(wc_unhealthy_limit))
    wc_range = max(wc_bad - wc_good, 0.001)

    summary["WC_Score"] = (wc_bad - summary["Avg_Water_Cut"]) / wc_range
    summary["WC_Score"] = summary["WC_Score"].clip(lower=0, upper=1)
    summary.loc[~wc_available, "WC_Score"] = np.nan

    summary["WC_Component_Used"] = wc_available

    # =========================================================================
    # 3. VRR SCORE
    # =========================================================================

    tol = max(float(vrr_tolerance), 0.001)

    vrr_available = has_production & has_injection & summary["Avg_VRR"].notna()

    vrr_error = (summary["Avg_VRR"] - target_vrr).abs() / tol

    summary["VRR_Score"] = 1.0 / (1.0 + vrr_error ** 2)
    summary["VRR_Score"] = summary["VRR_Score"].clip(lower=0, upper=1)
    summary.loc[~vrr_available, "VRR_Score"] = np.nan

    summary["VRR_Component_Used"] = vrr_available

    # =========================================================================
    # FINAL SCORE
    # =========================================================================

    summary["_Score_Numerator"] = 0.0
    summary["_Score_Denominator"] = 0.0

    if pressure_weight > 0:
        mask = summary["Pressure_Component_Used"]
        summary.loc[mask, "_Score_Numerator"] += (
            summary.loc[mask, "Pressure_Score"] * pressure_weight
        )
        summary.loc[mask, "_Score_Denominator"] += pressure_weight

    if wc_weight > 0:
        mask = summary["WC_Component_Used"]
        summary.loc[mask, "_Score_Numerator"] += (
            summary.loc[mask, "WC_Score"] * wc_weight
        )
        summary.loc[mask, "_Score_Denominator"] += wc_weight

    if vrr_weight > 0:
        mask = summary["VRR_Component_Used"]
        summary.loc[mask, "_Score_Numerator"] += (
            summary.loc[mask, "VRR_Score"] * vrr_weight
        )
        summary.loc[mask, "_Score_Denominator"] += vrr_weight

    summary["Final_Score"] = safe_divide(
        summary["_Score_Numerator"],
        summary["_Score_Denominator"],
    )

    summary["Final_Score"] = pd.to_numeric(
        summary["Final_Score"],
        errors="coerce",
    ).fillna(0.0)

    summary["Final_Score"] = summary["Final_Score"].clip(lower=0, upper=1)

    summary["Score_Components_Used"] = summary.apply(row_component_list, axis=1)

    summary.loc[
        summary["Calc_Status"] == "No production / no injection",
        "Final_Score",
    ] = 0

    if value_driver == "Cumulative oil volume":
        driver = summary["Cum_Oil_Mstb"].fillna(0)

    elif value_driver == "Average injection rate":
        driver = summary["Avg_WI_Mstbd"].fillna(0)

    elif value_driver == "Average liquid rate":
        driver = (
            summary["Avg_Oil_Mstbd"]
            + summary["Avg_Water_Mstbd"]
        ).fillna(0)

    else:
        driver = pd.Series(1.0, index=summary.index)

    summary["Value_Index"] = summary["Final_Score"] * driver

    summary = summary.sort_values("Value_Index", ascending=False).reset_index(drop=True)

    summary["Rank"] = np.arange(1, len(summary) + 1)
    summary["Cumulative_Value_Index"] = summary["Value_Index"].cumsum()

    total_value = summary["Value_Index"].sum()

    if total_value > 0:
        summary["Cumulative_Value_Percent"] = (
            summary["Cumulative_Value_Index"] / total_value
        )
    else:
        summary["Cumulative_Value_Percent"] = np.nan

    summary["Health_Status"] = np.select(
        [
            summary["Final_Score"] >= 0.75,
            summary["Final_Score"] >= 0.50,
            summary["Final_Score"] >= 0.25,
        ],
        [
            "Healthy",
            "Watch",
            "Risk",
        ],
        default="Unhealthy",
    )

    summary.loc[
        summary["Calc_Status"] == "Injection only / no production",
        "Health_Status",
    ] = "Injection Only"

    summary.loc[
        summary["Calc_Status"] == "Production only / no injection",
        "Health_Status",
    ] = "No Injection Support"

    summary.loc[
        summary["Calc_Status"] == "No production / no injection",
        "Health_Status",
    ] = "No Activity"

    summary = summary.drop(
        columns=["_Score_Numerator", "_Score_Denominator"],
        errors="ignore",
    )

    return summary, pattern_time

def weighted_average(values, weights):
    values = pd.to_numeric(values, errors="coerce")
    weights = pd.to_numeric(weights, errors="coerce").fillna(0)

    valid = values.notna() & weights.notna() & (weights > 0)

    if valid.sum() == 0:
        return np.nan

    return np.average(values[valid], weights=weights[valid])


def build_injector_summary(df, pattern_summary):
    """
    Build unique injector ranking table from pattern-level values.

    Logic:
    - Find injector wells from Rate_Type == 'WI Rate'
    - For each Pattern_Name, divide pattern injection equally among all injectors
    - Keep pressure and score components from the parent pattern
    - If one injector appears in multiple patterns, aggregate into one unique injector row
    """

    injector_rows = df[df["Rate_Type"] == "WI Rate"].copy()

    if injector_rows.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Unique injector wells per pattern.
    pattern_injectors = (
        injector_rows
        .groupby(["Pattern_Name", "Well"], dropna=False)
        .agg(
            Injector_CPS=("CPS", unique_join),
            Injector_Manifold=("Manifold", unique_join),
            Injector_Reservoir=("Reservoir", unique_join),
            Injector_AREA=("AREA", unique_join),
        )
        .reset_index()
        .rename(columns={"Well": "Water_Injector"})
    )

    injector_count = (
        pattern_injectors
        .groupby("Pattern_Name", dropna=False)
        .agg(
            Injectors_In_Pattern=("Water_Injector", "nunique"),
            Injectors_In_Pattern_List=("Water_Injector", unique_join),
        )
        .reset_index()
    )

    pattern_injectors = pattern_injectors.merge(
        injector_count,
        on="Pattern_Name",
        how="left",
    )

    pattern_cols_to_keep = [
        "Pattern_Name",
        "Rank",
        "Health_Status",
        "Calc_Status",
        "Score_Components_Used",
        "Rate_Types",
        "Final_Score",
        "Pressure_Score",
        "WC_Score",
        "VRR_Score",
        "Average_Res_P",
        "Average_P_Sat",
        "Pressure_Delta",
        "Pressure_Margin_Fraction",
        "Avg_VRR",
        "Avg_Water_Cut",
        "Avg_Oil_Mstbd",
        "Avg_Water_Mstbd",
        "Avg_WI_Mstbd",
        "Cum_Oil_Mstb",
        "Cum_Water_Mstb",
        "Cum_WI_Mstb",
        "Value_Index",
        "Cumulative_Value_Percent",
        "Producer_Count",
        "Water_Injector_Count",
        "Producers",
        "Water_Injectors",
        "CPS",
        "Manifold",
        "Reservoir",
        "AREA",
    ]

    pattern_cols_to_keep = [
        c for c in pattern_cols_to_keep
        if c in pattern_summary.columns
    ]

    injector_detail = pattern_injectors.merge(
        pattern_summary[pattern_cols_to_keep],
        on="Pattern_Name",
        how="left",
        suffixes=("", "_Pattern"),
    )

    # Safety: ensure expected text columns exist even if something was missing upstream.
    for col in [
        "Water_Injectors",
        "Producers",
        "Health_Status",
        "Calc_Status",
        "Score_Components_Used",
        "Reservoir",
        "AREA",
        "CPS",
        "Manifold",
    ]:
        if col not in injector_detail.columns:
            injector_detail[col] = ""

    injector_detail["Injectors_In_Pattern"] = pd.to_numeric(
        injector_detail["Injectors_In_Pattern"],
        errors="coerce",
    ).replace(0, np.nan)

    # Divide pattern values equally among injectors in the same pattern.
    injector_detail["Allocated_Avg_WI_Mstbd"] = (
        injector_detail["Avg_WI_Mstbd"]
        / injector_detail["Injectors_In_Pattern"]
    )

    injector_detail["Allocated_Cum_WI_Mstb"] = (
        injector_detail["Cum_WI_Mstb"]
        / injector_detail["Injectors_In_Pattern"]
    )

    injector_detail["Allocated_Value_Index"] = (
        injector_detail["Value_Index"]
        / injector_detail["Injectors_In_Pattern"]
    )

    injector_detail["Allocated_Avg_Oil_Mstbd"] = (
        injector_detail["Avg_Oil_Mstbd"]
        / injector_detail["Injectors_In_Pattern"]
    )

    injector_detail["Allocated_Cum_Oil_Mstb"] = (
        injector_detail["Cum_Oil_Mstb"]
        / injector_detail["Injectors_In_Pattern"]
    )

    rows = []

    for injector, g in injector_detail.groupby("Water_Injector", dropna=False):
        weights = g["Allocated_Cum_WI_Mstb"].fillna(0)

        if weights.sum() <= 0:
            weights = g["Allocated_Avg_WI_Mstbd"].fillna(0)

        rows.append(
            {
                "Water_Injector": injector,
                "Injector_Pattern_Count": g["Pattern_Name"].nunique(),
                "Patterns": unique_join(g["Pattern_Name"]),
                "Injector_CPS": unique_join(g["Injector_CPS"]),
                "Injector_Manifold": unique_join(g["Injector_Manifold"]),
                "Injector_Reservoir": unique_join(g["Injector_Reservoir"]),
                "Injector_AREA": unique_join(g["Injector_AREA"]),
                "Pattern_Reservoir": unique_join(g["Reservoir"]),
                "Pattern_AREA": unique_join(g["AREA"]),
                "Parent_Health_Status": unique_join(g["Health_Status"]),
                "Parent_Calc_Status": unique_join(g["Calc_Status"]),
                "Score_Components_Used": unique_join(g["Score_Components_Used"]),
                "Pattern_Producers": unique_join(g["Producers"]),
                "Pattern_All_Injectors": unique_join(g["Water_Injectors"]),
                "Injectors_In_Pattern_List": unique_join(g["Injectors_In_Pattern_List"]),
                "Allocated_Avg_WI_Mstbd": g["Allocated_Avg_WI_Mstbd"].sum(),
                "Allocated_Cum_WI_Mstb": g["Allocated_Cum_WI_Mstb"].sum(),
                "Allocated_Avg_Oil_Mstbd": g["Allocated_Avg_Oil_Mstbd"].sum(),
                "Allocated_Cum_Oil_Mstb": g["Allocated_Cum_Oil_Mstb"].sum(),
                "Allocated_Value_Index": g["Allocated_Value_Index"].sum(),
                "Injector_Final_Score": weighted_average(g["Final_Score"], weights),
                "Injector_Pressure_Score": weighted_average(g["Pressure_Score"], weights),
                "Injector_WC_Score": weighted_average(g["WC_Score"], weights),
                "Injector_VRR_Score": weighted_average(g["VRR_Score"], weights),
                "Average_Res_P": weighted_average(g["Average_Res_P"], weights),
                "Average_P_Sat": weighted_average(g["Average_P_Sat"], weights),
                "Pressure_Delta": weighted_average(g["Pressure_Delta"], weights),
                "Pressure_Margin_Fraction": weighted_average(g["Pressure_Margin_Fraction"], weights),
                "Avg_VRR": weighted_average(g["Avg_VRR"], weights),
                "Avg_Water_Cut": weighted_average(g["Avg_Water_Cut"], weights),
            }
        )

    injector_summary = pd.DataFrame(rows)

    if injector_summary.empty:
        return injector_summary, injector_detail

    injector_summary["Injector_Final_Score"] = pd.to_numeric(
        injector_summary["Injector_Final_Score"],
        errors="coerce",
    ).fillna(0).clip(lower=0, upper=1)

    injector_summary["Injector_Health_Status"] = np.select(
        [
            injector_summary["Injector_Final_Score"] >= 0.75,
            injector_summary["Injector_Final_Score"] >= 0.50,
            injector_summary["Injector_Final_Score"] >= 0.25,
        ],
        [
            "Healthy",
            "Watch",
            "Risk",
        ],
        default="Unhealthy",
    )

    injector_summary = injector_summary.sort_values(
        "Allocated_Value_Index",
        ascending=False,
    ).reset_index(drop=True)

    injector_summary["Injector_Rank"] = np.arange(1, len(injector_summary) + 1)

    first_cols = [
        "Injector_Rank",
        "Water_Injector",
        "Injector_Health_Status",
        "Injector_Final_Score",
        "Allocated_Value_Index",
    ]

    other_cols = [c for c in injector_summary.columns if c not in first_cols]

    injector_summary = injector_summary[first_cols + other_cols]

    return injector_summary, injector_detail
# =============================================================================
# SIDEBAR INPUTS
# =============================================================================

st.sidebar.title("⚙️ Controls")

uploaded_file = st.sidebar.file_uploader(
    "Browse forecast input file",
    type=["xlsx", "xlsm"],
    accept_multiple_files=False,
    help="Upload the Excel workbook that contains the production and injection forecast.",
)

if uploaded_file is None:
    st.title("📈 Flowline Replacement - Production Forecast & Pattern Health Ranking")
    st.info("Upload an Excel forecast file from the sidebar to start the dashboard.")
    st.stop()

file_bytes = uploaded_file.getvalue()

try:
    sheet_names = get_excel_sheet_names(file_bytes)
except Exception as e:
    st.error(f"Could not open the uploaded Excel workbook: {e}")
    st.stop()

if not sheet_names:
    st.error("The uploaded Excel workbook does not contain any worksheets.")
    st.stop()

default_sheet_index = (
    sheet_names.index(DEFAULT_SHEET)
    if DEFAULT_SHEET in sheet_names
    else 0
)

sheet_name = st.sidebar.selectbox(
    "Worksheet",
    options=sheet_names,
    index=default_sheet_index,
    help="The expected default worksheet is prod_wi_forecast when it is available.",
)

date_format_choice = st.sidebar.radio(
    "Date header format",
    options=[
        "Month/Day/Year e.g. 02/01/2027 = Feb-2027",
        "Day/Month/Year e.g. 02/01/2027 = 2-Jan-2027",
    ],
    index=0,
)

dayfirst = date_format_choice.startswith("Day")


# =============================================================================
# LOAD EXCEL
# =============================================================================

st.title("📈 Flowline Replacement - Production Forecast & Pattern Health Ranking")

try:
    raw_df, model_df, forecast_date_cols, rate_types_found, pressure_cols_found = load_forecast_excel(
        file_bytes=file_bytes,
        sheet_name=sheet_name,
        dayfirst=dayfirst,
    )
except Exception as e:
    st.error(f"Could not load worksheet '{sheet_name}': {e}")
    st.stop()

bo_median = pd.to_numeric(model_df["Bo"], errors="coerce").replace(0, np.nan).median()

if pd.isna(bo_median):
    bo_median = 1.0

vrr_input_median = pd.to_numeric(model_df["VRR_Input"], errors="coerce").replace(0, np.nan).median()

if pd.isna(vrr_input_median):
    vrr_input_median = 1.0

default_bo = st.sidebar.number_input(
    "Default Bo if blank",
    min_value=0.1,
    max_value=5.0,
    value=float(round(bo_median, 3)),
    step=0.01,
)

metric_df = add_metrics(model_df, default_bo=default_bo)


# =============================================================================
# FILTERS
# =============================================================================

st.sidebar.divider()
st.sidebar.subheader("🔎 Filters")

min_date = pd.to_datetime(metric_df["Date"]).min().date()
max_date = pd.to_datetime(metric_df["Date"]).max().date()

date_range = st.sidebar.date_input(
    "Forecast date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

date_filtered_df = metric_df.copy()

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date = pd.Timestamp(date_range[0])
    end_date = pd.Timestamp(date_range[1])

    date_filtered_df = date_filtered_df[
        (date_filtered_df["Date"] >= start_date)
        & (date_filtered_df["Date"] <= end_date)
    ].copy()

pattern_level_filter = st.sidebar.checkbox(
    "Pattern-level filters: include all producer + injector wells in matched patterns",
    value=True,
    help=(
        "Recommended ON. If any row in a pattern matches the selected CPS/Manifold/Well, "
        "the dashboard keeps the full Pattern_Name including all producers and injectors."
    ),
)

filter_cols = [
    ("Reservoir", "Reservoir"),
    ("AREA", "Area"),
    ("Pattern_Name", "Pattern"),
    ("Well", "Well"),
    ("CPS", "CPS"),
    ("Manifold", "Manifold"),
]

if pattern_level_filter:
    selected_patterns = set(
        date_filtered_df["Pattern_Name"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    for col, label in filter_cols:
        if col in date_filtered_df.columns:
            option_base = date_filtered_df[
                date_filtered_df["Pattern_Name"].astype(str).isin(selected_patterns)
            ]

            options = sorted_unique(option_base[col])

            selected = st.sidebar.multiselect(
                label,
                options=options,
                default=[],
                key=f"pattern_level_filter_{col}",
            )

            if selected:
                matched_patterns = set(
                    date_filtered_df.loc[
                        date_filtered_df[col].astype(str).isin(selected),
                        "Pattern_Name",
                    ]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )

                selected_patterns = selected_patterns.intersection(matched_patterns)

    filtered_df = date_filtered_df[
        date_filtered_df["Pattern_Name"].astype(str).isin(selected_patterns)
    ].copy()

else:
    filtered_df = date_filtered_df.copy()

    for col, label in filter_cols:
        if col in filtered_df.columns:
            options = sorted_unique(filtered_df[col])

            selected = st.sidebar.multiselect(
                label,
                options=options,
                default=[],
                key=f"row_level_filter_{col}",
            )

            if selected:
                filtered_df = filtered_df[
                    filtered_df[col].astype(str).isin(selected)
                ].copy()

st.sidebar.divider()
st.sidebar.subheader("🚫 Exclude Patterns")

exclude_patterns = st.sidebar.multiselect(
    "Deselect / exclude pattern(s)",
    options=sorted_unique(filtered_df["Pattern_Name"]),
    default=[],
)

if exclude_patterns:
    filtered_df = filtered_df[
        ~filtered_df["Pattern_Name"].astype(str).isin(exclude_patterns)
    ].copy()

if filtered_df.empty:
    st.warning("No data after filters or excluded patterns. Please relax the selection.")
    st.stop()


# =============================================================================
# SCORING CONTROLS
# =============================================================================

st.sidebar.divider()
st.sidebar.subheader("🏆 Pattern scoring")

target_vrr = st.sidebar.number_input(
    "Target VRR",
    min_value=0.0,
    max_value=5.0,
    value=float(round(vrr_input_median, 2)),
    step=0.05,
)

vrr_tolerance = st.sidebar.number_input(
    "VRR tolerance",
    min_value=0.05,
    max_value=3.0,
    value=0.40,
    step=0.05,
    help="VRR score reduces smoothly as VRR moves away from target.",
)

wc_healthy_limit = st.sidebar.number_input(
    "Healthy WC limit",
    min_value=0.0,
    max_value=1.0,
    value=0.30,
    step=0.05,
    help="Water cut at or below this value gets full WC score.",
)

wc_unhealthy_limit = st.sidebar.number_input(
    "Unhealthy WC limit",
    min_value=0.0,
    max_value=1.0,
    value=0.70,
    step=0.05,
    help="Water cut at or above this value gets zero WC score.",
)

pressure_full_score_margin_psi = st.sidebar.number_input(
    "Pressure full-score margin above Psat, psi",
    min_value=1.0,
    max_value=2000.0,
    value=300.0,
    step=50.0,
    help="If Res P is Psat + this margin or higher, pressure score becomes 1.",
)

st.sidebar.caption("Final Score = weighted average of available components only.")

pressure_weight = st.sidebar.slider(
    "Pressure weight",
    min_value=0,
    max_value=100,
    value=40,
    step=5,
)

wc_weight = st.sidebar.slider(
    "Water Cut weight",
    min_value=0,
    max_value=100,
    value=30,
    step=5,
)

vrr_weight = st.sidebar.slider(
    "VRR weight",
    min_value=0,
    max_value=100,
    value=30,
    step=5,
)

value_driver = st.sidebar.selectbox(
    "Creaming curve value driver",
    options=[
        "Cumulative oil volume",
        "Average injection rate",
        "Average liquid rate",
        "Score only",
    ],
    index=0,
)


# =============================================================================
# AGGREGATIONS
# =============================================================================

time_df = aggregate_time(filtered_df)

pattern_summary, pattern_time_df = build_pattern_summary(
    filtered_df,
    target_vrr=target_vrr,
    vrr_tolerance=vrr_tolerance,
    wc_healthy_limit=wc_healthy_limit,
    wc_unhealthy_limit=wc_unhealthy_limit,
    pressure_full_score_margin_psi=pressure_full_score_margin_psi,
    pressure_weight=pressure_weight,
    wc_weight=wc_weight,
    vrr_weight=vrr_weight,
    value_driver=value_driver,
)
injector_summary, injector_pattern_detail = build_injector_summary(
    filtered_df,
    pattern_summary,
)
latest_date = time_df["Date"].max()
latest = time_df[time_df["Date"] == latest_date].iloc[0]

avg_oil = time_df["Oil_Rate_Mstbd"].mean()
avg_water = time_df["Water_Rate_Mstbd"].mean()
avg_wi = time_df["WI_Rate_Mstbd"].mean()
avg_vrr = time_df["VRR"].mean()
avg_wc = time_df["Water_Cut"].mean()
avg_pressure_delta = pattern_summary["Pressure_Delta"].mean()

n_patterns = filtered_df["Pattern_Name"].nunique()
n_wells = filtered_df["Well"].nunique()


# =============================================================================
# KPI CARDS
# =============================================================================

st.caption(f"Loaded worksheet `{sheet_name}` from `{uploaded_file.name}`")

kpi_cols = st.columns(9)

kpi_cols[0].metric("Patterns", f"{n_patterns:,}")
kpi_cols[1].metric("Wells", f"{n_wells:,}")
kpi_cols[2].metric("Avg Oil", f"{avg_oil:,.1f}")
kpi_cols[3].metric("Avg Water", f"{avg_water:,.1f}")
kpi_cols[4].metric("Avg WI", f"{avg_wi:,.1f}")
kpi_cols[5].metric("Avg VRR", f"{avg_vrr:,.2f}" if pd.notna(avg_vrr) else "N/A")
kpi_cols[6].metric("Avg WC", f"{avg_wc:.1%}" if pd.notna(avg_wc) else "N/A")
kpi_cols[7].metric("Avg P-Psat", f"{avg_pressure_delta:,.0f}" if pd.notna(avg_pressure_delta) else "N/A")
kpi_cols[8].metric("Latest VRR", f"{latest['VRR']:,.2f}" if pd.notna(latest["VRR"]) else "N/A")


# =============================================================================
# TABS
# =============================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📊 Forecast plots",
        "🏆 Pattern ranking",
        "🔥 Pattern heat map",
        "📈 Creaming curve",
        "🧾 Data table",
    ]
)


# =============================================================================
# TAB 1 - FORECAST PLOTS
# =============================================================================

with tab1:
    st.subheader("Production and Injection Forecast")

    plot_df = time_df.melt(
        id_vars=["Date"],
        value_vars=["Oil_Rate_Mstbd", "Water_Rate_Mstbd", "WI_Rate_Mstbd"],
        var_name="Stream",
        value_name="Rate_Mstbd",
    )

    stream_name_map = {
        "Oil_Rate_Mstbd": "Oil Rate",
        "Water_Rate_Mstbd": "Water Rate",
        "WI_Rate_Mstbd": "WI Rate",
    }

    plot_df["Stream"] = plot_df["Stream"].map(stream_name_map)

    fig_area = px.area(
        plot_df,
        x="Date",
        y="Rate_Mstbd",
        color="Stream",
        markers=True,
        title="Oil, Water and Injection Forecast",
        labels={
            "Rate_Mstbd": "Rate, Mstb/d",
            "Date": "Date",
        },
    )

    fig_area.update_layout(
        height=520,
        legend_title_text="Stream",
        hovermode="x unified",
    )

    st.plotly_chart(fig_area, use_container_width=True)

    st.subheader("VRR, Water Cut and Oil Cut")

    fig_vrr_wc = make_subplots(specs=[[{"secondary_y": True}]])

    fig_vrr_wc.add_trace(
        go.Scatter(
            x=time_df["Date"],
            y=time_df["VRR"],
            name="VRR",
            mode="lines+markers",
        ),
        secondary_y=False,
    )

    fig_vrr_wc.add_trace(
        go.Scatter(
            x=time_df["Date"],
            y=time_df["Water_Cut"],
            name="Water Cut",
            mode="lines+markers",
        ),
        secondary_y=True,
    )

    fig_vrr_wc.add_trace(
        go.Scatter(
            x=time_df["Date"],
            y=time_df["Oil_Cut"],
            name="Oil Cut",
            mode="lines+markers",
        ),
        secondary_y=True,
    )

    fig_vrr_wc.add_hline(
        y=target_vrr,
        line_dash="dash",
        annotation_text=f"Target VRR = {target_vrr:.2f}",
        secondary_y=False,
    )

    fig_vrr_wc.update_yaxes(title_text="VRR", secondary_y=False)
    fig_vrr_wc.update_yaxes(title_text="Cut fraction", tickformat=".0%", secondary_y=True)

    fig_vrr_wc.update_layout(
        title="VRR vs Water Cut / Oil Cut",
        height=520,
        hovermode="x unified",
    )

    st.plotly_chart(fig_vrr_wc, use_container_width=True)

    st.subheader("Pressure vs Psat")

    if time_df["Average_Res_P"].notna().any() and time_df["Average_P_Sat"].notna().any():
        fig_pressure = go.Figure()

        fig_pressure.add_trace(
            go.Scatter(
                x=time_df["Date"],
                y=time_df["Average_Res_P"],
                mode="lines+markers",
                name="Average Reservoir Pressure",
            )
        )

        fig_pressure.add_trace(
            go.Scatter(
                x=time_df["Date"],
                y=time_df["Average_P_Sat"],
                mode="lines+markers",
                name="Average Psat",
            )
        )

        fig_pressure.add_trace(
            go.Bar(
                x=time_df["Date"],
                y=time_df["Pressure_Delta"],
                name="P - Psat",
                opacity=0.35,
            )
        )

        fig_pressure.update_layout(
            title="Average Pressure vs Psat",
            xaxis_title="Date",
            yaxis_title="Pressure, psi",
            height=520,
            hovermode="x unified",
        )

        st.plotly_chart(fig_pressure, use_container_width=True)

    else:
        st.info("Pressure columns are missing or empty, so pressure plot is skipped.")

    st.subheader("Top Patterns Over Time")

    max_top_n = min(30, max(len(pattern_summary), 5))

    top_n = st.slider(
        "Top N patterns to show",
        min_value=5,
        max_value=max_top_n,
        value=min(10, max_top_n),
        step=1,
    )

    top_patterns = pattern_summary.head(top_n)["Pattern_Name"].tolist()

    metric_choice = st.selectbox(
        "Pattern trend metric",
        options=[
            "Oil_Rate_Mstbd",
            "Water_Rate_Mstbd",
            "WI_Rate_Mstbd",
            "VRR",
            "Water_Cut",
            "Oil_Cut",
            "Pressure_Delta",
        ],
        format_func=lambda x: {
            "Oil_Rate_Mstbd": "Oil Rate",
            "Water_Rate_Mstbd": "Water Rate",
            "WI_Rate_Mstbd": "WI Rate",
            "VRR": "VRR",
            "Water_Cut": "Water Cut",
            "Oil_Cut": "Oil Cut",
            "Pressure_Delta": "Pressure - Psat",
        }[x],
    )

    trend_df = pattern_time_df[pattern_time_df["Pattern_Name"].isin(top_patterns)].copy()

    fig_trend = px.line(
        trend_df,
        x="Date",
        y=metric_choice,
        color="Pattern_Name",
        markers=True,
        title=f"Top {top_n} Patterns - Trend",
        labels={
            metric_choice: metric_choice.replace("_", " "),
            "Pattern_Name": "Pattern",
        },
    )

    if metric_choice in ["Water_Cut", "Oil_Cut"]:
        fig_trend.update_yaxes(tickformat=".0%")

    if metric_choice == "Pressure_Delta":
        fig_trend.add_hline(
            y=0,
            line_dash="dash",
            annotation_text="Psat boundary",
        )

    fig_trend.update_layout(height=560, hovermode="x unified")
    st.plotly_chart(fig_trend, use_container_width=True)


# =============================================================================
# TAB 2 - PATTERN RANKING
# =============================================================================

with tab2:
    st.subheader("Pattern Value Ranking")

    max_rank_top_n = min(50, max(len(pattern_summary), 5))

    rank_top_n = st.slider(
        "Top N ranked patterns",
        min_value=5,
        max_value=max_rank_top_n,
        value=min(20, max_rank_top_n),
        step=1,
    )

    top_ranked = pattern_summary.head(rank_top_n).copy()

    fig_rank = px.bar(
        top_ranked.sort_values("Value_Index", ascending=True),
        x="Value_Index",
        y="Pattern_Name",
        orientation="h",
        title=f"Top {rank_top_n} Patterns by Value Index",
        labels={
            "Value_Index": "Value Index",
            "Pattern_Name": "Pattern",
        },
        hover_data=[
            "Rank",
            "Final_Score",
            "Health_Status",
            "Calc_Status",
            "Score_Components_Used",
            "Rate_Types",
            "Average_Res_P",
            "Average_P_Sat",
            "Pressure_Delta",
            "Avg_VRR",
            "Avg_Water_Cut",
            "Avg_Oil_Mstbd",
            "Avg_WI_Mstbd",
            "Wells",
            "CPS",
            "Manifold",
        ],
    )

    fig_rank.update_layout(height=650)
    st.plotly_chart(fig_rank, use_container_width=True)

    st.subheader("Score Components")

    score_comp = top_ranked[
        [
            "Pattern_Name",
            "Pressure_Score",
            "WC_Score",
            "VRR_Score",
            "Final_Score",
        ]
    ].copy()

    score_comp_long = score_comp.melt(
        id_vars=["Pattern_Name"],
        value_vars=[
            "Pressure_Score",
            "WC_Score",
            "VRR_Score",
            "Final_Score",
        ],
        var_name="Score_Component",
        value_name="Score",
    )

    fig_scores = px.bar(
        score_comp_long,
        x="Pattern_Name",
        y="Score",
        color="Score_Component",
        barmode="group",
        title="Pattern Score Components, 0 to 1",
        labels={
            "Pattern_Name": "Pattern",
            "Score": "Score, 0 to 1",
            "Score_Component": "Component",
        },
    )

    fig_scores.update_yaxes(range=[0, 1])
    fig_scores.update_layout(height=560, xaxis_tickangle=-45)

    st.plotly_chart(fig_scores, use_container_width=True)


# =============================================================================
# TAB 3 - HEAT MAP
# =============================================================================

with tab3:
    st.subheader("Pattern Health Heat Map")

    max_heat_n = min(100, max(len(pattern_summary), 10))

    heatmap_top_n = st.slider(
        "Number of patterns in heat map",
        min_value=10,
        max_value=max_heat_n,
        value=min(40, max_heat_n),
        step=5,
    )

    heatmap_sort = st.selectbox(
        "Sort heat map by",
        options=[
            "Rank",
            "Final_Score",
            "Pressure_Delta",
            "Avg_VRR",
            "Avg_Water_Cut",
            "Avg_Oil_Mstbd",
            "Avg_WI_Mstbd",
        ],
        index=0,
    )

    heat_df = pattern_summary.copy()

    if heatmap_sort == "Rank":
        heat_df = heat_df.sort_values("Rank", ascending=True)
    elif heatmap_sort in ["Final_Score", "Pressure_Delta", "Avg_Oil_Mstbd", "Avg_WI_Mstbd"]:
        heat_df = heat_df.sort_values(heatmap_sort, ascending=False)
    else:
        heat_df = heat_df.sort_values(heatmap_sort, ascending=True)

    heat_df = heat_df.head(heatmap_top_n).copy()

    heat_df["Overall_Health"] = heat_df["Final_Score"]
    heat_df["Pressure_Health"] = heat_df["Pressure_Score"]
    heat_df["Water_Cut_Health"] = heat_df["WC_Score"]
    heat_df["VRR_Health"] = heat_df["VRR_Score"]

    heat_cols = [
        "Overall_Health",
        "Pressure_Health",
        "Water_Cut_Health",
        "VRR_Health",
    ]

    heat_matrix = heat_df.set_index("Pattern_Name")[heat_cols]

    fig_heat = px.imshow(
        heat_matrix,
        aspect="auto",
        color_continuous_scale="RdYlGn",
        zmin=0,
        zmax=1,
        title="Pattern Health Heat Map: Green = Healthy, Red = Unhealthy, Blank = Component skipped",
        labels=dict(
            x="Health Indicator",
            y="Pattern",
            color="Health Score",
        ),
    )

    fig_heat.update_layout(
        height=max(500, heatmap_top_n * 22),
        xaxis_title="Indicator",
        yaxis_title="Pattern",
    )

    st.plotly_chart(fig_heat, use_container_width=True)

    st.caption(
        "Pressure component is skipped where Average of Res P or Average of P Sat is missing. "
        "Final Score is calculated only from available components."
    )

    st.subheader("VRR vs Water Cut Pattern Health Map")
    st.subheader("Pattern Health Tree Map")

    treemap_df = pattern_summary.copy()
    
    treemap_df["Primary_AREA"] = (
        treemap_df["AREA"]
        .fillna("Blank")
        .astype(str)
        .str.split(",")
        .str[0]
        .str.strip()
    )
    
    treemap_df["Primary_CPS"] = (
        treemap_df["CPS"]
        .fillna("Blank")
        .astype(str)
        .str.split(",")
        .str[0]
        .str.strip()
    )
    
    treemap_df["Primary_Reservoir"] = (
        treemap_df["Reservoir"]
        .fillna("Blank")
        .astype(str)
        .str.split(",")
        .str[0]
        .str.strip()
    )
    
    treemap_metric_map = {
        "Overall Pattern Health": {
            "column": "Final_Score",
            "label": "Overall Health Score",
            "scale": "RdYlGn",
            "range": [0, 1],
            "note": "Green = healthier pattern, Red = weaker pattern.",
        },
        "Pressure Health": {
            "column": "Pressure_Score",
            "label": "Pressure Health Score",
            "scale": "RdYlGn",
            "range": [0, 1],
            "note": "Green = Res P safely above Psat, Red = close/below Psat.",
        },
        "Water Cut Health": {
            "column": "WC_Score",
            "label": "Water Cut Health Score",
            "scale": "RdYlGn",
            "range": [0, 1],
            "note": "Green = lower/healthier WC, Red = higher/unhealthy WC.",
        },
        "VRR Health": {
            "column": "VRR_Score",
            "label": "VRR Health Score",
            "scale": "RdYlGn",
            "range": [0, 1],
            "note": "Green = VRR close to target, Red = far from target.",
        },
        "Average VRR": {
            "column": "Avg_VRR",
            "label": "Average VRR",
            "scale": "RdYlGn",
            "range": None,
            "note": "Color is based on raw average VRR.",
        },
        "Average Water Cut": {
            "column": "Avg_Water_Cut",
            "label": "Average Water Cut",
            "scale": "RdYlGn_r",
            "range": [0, 1],
            "note": "Green = lower WC, Red = higher WC.",
        },
        "Pressure Margin P-Psat": {
            "column": "Pressure_Delta",
            "label": "Pressure Margin, psi",
            "scale": "RdYlGn",
            "range": None,
            "note": "Green = higher pressure margin above Psat.",
        },
    }
    
    treemap_metric = st.selectbox(
        "Tree map color by",
        options=list(treemap_metric_map.keys()),
        index=0,
    )
    
    treemap_size_by = st.selectbox(
        "Tree map box size by",
        options=[
            "Value Index",
            "Average Oil Rate",
            "Average WI Rate",
            "Average Liquid Rate",
            "Number of Wells",
        ],
        index=1,
    )
    
    if treemap_size_by == "Value Index":
        treemap_df["Tree_Size"] = treemap_df["Value_Index"]
    
    elif treemap_size_by == "Average Oil Rate":
        treemap_df["Tree_Size"] = treemap_df["Avg_Oil_Mstbd"]
    
    elif treemap_size_by == "Average WI Rate":
        treemap_df["Tree_Size"] = treemap_df["Avg_WI_Mstbd"]
    
    elif treemap_size_by == "Average Liquid Rate":
        treemap_df["Tree_Size"] = (
            treemap_df["Avg_Oil_Mstbd"].fillna(0)
            + treemap_df["Avg_Water_Mstbd"].fillna(0)
        )
    
    else:
        treemap_df["Tree_Size"] = treemap_df["Wells"]
    
    treemap_df["Tree_Size"] = pd.to_numeric(
        treemap_df["Tree_Size"],
        errors="coerce",
    ).fillna(0)
    
    # Plotly treemap does not like zero/negative box sizes.
    treemap_df["Tree_Size"] = treemap_df["Tree_Size"].clip(lower=0.001)
    
    selected_metric_info = treemap_metric_map[treemap_metric]
    color_col = selected_metric_info["column"]
    
    treemap_df[color_col] = pd.to_numeric(
        treemap_df[color_col],
        errors="coerce",
    )
    
    fig_tree = px.treemap(
        treemap_df,
        path=[
            "Primary_AREA",
            "Primary_CPS",
            "Primary_Reservoir",
            "Pattern_Name",
        ],
        values="Tree_Size",
        color=color_col,
        color_continuous_scale=selected_metric_info["scale"],
        range_color=selected_metric_info["range"],
        title=f"Pattern Tree Map - Color: {treemap_metric}, Size: {treemap_size_by}",
        hover_data={
            "Pattern_Name": True,
            "Health_Status": True,
            "Calc_Status": True,
            "Final_Score": ":.2f",
            "Pressure_Score": ":.2f",
            "WC_Score": ":.2f",
            "VRR_Score": ":.2f",
            "Avg_VRR": ":.2f",
            "Avg_Water_Cut": ":.1%",
            "Avg_Oil_Mstbd": ":.2f",
            "Avg_WI_Mstbd": ":.2f",
            "Average_Res_P": ":.0f",
            "Average_P_Sat": ":.0f",
            "Pressure_Delta": ":.0f",
            "Producer_Count": True,
            "Water_Injector_Count": True,
            "Producers": True,
            "Water_Injectors": True,
            "Tree_Size": False,
        },
    )
    
    fig_tree.update_traces(
        textinfo="label+value",
        textfont_size=12,
        marker=dict(line=dict(width=1)),
    )
    
    fig_tree.update_layout(
        height=850,
        margin=dict(t=60, l=10, r=10, b=10),
        coloraxis_colorbar=dict(
            title=selected_metric_info["label"],
        ),
    )
    
    st.plotly_chart(fig_tree, use_container_width=True)
    
    st.markdown(
        f"""
        **Tree map color coding:**  
        🟢 **Green** = healthier / better  
        🟡 **Yellow** = watch area  
        🔴 **Red** = unhealthy / weaker  
    
        **Selected metric:** `{treemap_metric}`  
        **Logic:** {selected_metric_info["note"]}
        """
    )

    bubble_df = heat_df.copy()

    fig_bubble = px.scatter(
        bubble_df,
        x="Avg_VRR",
        y="Avg_Water_Cut",
        size="Avg_Oil_Mstbd",
        color="Final_Score",
        hover_name="Pattern_Name",
        color_continuous_scale="RdYlGn",
        range_color=[0, 1],
        title="VRR vs Water Cut Pattern Health Map",
        labels={
            "Avg_VRR": "Average VRR",
            "Avg_Water_Cut": "Average Water Cut",
            "Final_Score": "Pattern Score",
            "Avg_Oil_Mstbd": "Avg Oil, Mstb/d",
        },
        hover_data=[
            "Rank",
            "Health_Status",
            "Calc_Status",
            "Score_Components_Used",
            "Rate_Types",
            "Average_Res_P",
            "Average_P_Sat",
            "Pressure_Delta",
            "Pressure_Score",
            "Avg_Oil_Mstbd",
            "Avg_Water_Mstbd",
            "Avg_WI_Mstbd",
            "Final_Score",
            "VRR_Score",
            "WC_Score",
            "CPS",
            "Manifold",
            "Well_List",
        ],
    )

    fig_bubble.add_vline(
        x=target_vrr,
        line_dash="dash",
        annotation_text=f"Target VRR {target_vrr:.2f}",
    )

    fig_bubble.update_yaxes(tickformat=".0%")

    fig_bubble.update_layout(
        height=600,
        hovermode="closest",
    )

    st.plotly_chart(fig_bubble, use_container_width=True)

    st.subheader("Pressure Health Map")

    pressure_bubble_df = heat_df.copy()

    fig_pressure_bubble = px.scatter(
        pressure_bubble_df,
        x="Pressure_Delta",
        y="Final_Score",
        size="Avg_Oil_Mstbd",
        color="Pressure_Score",
        hover_name="Pattern_Name",
        color_continuous_scale="RdYlGn",
        range_color=[0, 1],
        title="Pressure Margin vs Pattern Score",
        labels={
            "Pressure_Delta": "Average Res P - Average Psat, psi",
            "Final_Score": "Final Pattern Score",
            "Pressure_Score": "Pressure Score",
            "Avg_Oil_Mstbd": "Avg Oil, Mstb/d",
        },
        hover_data=[
            "Rank",
            "Average_Res_P",
            "Average_P_Sat",
            "Pressure_Delta",
            "Pressure_Margin_Fraction",
            "Pressure_Score",
            "Final_Score",
            "Score_Components_Used",
            "Health_Status",
            "Calc_Status",
        ],
    )

    fig_pressure_bubble.add_vline(
        x=0,
        line_dash="dash",
        annotation_text="Psat boundary",
    )

    fig_pressure_bubble.update_layout(
        height=600,
        hovermode="closest",
    )

    st.plotly_chart(fig_pressure_bubble, use_container_width=True)


# =============================================================================
# TAB 4 - CREAMING CURVE
# =============================================================================

with tab4:
    st.subheader("Creaming Curve")

    fig_cream = go.Figure()

    fig_cream.add_trace(
        go.Scatter(
            x=pattern_summary["Rank"],
            y=pattern_summary["Cumulative_Value_Index"],
            mode="lines+markers",
            name="Cumulative Value Index",
            text=pattern_summary["Pattern_Name"],
            hovertemplate=(
                "Rank: %{x}<br>"
                "Pattern: %{text}<br>"
                "Cumulative Value: %{y:,.2f}<extra></extra>"
            ),
        )
    )

    fig_cream.update_layout(
        title="Creaming Curve - Ranked Pattern Value",
        xaxis_title="Pattern Rank",
        yaxis_title="Cumulative Value Index",
        height=560,
        hovermode="closest",
    )

    st.plotly_chart(fig_cream, use_container_width=True)

    fig_cream_pct = go.Figure()

    fig_cream_pct.add_trace(
        go.Scatter(
            x=pattern_summary["Rank"],
            y=pattern_summary["Cumulative_Value_Percent"],
            mode="lines+markers",
            name="Cumulative Value %",
            text=pattern_summary["Pattern_Name"],
            hovertemplate=(
                "Rank: %{x}<br>"
                "Pattern: %{text}<br>"
                "Cumulative Value: %{y:.1%}<extra></extra>"
            ),
        )
    )

    fig_cream_pct.update_yaxes(tickformat=".0%")

    fig_cream_pct.update_layout(
        title="Creaming Curve - Cumulative Value %",
        xaxis_title="Pattern Rank",
        yaxis_title="Cumulative Value %",
        height=520,
        hovermode="closest",
    )

    st.plotly_chart(fig_cream_pct, use_container_width=True)

    st.info(
        "Creaming curve ranks patterns from highest to lowest Value Index. "
        "Value Index = Final Score × selected value driver."
    )


# =============================================================================
# TAB 5 - DATA TABLES
# =============================================================================

with tab5:
    st.subheader("Pattern Ranking Table")

    display_df = pattern_summary.copy()

    table_cols = [
        "Rank",
        "Pattern_Name",
        "Health_Status",
        "Calc_Status",
        "Score_Components_Used",
        "Rate_Types",
        "Reservoir",
        "AREA",
        "Wells",
        "Producer_Count",
        "Water_Injector_Count",
        "Producers",
        "Water_Injectors",
        "Well_List",
        "CPS",
        "Manifold",
        "Average_Res_P",
        "Average_P_Sat",
        "Pressure_Delta",
        "Pressure_Margin_Fraction",
        "Pressure_Above_Psat",
        "Pressure_Score",
        "Avg_Oil_Mstbd",
        "Avg_Water_Mstbd",
        "Avg_WI_Mstbd",
        "Cum_Oil_Mstb",
        "Cum_Water_Mstb",
        "Cum_WI_Mstb",
        "Avg_VRR",
        "Min_VRR",
        "Max_VRR",
        "Avg_Water_Cut",
        "Max_Water_Cut",
        "Avg_Oil_Cut",
        "WC_Score",
        "VRR_Score",
        "Final_Score",
        "Value_Index",
        "Cumulative_Value_Percent",
        "Input_VRR",
        "AA",
    ]

    table_cols = [c for c in table_cols if c in display_df.columns]
    display_df = display_df[table_cols].copy()

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Avg_Water_Cut": st.column_config.NumberColumn(
                "Avg Water Cut",
                format="%.2f",
            ),
            "Max_Water_Cut": st.column_config.NumberColumn(
                "Max Water Cut",
                format="%.2f",
            ),
            "Avg_Oil_Cut": st.column_config.NumberColumn(
                "Avg Oil Cut",
                format="%.2f",
            ),
            "Pressure_Margin_Fraction": st.column_config.NumberColumn(
                "Pressure Margin",
                format="%.2f",
            ),
            "Pressure_Score": st.column_config.ProgressColumn(
                "Pressure Score",
                format="%.2f",
                min_value=0,
                max_value=1,
            ),
            "WC_Score": st.column_config.ProgressColumn(
                "WC Score",
                format="%.2f",
                min_value=0,
                max_value=1,
            ),
            "VRR_Score": st.column_config.ProgressColumn(
                "VRR Score",
                format="%.2f",
                min_value=0,
                max_value=1,
            ),
            "Final_Score": st.column_config.ProgressColumn(
                "Final Score",
                format="%.2f",
                min_value=0,
                max_value=1,
            ),
            "Cumulative_Value_Percent": st.column_config.ProgressColumn(
                "Cumulative Value",
                format="%.2f",
                min_value=0,
                max_value=1,
            ),
        },
    )

    csv = pattern_summary.to_csv(index=False).encode("utf-8-sig")


#_________________________________________________    
st.divider()
st.subheader("Water Injector Ranking Table")

if injector_summary.empty:
    st.info("No water injectors found in the selected data.")
else:
    injector_display = injector_summary.copy()

    injector_table_cols = [
        "Injector_Rank",
        "Water_Injector",
        "Injector_Health_Status",
        "Injector_Final_Score",
        "Injector_Pressure_Score",
        "Injector_WC_Score",
        "Injector_VRR_Score",
        "Allocated_Value_Index",
        "Allocated_Avg_WI_Mstbd",
        "Allocated_Cum_WI_Mstb",
        "Allocated_Avg_Oil_Mstbd",
        "Allocated_Cum_Oil_Mstb",
        "Average_Res_P",
        "Average_P_Sat",
        "Pressure_Delta",
        "Pressure_Margin_Fraction",
        "Avg_VRR",
        "Avg_Water_Cut",
        "Injector_Pattern_Count",
        "Patterns",
        "Injector_CPS",
        "Injector_Manifold",
        "Reservoir",
        "AREA",
        "Parent_Health_Status",
        "Parent_Calc_Status",
        "Score_Components_Used",
        "Pattern_Producers",
        "Pattern_All_Injectors",
    ]

    injector_table_cols = [
        c for c in injector_table_cols
        if c in injector_display.columns
    ]

    injector_display = injector_display[injector_table_cols].copy()

    st.dataframe(
        injector_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Injector_Final_Score": st.column_config.ProgressColumn(
                "Injector Final Score",
                format="%.2f",
                min_value=0,
                max_value=1,
            ),
            "Injector_Pressure_Score": st.column_config.ProgressColumn(
                "Pressure Score",
                format="%.2f",
                min_value=0,
                max_value=1,
            ),
            "Injector_WC_Score": st.column_config.ProgressColumn(
                "WC Score",
                format="%.2f",
                min_value=0,
                max_value=1,
            ),
            "Injector_VRR_Score": st.column_config.ProgressColumn(
                "VRR Score",
                format="%.2f",
                min_value=0,
                max_value=1,
            ),
            "Avg_Water_Cut": st.column_config.NumberColumn(
                "Avg Water Cut",
                format="%.2f",
            ),
            "Pressure_Margin_Fraction": st.column_config.NumberColumn(
                "Pressure Margin",
                format="%.2f",
            ),
        },
    )

    injector_csv = injector_summary.to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        label="⬇️ Download water injector ranking CSV",
        data=injector_csv,
        file_name="water_injector_ranking.csv",
        mime="text/csv",
    )

    with st.expander("Injector allocation detail by Pattern"):
        detail_cols = [
            "Pattern_Name",
            "Water_Injector",
            "Injectors_In_Pattern",
            "Allocated_Avg_WI_Mstbd",
            "Allocated_Cum_WI_Mstb",
            "Allocated_Value_Index",
            "Final_Score",
            "Pressure_Score",
            "WC_Score",
            "VRR_Score",
            "Average_Res_P",
            "Average_P_Sat",
            "Pressure_Delta",
            "Avg_VRR",
            "Avg_Water_Cut",
            "Health_Status",
            "Calc_Status",
            "Producers",
            "Water_Injectors",
            "Injector_CPS",
            "Injector_Manifold",
        ]

        detail_cols = [
            c for c in detail_cols
            if c in injector_pattern_detail.columns
        ]

        st.dataframe(
            injector_pattern_detail[detail_cols],
            use_container_width=True,
            hide_index=True,
        )

        detail_csv = injector_pattern_detail.to_csv(index=False).encode("utf-8-sig")

        st.download_button(
            label="⬇️ Download injector allocation detail CSV",
            data=detail_csv,
            file_name="water_injector_pattern_allocation_detail.csv",
            mime="text/csv",
        )
#----------------------------------------------------------------------------
    st.download_button(
        label="⬇️ Download pattern ranking CSV",
        data=csv,
        file_name="flowline_pattern_ranking.csv",
        mime="text/csv",

    )

    st.subheader("Monthly Aggregated Forecast Table")

    monthly_display = time_df.copy()
    monthly_display["Water_Cut_%"] = monthly_display["Water_Cut"] * 100
    monthly_display["Oil_Cut_%"] = monthly_display["Oil_Cut"] * 100

    monthly_cols = [
        "Date",
        "Oil_Rate_Mstbd",
        "Water_Rate_Mstbd",
        "WI_Rate_Mstbd",
        "Produced_Liquid_Mstbd",
        "Produced_Voidage_Mstbd",
        "VRR",
        "Water_Cut_%",
        "Oil_Cut_%",
        "Average_Res_P",
        "Average_P_Sat",
        "Pressure_Delta",
        "Oil_Volume_Mstb",
        "Water_Volume_Mstb",
        "WI_Volume_Mstb",
    ]

    monthly_cols = [c for c in monthly_cols if c in monthly_display.columns]

    st.dataframe(
        monthly_display[monthly_cols],
        use_container_width=True,
        hide_index=True,
    )


# =============================================================================
# DATA QUALITY / QA
# =============================================================================

with st.expander("🔍 Data quality / detected structure"):
    st.write("Detected rate streams:", rate_types_found)
    st.write("Pressure columns found:", pressure_cols_found)
    st.write("Number of detected forecast date columns:", len(forecast_date_cols))
    st.write("First detected date:", pd.to_datetime(model_df["Date"]).min())
    st.write("Last detected date:", pd.to_datetime(model_df["Date"]).max())
    st.write("Raw rows:", len(raw_df))
    st.write("Long model rows after reshaping:", len(model_df))

    st.subheader("Pattern Rate QA")

    pattern_options = sorted_unique(metric_df["Pattern_Name"])

    default_pattern_index = (
        pattern_options.index("CPS-1CS")
        if "CPS-1CS" in pattern_options
        else 0
    )

    qa_pattern = st.selectbox(
        "Check Pattern_Name",
        options=pattern_options,
        index=default_pattern_index,
        key="qa_pattern_check",
    )

    qa_df = metric_df[
        metric_df["Pattern_Name"].astype(str) == str(qa_pattern)
    ].copy()

    qa_rate_summary = (
        qa_df
        .groupby(["Pattern_Name", "Rate_Type"], dropna=False)
        .agg(
            Wells=("Well", "nunique"),
            Avg_Rate=("Rate_Mstbd", "mean"),
            Max_Rate=("Rate_Mstbd", "max"),
            Rows=("Rate_Mstbd", "size"),
        )
        .reset_index()
    )

    st.write("Rate types found for selected pattern:")
    st.dataframe(
        qa_rate_summary,
        use_container_width=True,
        hide_index=True,
    )

    st.write("Rows included for selected pattern:")
    st.dataframe(
        qa_df[
            [
                "Pattern_Name",
                "Well",
                "CPS",
                "Manifold",
                "RATES",
                "Rate_Type",
                "Reservoir",
                "AREA",
                "Date",
                "Rate_Mstbd",
                "Average_Res_P",
                "Average_P_Sat",
            ]
        ].head(300),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Preview of reshaped long model data")
    st.dataframe(model_df.head(100), use_container_width=True)
