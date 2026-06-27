import os
import json
import sqlite3
from pathlib import Path
import cohere
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# =========================================================
# ADVANCED TELANGANA WATER STRESS STREAMLIT APP
# Save this file as: app.py
# Run command: streamlit run app.py
# =========================================================

st.set_page_config(
    page_title="Telangana Water Stress Dashboard",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# 1. PROJECT PATH SETUP
# -----------------------------
# Your folder path. Keep all CSV and GeoJSON files inside this folder.
DEFAULT_PROJECT_PATH = r"C:\Users\debas\Python_File_Social Prachar\water_stress_project\water_stress_project"

# If app.py is inside the same folder as datasets, this also works.
CURRENT_FILE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = Path(DEFAULT_PROJECT_PATH)

if not PROJECT_DIR.exists():
    PROJECT_DIR = CURRENT_FILE_DIR

# File names based on your uploaded datasets.
FILE_CANDIDATES = {
    "main": [
        "telangana_water_data_updated.csv",
        "telangana_water_data_updated.csv",
        "telangana_water_data.csv",
        "telangana_water_data.csv",
    ],
    "districts": ["Dim_Districts.csv", "Dim_Districts.csv"],
    "date": ["Dim_Date.csv", "Dim_Date.csv"],
    "rainfall": ["Fact_Rainfall.csv", "Fact_Rainfall.csv"],
    "groundwater": ["Fact_Groundwater.csv", "Fact_Groundwater.csv"],
    "reservoirs": ["Fact_Reservoirs.csv", "Fact_Reservoirs.csv"],
    "geojson": ["TELANGANA_DISTRICTS.geojson"],
}


def find_file(file_list):
    """Find first available file from candidate names."""
    for name in file_list:
        path = PROJECT_DIR / name
        if path.exists():
            return path
    return None


def clean_col_names(df):
    """Standard clean column names: remove extra spaces."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def normalize_text(x):
    """Normalize district names so dataset and GeoJSON can match correctly."""
    if pd.isna(x):
        return ""
    text = str(x).strip().lower()
    text = text.replace("-", " ")
    text = " ".join(text.split())

    # Common spelling differences between Dim_Districts and Telangana_LGD.geojson
    fixes = {
        "hanamkonda": "hanumakonda",
        "jagtial": "jagitial",
        "jangaon": "jangoan",
        "jayashankar bhupalpally": "jayashankar bhupalapally",
        "komaram bheem asifabad": "kumuram bheem asifabad",
        "medchal malkajgiri": "medchal malkajgiri",
        "medchal malkajgiri": "medchal malkajgiri",
    }
    return fixes.get(text, text)


@st.cache_data(show_spinner=False)
def load_csv(path):
    if path is None:
        return pd.DataFrame()
    return clean_col_names(pd.read_csv(path))


@st.cache_data(show_spinner=False)
def load_geojson(path):
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------
# 2. LOAD DATASETS
# -----------------------------
main_path = find_file(FILE_CANDIDATES["main"])
district_path = find_file(FILE_CANDIDATES["districts"])
date_path = find_file(FILE_CANDIDATES["date"])
rainfall_path = find_file(FILE_CANDIDATES["rainfall"])
groundwater_path = find_file(FILE_CANDIDATES["groundwater"])
reservoir_path = find_file(FILE_CANDIDATES["reservoirs"])
geojson_path = find_file(FILE_CANDIDATES["geojson"])

main_df = load_csv(main_path)
district_df = load_csv(district_path)
date_df = load_csv(date_path)
rainfall_df = load_csv(rainfall_path)
groundwater_df = load_csv(groundwater_path)
reservoir_df = load_csv(reservoir_path)
geo_data = load_geojson(geojson_path)

# -----------------------------
# 3. DATA PREPARATION
# -----------------------------

def prepare_main_data(df):
    df = df.copy()

    # Create date safely. Your updated file has "date"; older file has year/month.
    if "date" in df.columns:
        df["Date"] = pd.to_datetime(df["date"], errors="coerce")
    elif "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    elif {"year", "month"}.issubset(df.columns):
        df["Date"] = pd.to_datetime(
            dict(year=df["year"], month=df["month"], day=1), errors="coerce"
        )
    elif {"Year", "Month"}.issubset(df.columns):
        df["Date"] = pd.to_datetime(
            dict(year=df["Year"], month=df["Month"], day=1), errors="coerce"
        )
    else:
        df["Date"] = pd.NaT

    # Standard district column.
    if "district" not in df.columns:
        for col in ["District", "Name", "district_name"]:
            if col in df.columns:
                df["district"] = df[col]
                break

    # Numeric safety.
    numeric_cols = [
        "population", "rainfall", "groundwater", "temperature", "water_usage",
        "storage_capacity", "urbanization_rate", "wsi", "water_stress_ratio",
        "population_density", "water_availability", "Precipitation_mm",
        "Current_Depth_m", "Historical_Avg_mm", "Rainfall_Deviation_Percent",
        "WSI_Score_Custom", "high_risk_flag"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Create missing useful columns if not present.
    if "year" not in df.columns and df["Date"].notna().any():
        df["year"] = df["Date"].dt.year
    if "month" not in df.columns and df["Date"].notna().any():
        df["month"] = df["Date"].dt.month

    if "Stress_Category_Custom" not in df.columns:
        if "stress_level" in df.columns:
            df["Stress_Category_Custom"] = df["stress_level"]
        elif "wsi" in df.columns:
            df["Stress_Category_Custom"] = pd.cut(
                df["wsi"],
                bins=[-np.inf, 0.25, 0.50, 0.75, np.inf],
                labels=["Low", "Medium", "High", "Critical"]
            ).astype(str)
        else:
            df["Stress_Category_Custom"] = "Unknown"

    if "wsi" not in df.columns:
        if "WSI_Score_Custom" in df.columns:
            df["wsi"] = df["WSI_Score_Custom"] / 100
        else:
            df["wsi"] = np.nan

    return df


def prepare_star_schema(districts, rainfall, groundwater, reservoirs):
    """Prepare district-level summary from star schema files as fallback/additional analysis."""
    if districts.empty:
        return pd.DataFrame()

    districts = districts.copy()
    if "Name" in districts.columns:
        districts["district"] = districts["Name"]

    summary = districts[[c for c in ["District_ID", "district", "Population_2023", "Industrial_Zone_Count", "Lat", "Long"] if c in districts.columns]].copy()

    if not rainfall.empty and "District_ID" in rainfall.columns:
        rf = rainfall.groupby("District_ID", as_index=False).agg(
            Avg_Rainfall=("Precipitation_mm", "mean"),
            Avg_Historical_Rainfall=("Historical_Avg_mm", "mean")
        )
        rf["Rainfall_Deviation_Percent"] = ((rf["Avg_Rainfall"] - rf["Avg_Historical_Rainfall"]) / rf["Avg_Historical_Rainfall"].replace(0, np.nan)) * 100
        summary = summary.merge(rf, on="District_ID", how="left")

    if not groundwater.empty and "District_ID" in groundwater.columns:
        gw = groundwater.groupby("District_ID", as_index=False).agg(
            Avg_Groundwater_Depth=("Water_Table_Depth_m", "mean"),
            Avg_Recharge_Rate=("Recharge_Rate", "mean")
        )
        summary = summary.merge(gw, on="District_ID", how="left")

    if not reservoirs.empty and "District_ID" in reservoirs.columns:
        res = reservoirs.groupby("District_ID", as_index=False).agg(
            Avg_Current_Capacity=("Current_Capacity_TMC", "mean"),
            Avg_Max_Capacity=("Max_Capacity_TMC", "mean")
        )
        res["Reservoir_Fill_Percent"] = (res["Avg_Current_Capacity"] / res["Avg_Max_Capacity"].replace(0, np.nan)) * 100
        summary = summary.merge(res, on="District_ID", how="left")

    return summary


main_df = prepare_main_data(main_df)
star_summary = prepare_star_schema(district_df, rainfall_df, groundwater_df, reservoir_df)

# Add lat/long to main data using district table.
if not main_df.empty and not district_df.empty and "district" in main_df.columns and "Name" in district_df.columns:
    district_lookup = district_df.copy()
    district_lookup["district_key"] = district_lookup["Name"].apply(normalize_text)
    main_df["district_key"] = main_df["district"].apply(normalize_text)
    keep_cols = ["district_key"] + [c for c in ["District_ID", "Lat", "Long", "Total_Area_SQKM", "Population_2023", "Industrial_Zone_Count"] if c in district_lookup.columns]
    main_df = main_df.merge(district_lookup[keep_cols], on="district_key", how="left")

# -----------------------------
# 4. SIDEBAR
# -----------------------------
st.sidebar.title("💧 Water Stress Controls")
st.sidebar.caption("Telangana Water Stress Analysis Project")

with st.sidebar.expander("📁 Loaded Files", expanded=False):
    st.write("Project folder:")
    st.code(str(PROJECT_DIR))
    st.write("Main data:", main_path.name if main_path else "Not found")
    st.write("GeoJSON:", geojson_path.name if geojson_path else "Not found")
    st.write("Districts:", district_path.name if district_path else "Not found")
    st.write("Rainfall:", rainfall_path.name if rainfall_path else "Not found")
    st.write("Groundwater:", groundwater_path.name if groundwater_path else "Not found")
    st.write("Reservoirs:", reservoir_path.name if reservoir_path else "Not found")

if main_df.empty:
    st.error("No main CSV file found. Place telangana_water_data_updated.csv or telangana_water_data.csv in your project folder.")
    st.stop()

# Filters
available_districts = sorted(main_df["district"].dropna().unique()) if "district" in main_df.columns else []
selected_districts = st.sidebar.multiselect(
    "Select Districts",
    available_districts,
    default=available_districts
)

if "year" in main_df.columns:
    years = sorted(main_df["year"].dropna().astype(int).unique())
    selected_years = st.sidebar.multiselect("Select Years", years, default=years)
else:
    selected_years = []

if "season" in main_df.columns:
    seasons = sorted(main_df["season"].dropna().unique())
    selected_seasons = st.sidebar.multiselect("Select Seasons", seasons, default=seasons)
else:
    selected_seasons = []

stress_col = "Stress_Category_Custom"
if stress_col in main_df.columns:
    stress_values = sorted(main_df[stress_col].dropna().unique())
    selected_stress = st.sidebar.multiselect("Select Stress Category", stress_values, default=stress_values)
else:
    selected_stress = []

# Apply filters safely.
filtered_df = main_df.copy()
if selected_districts and "district" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["district"].isin(selected_districts)]
if selected_years and "year" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["year"].isin(selected_years)]
if selected_seasons and "season" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["season"].isin(selected_seasons)]
if selected_stress and stress_col in filtered_df.columns:
    filtered_df = filtered_df[filtered_df[stress_col].isin(selected_stress)]

# -----------------------------
# 5. HELPER FUNCTIONS FOR UI
# -----------------------------
def metric_card(label, value, help_text=None):
    st.metric(label, value, help=help_text)


def safe_mean(df, col):
    if col in df.columns and not df[col].dropna().empty:
        return df[col].mean()
    return np.nan


def safe_sum(df, col):
    if col in df.columns and not df[col].dropna().empty:
        return df[col].sum()
    return np.nan


def format_num(x, decimals=2):
    if pd.isna(x):
        return "N/A"
    return f"{x:,.{decimals}f}"


def district_summary(df):
    agg_dict = {}
    for col in ["wsi", "WSI_Score_Custom", "rainfall", "groundwater", "water_usage", "population", "water_availability", "high_risk_flag", "Lat", "Long"]:
        if col in df.columns:
            agg_dict[col] = "mean"
    if not agg_dict or "district" not in df.columns:
        return pd.DataFrame()
    out = df.groupby("district", as_index=False).agg(agg_dict)
    return out


def classify_wsi(score):
    if pd.isna(score):
        return "Unknown"
    if score >= 75:
        return "Critical"
    if score >= 50:
        return "High"
    if score >= 25:
        return "Medium"
    return "Low"


# -----------------------------
# 6. MAIN LAYOUT
# -----------------------------
st.title("💧 Advanced Telangana Water Stress Dashboard")
st.markdown(
    "This dashboard analyzes rainfall, groundwater, reservoir status, population pressure, and custom Water Stress Index for Telangana districts."
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Executive Summary",
        "District Analysis",
        "Temporal Trends",
        "Geo Map",
        "Stress Simulator",
        "Star Schema Analysis",
        "AI Insights",
        "Raw Data & Quality Check",
    ]
)

# -----------------------------
# PAGE 1: EXECUTIVE SUMMARY
# -----------------------------
if page == "Executive Summary":
    st.subheader("📌 Executive Summary")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        metric_card("Districts", filtered_df["district"].nunique() if "district" in filtered_df.columns else "N/A")
    with c2:
        metric_card("Avg WSI", format_num(safe_mean(filtered_df, "wsi"), 3))
    with c3:
        metric_card("Avg Custom WSI", format_num(safe_mean(filtered_df, "WSI_Score_Custom"), 2))
    with c4:
        metric_card("Avg Rainfall", format_num(safe_mean(filtered_df, "rainfall"), 2))
    with c5:
        metric_card("High Risk Avg", format_num(safe_mean(filtered_df, "high_risk_flag"), 2))

    st.divider()

    left, right = st.columns(2)

    with left:
        if stress_col in filtered_df.columns:
            stress_count = filtered_df[stress_col].value_counts().reset_index()
            stress_count.columns = ["Stress Category", "Records"]
            fig = px.pie(stress_count, names="Stress Category", values="Records", title="Stress Category Distribution", hole=0.45)
            st.plotly_chart(fig, use_container_width=True)

    with right:
        ds = district_summary(filtered_df)
        if not ds.empty and "WSI_Score_Custom" in ds.columns:
            top = ds.sort_values("WSI_Score_Custom", ascending=False).head(10)
            fig = px.bar(top, x="district", y="WSI_Score_Custom", title="Top 10 Districts by Custom WSI", text_auto=True)
            fig.update_layout(xaxis_title="District", yaxis_title="Custom WSI Score")
            st.plotly_chart(fig, use_container_width=True)
        elif not ds.empty and "wsi" in ds.columns:
            top = ds.sort_values("wsi", ascending=False).head(10)
            fig = px.bar(top, x="district", y="wsi", title="Top 10 Districts by WSI", text_auto=True)
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("🔎 Key Insights")
    insights = []
    if "district" in filtered_df.columns and "WSI_Score_Custom" in filtered_df.columns:
        ds = district_summary(filtered_df)
        if not ds.empty:
            highest = ds.sort_values("WSI_Score_Custom", ascending=False).iloc[0]
            lowest = ds.sort_values("WSI_Score_Custom", ascending=True).iloc[0]
            insights.append(f"Highest average custom WSI district: **{highest['district']}** ({highest['WSI_Score_Custom']:.2f}).")
            insights.append(f"Lowest average custom WSI district: **{lowest['district']}** ({lowest['WSI_Score_Custom']:.2f}).")
    if "Rainfall_Deviation_Percent" in filtered_df.columns:
        avg_dev = filtered_df["Rainfall_Deviation_Percent"].mean()
        insights.append(f"Average rainfall deviation: **{avg_dev:.2f}%**.")
    if "high_risk_flag" in filtered_df.columns:
        risk_pct = filtered_df["high_risk_flag"].mean() * 100
        insights.append(f"High-risk record percentage: **{risk_pct:.2f}%**.")

    if insights:
        for item in insights:
            st.markdown("- " + item)
    else:
        st.info("Insights will appear when required columns are available.")

# -----------------------------
# PAGE 2: DISTRICT ANALYSIS
# -----------------------------
elif page == "District Analysis":
    st.subheader("🏙️ District-wise Analysis")

    ds = district_summary(filtered_df)
    if ds.empty:
        st.warning("District summary cannot be created because district column is missing.")
    else:
        sort_col = "WSI_Score_Custom" if "WSI_Score_Custom" in ds.columns else "wsi"
        ds = ds.sort_values(sort_col, ascending=False)

        st.dataframe(ds, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            if sort_col in ds.columns:
                fig = px.bar(ds.head(15), x="district", y=sort_col, title=f"District Ranking by {sort_col}", text_auto=True)
                st.plotly_chart(fig, use_container_width=True)
        with c2:
            scatter_x = "rainfall" if "rainfall" in ds.columns else None
            scatter_y = "groundwater" if "groundwater" in ds.columns else None
            if scatter_x and scatter_y:
                fig = px.scatter(
                    ds, x=scatter_x, y=scatter_y, size=sort_col if sort_col in ds.columns else None,
                    hover_name="district", title="Rainfall vs Groundwater Pressure"
                )
                st.plotly_chart(fig, use_container_width=True)

        st.subheader("📊 Correlation Heatmap")
        num = filtered_df.select_dtypes(include=np.number)
        if num.shape[1] >= 2:
            corr = num.corr(numeric_only=True)
            fig = px.imshow(corr, text_auto=True, aspect="auto", title="Numeric Column Correlation")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough numeric columns for correlation.")

# -----------------------------
# PAGE 3: TEMPORAL TRENDS
# -----------------------------
elif page == "Temporal Trends":
    st.subheader("📈 Temporal Trends")

    time_df = filtered_df.dropna(subset=["Date"]).copy() if "Date" in filtered_df.columns else pd.DataFrame()
    if time_df.empty:
        st.warning("Date data is not available. Check date/year/month columns.")
    else:
        freq = st.radio("Trend Level", ["Monthly", "Yearly"], horizontal=True)
        if freq == "Monthly":
            time_df["Period"] = time_df["Date"].dt.to_period("M").astype(str)
        else:
            time_df["Period"] = time_df["Date"].dt.year.astype(str)

        metrics = [c for c in ["wsi", "WSI_Score_Custom", "rainfall", "groundwater", "water_usage", "water_availability", "Rainfall_Deviation_Percent"] if c in time_df.columns]
        selected_metric = st.selectbox("Select Metric", metrics)

        trend = time_df.groupby("Period", as_index=False)[selected_metric].mean()
        fig = px.line(trend, x="Period", y=selected_metric, markers=True, title=f"{selected_metric} Trend")
        st.plotly_chart(fig, use_container_width=True)

        if "district" in time_df.columns:
            top_districts = time_df["district"].value_counts().head(8).index.tolist()
            compare = time_df[time_df["district"].isin(top_districts)].groupby(["Period", "district"], as_index=False)[selected_metric].mean()
            fig = px.line(compare, x="Period", y=selected_metric, color="district", markers=True, title=f"District-wise {selected_metric} Trend")
            st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# PAGE 4: GEO MAP
# -----------------------------
elif page == "Geo Map":
    st.subheader("🗺️ Telangana 33 District Geo Map")
    st.info("This page uses the full GeoJSON boundary file, so all 33 Telangana districts are shown even when sidebar district filters are changed.")

    # IMPORTANT: use main_df, not filtered_df, because filtered_df may contain only selected districts.
    full_ds = district_summary(main_df)

    # Choose metric safely.
    available_map_metrics = [
        c for c in ["WSI_Score_Custom", "wsi", "rainfall", "groundwater", "water_usage", "high_risk_flag"]
        if c in full_ds.columns
    ]

    if full_ds.empty or not available_map_metrics:
        st.warning("Map data is not available. Please check district and metric columns.")
    else:
        map_metric = st.selectbox("Map Metric", available_map_metrics)

        # Create normalized join key in data.
        map_df = full_ds.copy()
        map_df["district_key"] = map_df["district"].apply(normalize_text)

        # Prepare GeoJSON properties with same normalized key.
        if geo_data is not None and "features" in geo_data:
            geo_data_fixed = json.loads(json.dumps(geo_data))  # deep copy
            for feature in geo_data_fixed.get("features", []):
                props = feature.get("properties", {})
                gname = props.get("district_name") or props.get("Dist") or props.get("dtname")
                props["district_key"] = normalize_text(gname)
                props["display_district"] = gname

            geo_keys = {f.get("properties", {}).get("district_key", "") for f in geo_data_fixed.get("features", [])}
            data_keys = set(map_df["district_key"].dropna())

            # Keep all 33 GeoJSON districts, fill missing metric with 0 only for display.
            all_geo_districts = pd.DataFrame({"district_key": sorted(list(geo_keys))})
            map_df_all = all_geo_districts.merge(map_df, on="district_key", how="left")
            map_df_all["district"] = map_df_all["district"].fillna(map_df_all["district_key"].str.title())
            if map_metric in map_df_all.columns:
                map_df_all[map_metric] = map_df_all[map_metric].fillna(0)

            fig = px.choropleth_mapbox(
                map_df_all,
                geojson=geo_data_fixed,
                locations="district_key",
                featureidkey="properties.district_key",
                color=map_metric,
                hover_name="district",
                hover_data={map_metric: ":.2f", "district_key": False},
                mapbox_style="open-street-map",
                center={"lat": 17.9, "lon": 79.2},
                zoom=5.4,
                opacity=0.65,
                height=650,
                title=f"All 33 Telangana Districts by {map_metric}",
            )
            fig.update_layout(margin={"r": 0, "t": 45, "l": 0, "b": 0})
            st.plotly_chart(fig, use_container_width=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("GeoJSON Districts", len(geo_keys))
            c2.metric("Data Districts", len(data_keys))
            c3.metric("Matched Districts", len(data_keys.intersection(geo_keys)))

            missing_in_data = sorted(geo_keys - data_keys)
            missing_in_geo = sorted(data_keys - geo_keys)

            with st.expander("District matching details"):
                if missing_in_data:
                    st.warning("GeoJSON districts with no matching data. They are still shown on map with value 0:")
                    st.write(missing_in_data)
                if missing_in_geo:
                    st.warning("Data districts not matched with GeoJSON:")
                    st.write(missing_in_geo)
                if not missing_in_data and not missing_in_geo:
                    st.success("Perfect match: all 33 districts are matched.")

        else:
            st.warning("GeoJSON file not found. Showing point map using Lat/Long instead.")
            if {"Lat", "Long"}.issubset(map_df.columns):
                point_df = map_df.dropna(subset=["Lat", "Long"]).copy()
                fig = px.scatter_mapbox(
                    point_df,
                    lat="Lat",
                    lon="Long",
                    size=map_metric,
                    color=map_metric,
                    hover_name="district",
                    hover_data=[map_metric],
                    zoom=5.6,
                    height=650,
                    title=f"District Point Map by {map_metric}",
                )
                fig.update_layout(mapbox_style="open-street-map")
                fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})
                st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# PAGE 5: STRESS SIMULATOR
# -----------------------------
elif page == "Stress Simulator":
    st.subheader("🧪 Water Stress Simulator")
    st.markdown("Adjust rainfall, population, groundwater depth, and industry pressure to simulate WSI changes.")

    districts = sorted(main_df["district"].dropna().unique()) if "district" in main_df.columns else []
    selected_district = st.selectbox("Choose District", districts)
    base = main_df[main_df["district"] == selected_district].copy()

    if base.empty:
        st.warning("No data available for selected district.")
    else:
        base_wsi = safe_mean(base, "WSI_Score_Custom")
        if pd.isna(base_wsi):
            base_wsi = safe_mean(base, "wsi") * 100

        c1, c2, c3, c4 = st.columns(4)
        rainfall_change = c1.slider("Rainfall Change (%)", -50, 50, 0)
        population_change = c2.slider("Population Change (%)", -30, 50, 0)
        groundwater_change = c3.slider("Groundwater Depth Change (%)", -30, 50, 0)
        industry_change = c4.slider("Industry Pressure Change (%)", -30, 50, 0)

        # Logical WSI simulator:
        # More rainfall reduces stress, more population/groundwater depth/industry increases stress.
        simulated_wsi = base_wsi
        simulated_wsi += groundwater_change * 0.40
        simulated_wsi -= rainfall_change * 0.30
        simulated_wsi += population_change * 0.20
        simulated_wsi += industry_change * 0.10
        simulated_wsi = float(np.clip(simulated_wsi, 0, 100))

        base_category = classify_wsi(base_wsi)
        sim_category = classify_wsi(simulated_wsi)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Base WSI", format_num(base_wsi, 2))
        m2.metric("Simulated WSI", format_num(simulated_wsi, 2), delta=format_num(simulated_wsi - base_wsi, 2))
        m3.metric("Base Category", base_category)
        m4.metric("Simulated Category", sim_category)

        gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=simulated_wsi,
            delta={"reference": base_wsi},
            gauge={"axis": {"range": [0, 100]}},
            title={"text": f"Simulated WSI for {selected_district}"}
        ))
        st.plotly_chart(gauge, use_container_width=True)

        st.info(
            "Simulator formula: Base WSI + groundwater change × 40% - rainfall change × 30% + population change × 20% + industry change × 10%."
        )

# -----------------------------
# PAGE 6: STAR SCHEMA ANALYSIS
# -----------------------------
elif page == "Star Schema Analysis":
    st.subheader("⭐ Star Schema Dataset Analysis")

    if star_summary.empty:
        st.warning("Star schema files are not available or District_ID is missing.")
    else:
        st.dataframe(star_summary, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            if "Avg_Rainfall" in star_summary.columns:
                fig = px.bar(star_summary.sort_values("Avg_Rainfall", ascending=False).head(15), x="district", y="Avg_Rainfall", title="Average Rainfall by District")
                st.plotly_chart(fig, use_container_width=True)
        with c2:
            if "Avg_Groundwater_Depth" in star_summary.columns:
                fig = px.bar(star_summary.sort_values("Avg_Groundwater_Depth", ascending=False).head(15), x="district", y="Avg_Groundwater_Depth", title="Average Groundwater Depth by District")
                st.plotly_chart(fig, use_container_width=True)

        if "Reservoir_Fill_Percent" in star_summary.columns:
            fig = px.scatter(
                star_summary,
                x="Avg_Rainfall" if "Avg_Rainfall" in star_summary.columns else star_summary.index,
                y="Reservoir_Fill_Percent",
                size="Population_2023" if "Population_2023" in star_summary.columns else None,
                hover_name="district",
                title="Rainfall vs Reservoir Fill %"
            )
            st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# PAGE 7: AI INSIGHTS
# -----------------------------
elif page == "AI Insights":
    st.subheader("🤖 AI Insights: LLM-Based Problem & Solution Analysis")

    ds = district_summary(filtered_df)

    if ds.empty:
        st.warning("AI insights cannot be generated because district summary is empty.")
    else:
        score_col = "WSI_Score_Custom" if "WSI_Score_Custom" in ds.columns else "wsi"

        if score_col == "wsi":
            ds["AI_WSI"] = ds["wsi"] * 100
        else:
            ds["AI_WSI"] = ds[score_col]

        avg_wsi = ds["AI_WSI"].mean()
        highest = ds.sort_values("AI_WSI", ascending=False).iloc[0]
        top5 = ds.sort_values("AI_WSI", ascending=False).head(5)

        st.markdown("""
        <style>
        .ai-card {
            background: #ffffff;
            padding: 22px;
            border-radius: 18px;
            box-shadow: 0 4px 14px rgba(0,0,0,0.08);
            border-left: 7px solid #0d6efd;
            margin-bottom: 18px;
        }
        .risk-critical { border-left-color: #dc3545; }
        .risk-high { border-left-color: #fd7e14; }
        .risk-medium { border-left-color: #ffc107; }
        .risk-low { border-left-color: #198754; }
        .card-title {
            font-size: 22px;
            font-weight: 700;
            color: #1f2937;
            margin-bottom: 8px;
        }
        .card-small {
            font-size: 15px;
            color: #4b5563;
            margin-bottom: 6px;
        }
        .score-box {
            font-size: 32px;
            font-weight: 800;
            color: #0d6efd;
        }
        .tag {
            display: inline-block;
            padding: 5px 12px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 13px;
            background: #eef2ff;
            color: #3730a3;
        }
        </style>
        """, unsafe_allow_html=True)

        def get_ai_insight(prompt):
            try:
                co = cohere.ClientV2(st.secrets["COHERE_API_KEY"])

                response = co.chat(
                    model="command-a-03-2025",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a water resource analyst. Give clear, short, practical insights for Telangana government officials."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )

                return response.message.content[0].text

            except Exception as e:
                return f"AI insight could not be generated. Error: {e}"

        summary_prompt = f"""
        Analyze this Telangana water stress dashboard summary.

        Average WSI: {avg_wsi:.2f}
        Most affected district: {highest['district']}
        Highest WSI: {highest['AI_WSI']:.2f}

        Top 5 high-risk districts:
        {top5.to_string(index=False)}

        Give:
        1. Overall problem
        2. Main reasons
        3. Practical government-level solutions
        4. Priority action plan

        Keep it short and presentation-friendly.
        """

        with st.spinner("Generating AI insights using Cohere..."):
            overall_ai_text = get_ai_insight(summary_prompt)

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown(f"""
            <div class="ai-card">
                <div class="card-title">Most Affected District</div>
                <div class="score-box">{highest["district"]}</div>
                <div class="card-small">Highest water stress district</div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="ai-card">
                <div class="card-title">Highest WSI</div>
                <div class="score-box">{highest["AI_WSI"]:.2f}</div>
                <div class="card-small">Maximum district stress score</div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class="ai-card">
                <div class="card-title">Average WSI</div>
                <div class="score-box">{avg_wsi:.2f}</div>
                <div class="card-small">Overall Telangana average</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("## 📌 AI Generated Overall Insight")

        st.markdown(f"""
        <div class="ai-card risk-high">
            <div class="card-title">Generated Water Stress Analysis</div>
            <div class="card-small">{overall_ai_text.replace(chr(10), "<br>")}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("## 🚨 AI Generated District Priority Cards")

        for _, row in top5.iterrows():
            district_prompt = f"""
            Generate a short water stress insight for this district.

            District: {row['district']}
            WSI Score: {row['AI_WSI']:.2f}

            Available metrics:
            {row.to_string()}

            Give:
            - Risk level
            - Main problem
            - Recommended solution

            Keep it within 4 lines.
            """

            with st.spinner(f"Generating insight for {row['district']}..."):
                district_ai_text = get_ai_insight(district_prompt)

            if row["AI_WSI"] >= 75:
                risk_class = "critical"
            elif row["AI_WSI"] >= 50:
                risk_class = "high"
            elif row["AI_WSI"] >= 25:
                risk_class = "medium"
            else:
                risk_class = "low"

            st.markdown(f"""
            <div class="ai-card risk-{risk_class}">
                <div class="card-title">📍 {row["district"]}</div>
                <div class="card-small"><b>WSI Score:</b> {row["AI_WSI"]:.2f}</div>
                <div class="card-small">{district_ai_text.replace(chr(10), "<br>")}</div>
            </div>
            """, unsafe_allow_html=True)

# -----------------------------
# PAGE 8: RAW DATA & QUALITY CHECK
# -----------------------------
elif page == "Raw Data & Quality Check":
    st.subheader("🧾 Raw Data & Quality Check")

    tab1, tab2, tab3, tab4 = st.tabs(["Filtered Data", "Columns", "Missing Values", "Dataset Shapes"])

    with tab1:
        st.dataframe(filtered_df.head(1000), use_container_width=True)
        csv = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Filtered CSV", csv, "filtered_water_stress_data.csv", "text/csv")

    with tab2:
        col_info = pd.DataFrame({
            "Column": filtered_df.columns,
            "Data Type": [str(filtered_df[c].dtype) for c in filtered_df.columns],
            "Non Null Count": [filtered_df[c].notna().sum() for c in filtered_df.columns],
            "Unique Values": [filtered_df[c].nunique(dropna=True) for c in filtered_df.columns]
        })
        st.dataframe(col_info, use_container_width=True)

    with tab3:
        missing = filtered_df.isna().sum().reset_index()
        missing.columns = ["Column", "Missing Count"]
        missing["Missing %"] = (missing["Missing Count"] / len(filtered_df) * 100).round(2) if len(filtered_df) else 0
        st.dataframe(missing, use_container_width=True)

    with tab4:
        shapes = pd.DataFrame({
            "Dataset": ["Main", "Dim_Districts", "Dim_Date", "Fact_Rainfall", "Fact_Groundwater", "Fact_Reservoirs"],
            "Rows": [len(main_df), len(district_df), len(date_df), len(rainfall_df), len(groundwater_df), len(reservoir_df)],
            "Columns": [main_df.shape[1], district_df.shape[1], date_df.shape[1], rainfall_df.shape[1], groundwater_df.shape[1], reservoir_df.shape[1]],
        })
        st.dataframe(shapes, use_container_width=True)

st.sidebar.divider()
st.sidebar.success("App loaded successfully ✅")
