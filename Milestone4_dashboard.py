# -----------------------------------------------------------
# Milestone 4 - Weeks 7–8
# Streamlit Dashboard for Smart Traffic Violation Analyzer
# -----------------------------------------------------------

import streamlit as st
import pandas as pd
from pyspark.sql import SparkSession
import plotly.express as px
import plotly.graph_objects as go

# -------------------------
# Load Spark Only Once
# -------------------------
@st.cache_resource
def load_spark():
    return (
        SparkSession.builder
        .appName("M4-Dashboard")
        .getOrCreate()
    )

spark = load_spark()

# -------------------------
# Load Parquet Files (from Milestone 1–3)
# -------------------------
@st.cache_data
def load_data():
    df = spark.read.parquet("cleaned_traffic_data.parquet").toPandas()
    return df

df = load_data()

st.title("🚦 Smart Traffic Violation Pattern Detection Dashboard")

st.write("""
This interactive dashboard visualizes traffic violation patterns by  
*time, type, severity, and location*, as required in Milestone-4.
""")

# -------------------------------------------------
# Sidebar Controls (Filters)
# -------------------------------------------------
st.sidebar.header("🔍 Filters")

# Violation Type Filter
types = st.sidebar.multiselect(
    "Violation Type",
    df["Violation_Type"].unique().tolist(),
    default=df["Violation_Type"].unique().tolist()
)

# Severity Filter
severity_range = st.sidebar.slider(
    "Select Severity Range",
    1, 5, (1, 5)
)

# Date Filter
df["Date"] = pd.to_datetime(df["Timestamp"]).dt.date
min_date, max_date = df["Date"].min(), df["Date"].max()

date_range = st.sidebar.date_input(
    "Select Date Range",
    (min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Apply Filters
df_filtered = df[
    (df["Violation_Type"].isin(types)) &
    (df["Severity"].between(severity_range[0], severity_range[1])) &
    (df["Date"] >= date_range[0]) &
    (df["Date"] <= date_range[1])
]

# Extract Time Features
df_filtered["Hour"] = pd.to_datetime(df_filtered["Timestamp"]).dt.hour
df_filtered["DayOfWeek"] = pd.to_datetime(df_filtered["Timestamp"]).dt.day_name()

# ---------------------------------------------------------------------
# SECTION 1: Time-Based Visualizations
# ---------------------------------------------------------------------
st.header("⏳ Time-Based Violation Trends")

# Violations Per Hour
hour_counts = df_filtered.groupby("Hour").size().reset_index(name="Count")
fig_hour = px.bar(hour_counts, x="Hour", y="Count", title="Violations per Hour")
st.plotly_chart(fig_hour, use_container_width=True)

# Violations Per Day
day_counts = df_filtered.groupby("DayOfWeek").size().reset_index(name="Count")
fig_day = px.bar(day_counts, x="DayOfWeek", y="Count", title="Violations per Day")
st.plotly_chart(fig_day, use_container_width=True)

# ---------------------------------------------------------------------
# SECTION 2: Violation Type Distribution
# ---------------------------------------------------------------------
st.header("🚘 Violation Type Distribution")

type_counts = df_filtered["Violation_Type"].value_counts().reset_index()
type_counts.columns = ["Violation_Type", "Count"]

fig_type = px.pie(type_counts, values="Count", names="Violation_Type",
                  title="Violation Type Breakdown")
st.plotly_chart(fig_type, use_container_width=True)

# ---------------------------------------------------------------------
# SECTION 3: Top Locations
# ---------------------------------------------------------------------
st.header("📍 High-Risk Locations (Top N)")

top_n = st.slider("Select N", 3, 20, 5)
loc_counts = (
    df_filtered.groupby("Location").size().reset_index(name="Count")
    .sort_values(by="Count", ascending=False)
    .head(top_n)
)

fig_loc = px.bar(loc_counts, x="Location", y="Count",
                 title=f"Top {top_n} Locations with Most Violations")
st.plotly_chart(fig_loc, use_container_width=True)

st.dataframe(loc_counts)

# ---------------------------------------------------------------------
# SECTION 4: Optional Hotspot Chart (If Lat/Long Present)
# ---------------------------------------------------------------------
if "Latitude" in df.columns and "Longitude" in df.columns:
    st.header("🔥 Hotspot Map")
    fig_map = px.density_mapbox(
        df_filtered, lat="Latitude", lon="Longitude", radius=10,
        center=dict(lat=df["Latitude"].mean(), lon=df["Longitude"].mean()),
        zoom=10, mapbox_style="open-street-map"
    )
    st.plotly_chart(fig_map, use_container_width=True)

# ---------------------------------------------------------------------
# SECTION 5: Export Data (CSV / JSON)
# ---------------------------------------------------------------------
st.header("📤 Export Reports")

csv_data = df_filtered.to_csv(index=False).encode('utf-8')
json_data = df_filtered.to_json(orient="records").encode('utf-8')

st.download_button("Download CSV Report", csv_data, "violations_report.csv")
st.download_button("Download JSON Report", json_data, "violations_report.json")

st.success("Dashboard running successfully!")