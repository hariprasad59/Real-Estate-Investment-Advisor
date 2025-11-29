# app.py
"""
Multi-page Streamlit app (single file):
Page 1 — Intro / Landing
Page 2 — EDA
Page 3 — Filters & Search
Page 4 — Predict (single property)
Uses:
 - DATA_PATH: exact local CSV path you provided
 - Loads MLflow-registered sklearn models (Best_Regression, Best_Classifier)
 - No retraining
"""

import os
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import mlflow, mlflow.sklearn
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings("ignore")

# ----------------------
# CONFIG
# ----------------------
DATA_PATH = r"F:\python Proj\Projects\RealEstateInvestmentAdvisor\target_indian_house_price.csv"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:///F:/python Proj/Projects/RealEstateInvestmentAdvisor/mlruns")
REGRESSION_MODEL_NAME = "Best_Regression"
CLASSIFIER_MODEL_NAME = "Best_Classifier"
LOAD_PROD_STAGE = True

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# ----------------------
# PAGE STYLING (simple CSS)
# ----------------------
st.set_page_config(layout="wide", page_title="Real Estate Investment Advisor", page_icon="🏡")

st.markdown(
    """
    <style>
    .header {background: linear-gradient(90deg,#0f172a,#0b3a5b); padding:16px; border-radius:8px;}
    .big-title {color: white; font-size:32px; font-weight:700; margin-bottom:4px;}
    .sub {color: #cfe8ff; font-size:14px; margin-top:0;}
    .card {background: linear-gradient(180deg,#ffffff,#f6f9ff); padding:12px; border-radius:8px; box-shadow: 0 2px 8px rgba(15,23,42,0.06);}
    .accent {color:#0b74de; font-weight:700;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------
# Helpers: load data & models (cached)
# ----------------------
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found at: {path}")
    df = pd.read_csv(p)
    return df

def try_load_registered_model(name: str, stage="Production"):
    if LOAD_PROD_STAGE:
        try:
            return mlflow.sklearn.load_model(f"models:/{name}/{stage}")
        except Exception:
            pass
    try:
        return mlflow.sklearn.load_model(f"models:/{name}")
    except Exception:
        return None

def find_latest_local_artifact(mlruns_root="mlruns"):
    root = Path(mlruns_root)
    if not root.exists():
        return None
    candidates = list(root.rglob("artifacts/*"))
    candidates = [c for c in candidates if c.exists()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(candidates[0])

@st.cache_resource(show_spinner=False)
def load_models_with_fallback():
    reg = try_load_registered_model(REGRESSION_MODEL_NAME)
    clf = try_load_registered_model(CLASSIFIER_MODEL_NAME)
    if reg is None:
        local = find_latest_local_artifact(mlruns_root=MLFLOW_TRACKING_URI.replace("file:", "") if MLFLOW_TRACKING_URI.startswith("file:") else "mlruns")
        if local:
            try:
                reg = mlflow.sklearn.load_model(local)
            except Exception:
                reg = None
    if clf is None:
        local = find_latest_local_artifact(mlruns_root=MLFLOW_TRACKING_URI.replace("file:", "") if MLFLOW_TRACKING_URI.startswith("file:") else "mlruns")
        if local:
            try:
                clf = mlflow.sklearn.load_model(local)
            except Exception:
                clf = None
    return reg, clf

# small utility used by Predict page to prepare input like training did
def prepare_sample_from_inputs(inputs: dict, ref_df: pd.DataFrame) -> pd.DataFrame:
    exclude = {"ID","Future_Price_5Y","Good_Investment","growth_rate_city"}
    ref_cols = [c for c in ref_df.columns if c not in exclude]
    sample = pd.DataFrame([{c: np.nan for c in ref_cols}])
    for k,v in inputs.items():
        if k in sample.columns:
            sample.at[0,k] = v
        else:
            sample[k] = v
    # Price_per_SqFt
    if ("Price_per_SqFt" not in sample.columns) or pd.isna(sample.at[0,"Price_per_SqFt"]):
        if "Price_in_Lakhs" in sample.columns and "Size_in_SqFt" in sample.columns:
            p,s = sample.at[0,"Price_in_Lakhs"], sample.at[0,"Size_in_SqFt"]
            try:
                sample.at[0,"Price_per_SqFt"] = float(p)/float(s) if pd.notna(p) and pd.notna(s) and s!=0 else np.nan
            except Exception:
                sample.at[0,"Price_per_SqFt"] = np.nan
    # Amenity_Count
    if "Amenity_Count" in ref_cols:
        if "Amenities" in sample.columns and pd.notna(sample.at[0,"Amenities"]):
            a = sample.at[0,"Amenities"]
            sample.at[0,"Amenity_Count"] = len([s for s in str(a).split(",") if s.strip()!=""])
        else:
            sample.at[0,"Amenity_Count"] = 0
    # growth_rate_city fallback
    if "growth_rate_city" in ref_cols:
        if "City" in sample.columns and pd.notna(sample.at[0,"City"]):
            city_growth = {"Bangalore":0.10,"Hyderabad":0.12,"Chennai":0.08,"Pune":0.09,"Mumbai":0.07,"Delhi":0.06}
            sample.at[0,"growth_rate_city"] = city_growth.get(sample.at[0,"City"],0.08)
        else:
            sample.at[0,"growth_rate_city"] = 0.08
    # numeric dtype enforcement and fill medians/modes
    numeric_cols = ref_df.select_dtypes(include=[np.number]).columns.tolist()
    for c in sample.columns:
        if c in numeric_cols:
            sample[c] = pd.to_numeric(sample[c], errors="coerce")
    for c in sample.columns:
        if pd.isna(sample.at[0,c]):
            if c in numeric_cols and c in ref_df.columns:
                sample.at[0,c] = ref_df[c].median()
            elif c in ref_df.columns:
                mode = ref_df[c].mode()
                sample.at[0,c] = mode.iloc[0] if len(mode)>0 else ""
            else:
                sample.at[0,c] = "" if sample[c].dtype==object else 0
    sample = sample[[c for c in ref_cols if c in sample.columns]]
    return sample

# ----------------------
# Load dataset & models
# ----------------------
try:
    df = load_data(DATA_PATH)
except Exception as e:
    st.error(f"Could not load data: {e}")
    st.stop()

reg_model, clf_model = load_models_with_fallback()

# ----------------------
# Navigation (pages) — query-param aware (non-experimental)
# ----------------------
params = st.query_params  # non-experimental getter
# if ?page=... present in URL, use that; otherwise show sidebar radio as fallback
if "page" in params:
    # params values are lists; take first
    page = params["page"][0]
else:
    page = st.sidebar.radio("Navigate", ["Home", "EDA", "Filters & Search", "Predict"], index=0)




# ----------------------
# Page: Home / Intro (REPLACED: attractive landing page)
# ----------------------
if page == "Home":
    # Hero area with gradient and left-side vertical CTAs
    hero_col, stats_col = st.columns([3,1])
    with hero_col:
        st.markdown(
            """
            <div style="background:linear-gradient(90deg,#0b3a5b,#0f172a); padding:28px; border-radius:12px; color:white;">
              <h1 style="margin:0 0 6px 0; font-size:34px;">🏡 Real Estate Investment Advisor</h1>
              <p style="margin:0 0 12px 0; color:#d6e9ff; font-size:15px;">
                Explore, filter and get instant predictions — use the Predict page to estimate 5-year prices and investment suitability.
              </p>
              <div style="display:flex; gap:10px; margin-top:12px;">
                <a href="?page=EDA" style="text-decoration:none;">
                  <div style="background:#ffffff10; color:white; padding:10px 18px; border-radius:8px; display:inline-block;">📊 Explore Data</div>
                </a>
                <a href="?page=Filters & Search" style="text-decoration:none;">
                  <div style="background:#ffffff10; color:white; padding:10px 18px; border-radius:8px; display:inline-block;">🔎 Filter & Search</div>
                </a>
                <a href="?page=Predict" style="text-decoration:none;">
                  <div style="background:#ffdd57; color:#062b4f; padding:10px 18px; border-radius:8px; display:inline-block; font-weight:700;">Predict a Property →</div>
                </a>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br/>", unsafe_allow_html=True)

        # Quick search box (simple and friendly)
        st.markdown("### Quick search")
        q = st.text_input("Search Locality or Property Type (press Enter)", placeholder="e.g. Koramangala, Apartment")
        if q:
            subset = df[df.apply(lambda r: q.lower() in str(r.get("Locality","")).lower() or q.lower() in str(r.get("Property_Type","")).lower(), axis=1)]
            st.markdown(f"**{len(subset):,} results** — showing top 8 matches")
            st.dataframe(subset.head(8), use_container_width=True)

        st.markdown("<br/>", unsafe_allow_html=True)

        # Featured mini-chart: top 8 cities by avg price
        if 'City' in df.columns and 'Price_in_Lakhs' in df.columns:
            top8 = df.groupby("City")["Price_in_Lakhs"].mean().reset_index().sort_values("Price_in_Lakhs", ascending=False).head(8)
            fig = px.bar(top8, x="City", y="Price_in_Lakhs", title="Top 8 cities — avg price (lakhs)", template="plotly_white")
            fig.update_layout(margin=dict(l=0,r=0,t=30,b=0), height=260)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("<br/>", unsafe_allow_html=True)

        # Feature cards: short blurbs
        c1, c2, c3 = st.columns(3)
        c1.markdown("<div style='padding:12px; border-radius:8px; background:linear-gradient(180deg,#ffffff,#f2fbff);'><b>Why use this app?</b><br/>Quick, explainable estimates & easy filters</div>", unsafe_allow_html=True)
        c2.markdown("<div style='padding:12px; border-radius:8px; background:linear-gradient(180deg,#ffffff,#fff8e6);'><b>Compare</b><br/>See both model estimate & formula baseline</div>", unsafe_allow_html=True)
        c3.markdown("<div style='padding:12px; border-radius:8px; background:linear-gradient(180deg,#ffffff,#f6fff4);'><b>Download data</b><br/>Export filtered slices</div>", unsafe_allow_html=True)

    with stats_col:
        # right column - attractive KPI cards
        st.markdown("<div style='padding:6px; border-radius:8px; background:linear-gradient(180deg,#ffffff,#eef6ff);'>", unsafe_allow_html=True)
        st.markdown("<h3 style='margin:2px 0 6px 0;'>Snapshot</h3>", unsafe_allow_html=True)
        # compact stats (no raw MLflow details)
        st.metric("Properties", f"{df.shape[0]:,}")
        if "Price_in_Lakhs" in df.columns:
            st.metric("Median Price (lakhs)", f"{df['Price_in_Lakhs'].median():.1f}")
        if "Price_per_SqFt" in df.columns:
            st.metric("Median PpSqFt", f"{df['Price_per_SqFt'].median():.2f}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br/>", unsafe_allow_html=True)

    # small footer area
    st.markdown("---")
    st.markdown("<div style='color:#6b7280;'>Tip: Use <b>Predict</b> to enter property details and compare model vs formula estimates.</div>", unsafe_allow_html=True)


# ----------------------
# Page: EDA (REPLACEMENT: dropdown with 20 items)
# ----------------------
elif page == "EDA":
    st.header("Exploratory Data Analysis — Quick checks")
    st.markdown("Choose an analysis from the dropdown to display the plot and summary statistics.")

    # list of EDA options
    eda_options = [
        "1. Distribution of property prices",
        "2. Distribution of property sizes",
        "3. Price per SqFt by property type (boxplot)",
        "4. Relationship between property size and price (scatter)",
        "5. Outliers in Price_per_SqFt and Size_in_SqFt",
        "6. Average Price_per_SqFt by State",
        "7. Average property price by City (top 20)",
        "8. Median age of properties by Locality (top 20 young)",
        "9. BHK distribution across top cities (stacked)",
        "10. Price trends for top 5 most expensive localities",
        "11. Correlation matrix of numeric features",
        "12. Nearby schools vs Price_per_SqFt",
        "13. Nearby hospitals vs Price_per_SqFt",
        "14. Price by Furnished_Status",
        "15. Price variation by Facing direction",
        "16. Counts by Owner_Type",
        "17. Counts by Availability_Status",
        "18. Parking space vs Price_in_Lakhs",
        "19. Amenities (Amenity_Count) vs Price_per_SqFt",
        "20. Public transport accessibility vs Price_per_SqFt"
    ]

    choice = st.selectbox("Select EDA check", options=eda_options)

    # helper functions
    def safe_show_fig(fig):
        """Show either plotly or matplotlib figures."""
        if hasattr(fig, "to_html") or isinstance(fig, (px.Figure,)):
            try:
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.write(fig)
        else:
            st.pyplot(fig)

    # Precompute some common series/values to reuse
    data = df  # existing df variable from app
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

    # 1. Distribution of property prices
    if choice.startswith("1"):
        if "Price_in_Lakhs" in data.columns:
            price = data["Price_in_Lakhs"].dropna()
            st.write(price.describe())
            fig = px.histogram(price, nbins=60, title="Distribution of Property Prices (Lakhs)")
            fig.update_layout(xaxis_title="Price (Lakhs)", yaxis_title="Count", height=420)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Column Price_in_Lakhs not found in dataset.")

    # 2. Distribution of property sizes
    elif choice.startswith("2"):
        if "Size_in_SqFt" in data.columns:
            size = data["Size_in_SqFt"].dropna()
            st.write(size.describe())
            fig = px.histogram(size, nbins=60, title="Distribution of Property Sizes (SqFt)")
            fig.update_layout(xaxis_title="Size (SqFt)", yaxis_title="Count", height=420)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Column Size_in_SqFt not found in dataset.")

    # 3. Price per SqFt by property type (boxplot top 12 by count)
    elif choice.startswith("3"):
        if "Property_Type" in data.columns and "Price_per_SqFt" in data.columns:
            top_by_count = data["Property_Type"].value_counts().head(12).index.tolist()
            subset = data[data["Property_Type"].isin(top_by_count)]
            # use plotly box
            fig = px.box(subset, x="Price_per_SqFt", y="Property_Type", orientation="h",
                         category_orders={"Property_Type": top_by_count},
                         title="Price per SqFt by Property Type (top 12 by count)", points="outliers")
            fig.update_layout(height=520)
            st.plotly_chart(fig, use_container_width=True)
            st.write(subset.groupby("Property_Type")["Price_per_SqFt"].agg(["count","median","mean"]).sort_values("median", ascending=False))
        else:
            st.info("Columns Property_Type or Price_per_SqFt missing.")

    # 4. Relationship between property size and price
    elif choice.startswith("4"):
        if "Size_in_SqFt" in data.columns and "Price_in_Lakhs" in data.columns:
            x = data["Size_in_SqFt"]
            y = data["Price_in_Lakhs"]
            corr = x.corr(y)
            st.write(f"Correlation between Size and Price: {corr:.3f}")
            fig, ax = plt.subplots(figsize=(8,5))
            ax.scatter(x, y, alpha=0.35, s=10)
            ax.set_xlabel("Size (SqFt)"); ax.set_ylabel("Price (Lakhs)")
            ax.set_title(f"Size vs Price (corr={corr:.3f})")
            ax.grid(True)
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Columns Size_in_SqFt or Price_in_Lakhs missing.")

    # 5. Outliers in price per sq ft and property size
    elif choice.startswith("5"):
        cols = []
        if "Price_per_SqFt" in data.columns:
            pp = data["Price_per_SqFt"].dropna()
            q1,q3 = pp.quantile(0.25), pp.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
            out_pp = data[(data["Price_per_SqFt"] < lower) | (data["Price_per_SqFt"] > upper)]
            st.write("Outliers in Price_per_SqFt:", len(out_pp))
            fig1, ax1 = plt.subplots(figsize=(6,3))
            ax1.boxplot(pp, vert=True)
            ax1.set_title("Boxplot: Price_per_SqFt")
            st.pyplot(fig1)
            plt.close(fig1)
            cols.append(("Price_per_SqFt", len(out_pp)))
        if "Size_in_SqFt" in data.columns:
            sz = data["Size_in_SqFt"].dropna()
            q1s,q3s = sz.quantile(0.25), sz.quantile(0.75)
            iqrs = q3s - q1s
            lower_s, upper_s = q1s - 1.5*iqrs, q3s + 1.5*iqrs
            out_sz = data[(data["Size_in_SqFt"] < lower_s) | (data["Size_in_SqFt"] > upper_s)]
            st.write("Outliers in Size_in_SqFt:", len(out_sz))
            fig2, ax2 = plt.subplots(figsize=(6,3))
            ax2.boxplot(sz, vert=True)
            ax2.set_title("Boxplot: Size_in_SqFt")
            st.pyplot(fig2)
            plt.close(fig2)
            cols.append(("Size_in_SqFt", len(out_sz)))
        if not cols:
            st.info("Price_per_SqFt and Size_in_SqFt not present.")

    # 6. Average price per sq ft by state
    elif choice.startswith("6"):
        if "State" in data.columns and "Price_per_SqFt" in data.columns:
            avg_price = data.groupby("State")["Price_per_SqFt"].mean().sort_values(ascending=False)
            st.write(avg_price)
            fig = px.bar(avg_price.reset_index(), x="Price_per_SqFt", y="State", orientation="h", title="Average Price_per_SqFt by State")
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Columns State or Price_per_SqFt missing.")

    # 7. Average property price by City (top 20)
    elif choice.startswith("7"):
        if "City" in data.columns and "Price_in_Lakhs" in data.columns:
            avg_city = data.groupby("City")["Price_in_Lakhs"].mean().sort_values(ascending=False).head(20)
            st.write(avg_city)
            fig = px.bar(avg_city.reset_index(), x="Price_in_Lakhs", y="City", orientation="h", title="Top 20 Cities by Average Price (Lakhs)")
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Columns City or Price_in_Lakhs missing.")

    # 8. Median age of properties by locality (top 20 young)
    elif choice.startswith("8"):
        if "Locality" in data.columns and "Age_of_Property" in data.columns:
            med_age = data.groupby("Locality")["Age_of_Property"].median().sort_values()
            st.write(med_age.head(20))
            top20 = med_age.head(20)
            fig = px.bar(top20.reset_index(), x="Age_of_Property", y="Locality", orientation="h", title="Localities with Youngest Median Age")
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Columns Locality or Age_of_Property missing.")

    # 9. BHK distribution across top cities
    elif choice.startswith("9"):
        if "City" in data.columns and "BHK" in data.columns:
            bhk_city = data.groupby(["City","BHK"]).size().unstack(fill_value=0)
            top10 = data["City"].value_counts().head(10).index
            bhk_top = bhk_city.loc[bhk_city.index.isin(top10)]
            fig = bhk_top.plot(kind="bar", stacked=True, figsize=(12,5)).get_figure()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Columns City or BHK missing.")

    # 10. Price trends for top 5 most expensive localities
    elif choice.startswith("10"):
        if "Locality" in data.columns and "Price_in_Lakhs" in data.columns and "Size_in_SqFt" in data.columns:
            top5 = data.groupby("Locality")["Price_in_Lakhs"].median().sort_values(ascending=False).head(5).index
            fig, ax = plt.subplots(figsize=(10,5))
            for loc in top5:
                subset = data[data["Locality"]==loc].sort_values("Size_in_SqFt")
                if len(subset) < 5:
                    continue
                rolling = subset["Price_in_Lakhs"].rolling(window=200, min_periods=5).median()
                ax.plot(subset["Size_in_SqFt"], rolling, label=loc)
            ax.set_xlabel("Size (SqFt)"); ax.set_ylabel("Price (Lakhs)")
            ax.set_title("Price trend (rolling median) for Top 5 localities")
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Locality, Price_in_Lakhs or Size_in_SqFt missing.")

    # 11. Correlation matrix of numeric features
    elif choice.startswith("11"):
        if len(numeric_cols) >= 2:
            corr = data[numeric_cols].corr()
            fig = px.imshow(corr, title="Correlation matrix (numeric features)", labels=dict(x="", y=""))
            st.plotly_chart(fig, use_container_width=True)
            st.write(corr)
        else:
            st.info("Not enough numeric columns to compute correlation.")

    # 12. Nearby schools vs Price_per_SqFt
    elif choice.startswith("12"):
        if "Nearby_Schools" in data.columns and "Price_per_SqFt" in data.columns:
            grouped = data.groupby("Nearby_Schools")["Price_per_SqFt"].median()
            st.write(grouped.head(20))
            fig = px.line(grouped.reset_index(), x="Nearby_Schools", y="Price_per_SqFt", markers=True, title="Nearby Schools vs Median Price_per_SqFt")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Columns Nearby_Schools or Price_per_SqFt missing.")

    # 13. Nearby hospitals vs Price_per_SqFt
    elif choice.startswith("13"):
        if "Nearby_Hospitals" in data.columns and "Price_per_SqFt" in data.columns:
            grouped = data.groupby("Nearby_Hospitals")["Price_per_SqFt"].median()
            st.write(grouped.head(20))
            fig = px.line(grouped.reset_index(), x="Nearby_Hospitals", y="Price_per_SqFt", markers=True, title="Nearby Hospitals vs Median Price_per_SqFt")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Columns Nearby_Hospitals or Price_per_SqFt missing.")

    # 14. Price by Furnished_Status
    elif choice.startswith("14"):
        if "Furnished_Status" in data.columns and "Price_in_Lakhs" in data.columns:
            grouped = data.groupby("Furnished_Status")["Price_in_Lakhs"].median().sort_values()
            st.write(grouped)
            fig = px.bar(grouped.reset_index(), x="Furnished_Status", y="Price_in_Lakhs", title="Median Price by Furnished Status")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Columns Furnished_Status or Price_in_Lakhs missing.")

    # 15. Price variation by Facing direction
    elif choice.startswith("15"):
        if "Facing" in data.columns and "Price_in_Lakhs" in data.columns:
            grouped = data.groupby("Facing")["Price_in_Lakhs"].median().sort_values()
            st.write(grouped)
            fig = px.bar(grouped.reset_index(), x="Facing", y="Price_in_Lakhs", title="Median Price by Facing")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Columns Facing or Price_in_Lakhs missing.")

    # 16. Counts by Owner_Type
    elif choice.startswith("16"):
        if "Owner_Type" in data.columns:
            counts = data["Owner_Type"].value_counts()
            st.write(counts)
            fig = px.bar(counts.reset_index(), x="index", y="Owner_Type", title="Number of Properties by Owner Type")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Column Owner_Type missing.")

    # 17. Counts by Availability_Status
    elif choice.startswith("17"):
        if "Availability_Status" in data.columns:
            counts = data["Availability_Status"].value_counts()
            st.write(counts)
            fig = px.bar(counts.reset_index(), x="index", y="Availability_Status", title="Properties by Availability Status")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Column Availability_Status missing.")

    # 18. Parking space vs Price_in_Lakhs
    elif choice.startswith("18"):
        if "Parking_Space" in data.columns and "Price_in_Lakhs" in data.columns:
            grouped = data.groupby("Parking_Space")["Price_in_Lakhs"].median().sort_values()
            st.write(grouped)
            fig = px.bar(grouped.reset_index(), x="Parking_Space", y="Price_in_Lakhs", title="Parking Space vs Median Price (Lakhs)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Columns Parking_Space or Price_in_Lakhs missing.")

    # 19. Amenities affect Price_per_SqFt (Amenity_Count)
    elif choice.startswith("19"):
        if "Amenities" in data.columns:
            # ensure Amenity_Count exists
            def amenity_count(x):
                if isinstance(x, str):
                    x = x.replace("|", ",")
                    return len([a.strip() for a in x.split(",") if a.strip() != ""])
                return 0
            if "Amenity_Count" not in data.columns:
                data = data.copy()
                data["Amenity_Count"] = data["Amenities"].apply(amenity_count)
            grouped = data.groupby("Amenity_Count")["Price_per_SqFt"].median().sort_index()
            st.write(grouped.head(30))
            fig = px.line(grouped.reset_index(), x="Amenity_Count", y="Price_per_SqFt", markers=True, title="Amenity Count vs Median Price_per_SqFt")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Column Amenities missing.")

    # 20. Public transport accessibility vs Price_per_SqFt
    elif choice.startswith("20"):
        if "Public_Transport_Accessibility" in data.columns and "Price_per_SqFt" in data.columns:
            grouped = data.groupby("Public_Transport_Accessibility")["Price_per_SqFt"].median().sort_values()
            st.write(grouped)
            fig = px.bar(grouped.reset_index(), x="Public_Transport_Accessibility", y="Price_per_SqFt", title="Public Transport Accessibility vs Median Price_per_SqFt")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Columns Public_Transport_Accessibility or Price_per_SqFt missing.")

    else:
        st.info("Select an EDA check from the dropdown.")

    st.markdown("---")
    st.caption("Tip: use the Filters & Search page to create slices and then come back to EDA for focused analysis on the slice.")


# ----------------------
# Page: Filters & Search
# ----------------------
elif page == "Filters & Search":
    st.header("Interactive Filters & Search")
    left, right = st.columns([3,1])
    with left:
        st.markdown("Use filters to slice the dataset and download the result.")
        # filters
        city_opts = ["All"] + sorted(df['City'].dropna().unique().tolist()) if 'City' in df.columns else ["All"]
        city_sel = st.selectbox("City", city_opts)
        prop_types = ["All"] + sorted(df['Property_Type'].dropna().unique().tolist()) if 'Property_Type' in df.columns else ["All"]
        prop_sel = st.selectbox("Property_Type", prop_types)
        bhk_min = int(df['BHK'].min()) if 'BHK' in df.columns else 1
        bhk_max = int(df['BHK'].max()) if 'BHK' in df.columns else bhk_min+3
        bhk_range = st.slider("BHK range", bhk_min, bhk_max, (bhk_min, min(bhk_min+1, bhk_max)))
        price_min = float(df['Price_in_Lakhs'].min()) if 'Price_in_Lakhs' in df.columns else 0.0
        price_max = float(df['Price_in_Lakhs'].max()) if 'Price_in_Lakhs' in df.columns else 1000.0
        price_range = st.slider("Price (lakhs)", price_min, price_max, (price_min, price_max))

        # search box for locality
        search_locality = st.text_input("Search Locality (partial match)", "")

        # apply filters
        subset = df.copy()
        if city_sel != "All":
            subset = subset[subset['City'] == city_sel]
        if prop_sel != "All":
            subset = subset[subset['Property_Type'] == prop_sel]
        subset = subset[(subset['BHK'] >= bhk_range[0]) & (subset['BHK'] <= bhk_range[1])]
        subset = subset[(subset['Price_in_Lakhs'] >= price_range[0]) & (subset['Price_in_Lakhs'] <= price_range[1])]
        if search_locality:
            subset = subset[subset['Locality'].astype(str).str.contains(search_locality, case=False, na=False)]

        st.markdown(f"### Results: {len(subset):,} rows")
        st.dataframe(subset.head(200))
        st.download_button("Download filtered CSV", subset.to_csv(index=False).encode('utf-8'), file_name='filtered_india_housing.csv')
    with right:
        st.markdown("Quick stats on the slice")
        if len(subset)>0:
            st.metric("Median Price (lakhs)", f"{subset['Price_in_Lakhs'].median():.2f}")
            st.metric("Median Price_per_SqFt", f"{subset['Price_per_SqFt'].median():.2f}" if 'Price_per_SqFt' in subset.columns else "n/a")
            if 'Good_Investment' in subset.columns:
                st.metric("Good Investment %", f"{subset['Good_Investment'].mean()*100:.1f}%")
        else:
            st.info("No rows in the current slice.")

# ----------------------
# Page: Predict
# ----------------------
else:  # Predict page
    st.header("Predict — single property (no retrain)")
    st.markdown("Enter property details on the left, models will predict 5-year price and whether it is a Good Investment.")

    left, right = st.columns([2,1])
    with left:
        with st.form("predict"):
            city_vals = sorted(df['City'].dropna().unique().tolist()) if 'City' in df.columns else []
            city = st.selectbox("City", [""] + city_vals)
            locality = st.text_input("Locality (optional)")
            prop_type = st.selectbox("Property_Type", [""] + sorted(df['Property_Type'].dropna().unique().tolist()) if 'Property_Type' in df.columns else [""])
            bhk = st.number_input("BHK", int(df['BHK'].min()) if 'BHK' in df.columns else 1, int(df['BHK'].max()) if 'BHK' in df.columns else 5, int(df['BHK'].median()) if 'BHK' in df.columns else 2)
            size = st.number_input("Size_in_SqFt", int(df['Size_in_SqFt'].min()) if 'Size_in_SqFt' in df.columns else 500, int(df['Size_in_SqFt'].max()) if 'Size_in_SqFt' in df.columns else 2000, int(df['Size_in_SqFt'].median()) if 'Size_in_SqFt' in df.columns else 800)
            price = st.number_input("Price_in_Lakhs", float(df['Price_in_Lakhs'].min()) if 'Price_in_Lakhs' in df.columns else 10.0, float(df['Price_in_Lakhs'].max()) if 'Price_in_Lakhs' in df.columns else 1000.0, float(df['Price_in_Lakhs'].median()) if 'Price_in_Lakhs' in df.columns else 50.0)
            amenities = st.text_area("Amenities (comma separated)", "")
            owner = st.selectbox("Owner_Type", [""] + sorted(df['Owner_Type'].dropna().unique().tolist()) if 'Owner_Type' in df.columns else [""])
            avail = st.selectbox("Availability_Status", [""] + sorted(df['Availability_Status'].dropna().unique().tolist()) if 'Availability_Status' in df.columns else [""])
            submitted = st.form_submit_button("Run prediction")

    with right:
        st.markdown("Model info & tips")
        st.write(f"Regression loaded: **{'Yes' if reg_model is not None else 'No'}**")
        st.write(f"Classifier loaded: **{'Yes' if clf_model is not None else 'No'}**")
        st.info("Tip: use values from the Filters page to ensure realistic inputs.")

    if submitted:
        user_inputs = {
            "City": city or np.nan,
            "Locality": locality or "",
            "Property_Type": prop_type or "",
            "BHK": bhk,
            "Size_in_SqFt": size,
            "Price_in_Lakhs": price,
            "Amenities": amenities or "",
            "Owner_Type": owner or "",
            "Availability_Status": avail or ""
        }
        sample_df = prepare_sample_from_inputs(user_inputs, df)
        st.subheader("Prepared input (features used by model)")
        st.dataframe(sample_df.T)

        # Regression predict
        est_price_5y = None
        if reg_model is not None:
            try:
                est_price_5y = float(reg_model.predict(sample_df)[0])
            except Exception as e:
                st.error(f"Regression predict failed: {e}")
        else:
            base_r = 0.08
            est_price_5y = float(sample_df['Price_in_Lakhs'].iloc[0]) * (1 + base_r) ** 5

        st.metric("Estimated Price in 5 years (lakhs)", f"{est_price_5y:.2f}")
        # show growth %
        try:
            current = float(sample_df['Price_in_Lakhs'].iloc[0])
            pct = (est_price_5y - current) / current * 100
            st.write(f"Projected change: **{pct:.1f}%** over 5 years  •  Annualized ≈ **{((est_price_5y/current)**(1/5)-1)*100:.2f}%**")
        except Exception:
            pass

        # Classify
        pred = None
        proba = None
        if clf_model is not None:
            try:
                pred = int(clf_model.predict(sample_df)[0])
                if hasattr(clf_model, "predict_proba"):
                    proba = float(clf_model.predict_proba(sample_df)[0][1])
            except Exception as e:
                st.error(f"Classifier predict failed: {e}")

        if pred is not None:
            st.metric("Good Investment?", "Yes" if pred==1 else "No", delta=f"Confidence: {proba:.2f}" if proba is not None else "")
        else:
            st.info("Classifier not available; only regression shown.")

        # Small local explanation: permutation importance (sample)
        if clf_model is not None and "Good_Investment" in df.columns:
            with st.expander("Local feature importance (permutation sample)"):
                try:
                    small = df.sample(min(2000, len(df)), random_state=1).copy()
                    X_perm = small.drop(columns=["Good_Investment","Future_Price_5Y"], errors="ignore")
                    y_perm = small["Good_Investment"]
                    perm = permutation_importance(clf_model, X_perm, y_perm, n_repeats=6, random_state=1, n_jobs=-1)
                    imp = pd.Series(perm.importances_mean, index=X_perm.columns).sort_values(ascending=False).head(12)
                    fig, ax = plt.subplots(figsize=(8,5))
                    sns.barplot(x=imp.values, y=imp.index, ax=ax)
                    ax.set_title("Top features (permutation importance)")
                    st.pyplot(fig)
                except Exception as e:
                    st.write("Could not compute permutation importance:", e)
        else:
            st.info("Permutation importance requires classifier & Good_Investment label in dataset.")

    st.markdown("---")
    st.caption("Notes: This app uses MLflow-registered sklearn models (if present). It prepares inputs to match training features, computes derived features like Amenity_Count and Price_per_SqFt, and shows predictions (no retrain).")

