import streamlit as st
import pandas as pd
import pyarrow.dataset as ds
import os

# Streamlit App Config
st.set_page_config(page_title="News Search Engine", layout="wide")

st.title("📡 News Search Engine")
st.write("Search news by year, stock symbol, sentiment, or title keywords.")

DATA_PATH = r"D:\Columbia\Fall2025\5400\project\layer\gold_with_sentiment_sample"   # Folder containing many parquet files

# 1) Load Multiple Parquet Files (PyArrow)
@st.cache_data
def load_data(path):
    """Load many parquet files efficiently using PyArrow Dataset."""

    if not os.path.exists(path):
        st.error(f"❌ Path not found: {path}")
        st.stop()

    try:
        # If it's a folder → load all parquet inside it
        if os.path.isdir(path):
            dataset = ds.dataset(path, format="parquet")
            df = dataset.to_table().to_pandas()
        else:
            # Single parquet file
            df = pd.read_parquet(path)
    except Exception as e:
        st.error(f"❌ Failed to load parquet files: {e}")
        st.stop()

    # Extract Year column
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Year"] = df["Date"].dt.year.astype("Int64")

    return df


# Load dataset
df = load_data(DATA_PATH)

# 2) Sidebar Filters
st.sidebar.header("🔍 Search Filters")

# Year
if "Year" in df.columns:
    years = sorted(df["Year"].dropna().unique().tolist(), reverse=True)
    selected_year = st.sidebar.selectbox("Filter by Year", ["(All)"] + years)
else:
    selected_year = "(All)"

# Stock symbol
if "Stock_symbol" in df.columns:
    symbols = ["(All)"] + sorted(df["Stock_symbol"].dropna().unique().tolist())
    symbol = st.sidebar.selectbox("Filter by Stock Symbol", symbols)
else:
    symbol = "(All)"

# Sentiment range
if "sentiment_score_signed" in df.columns:
    min_score = float(df["sentiment_score_signed"].min())
    max_score = float(df["sentiment_score_signed"].max())
    sentiment_range = st.sidebar.slider(
        "Sentiment Score Range", min_score, max_score, (min_score, max_score)
    )
else:
    sentiment_range = None

# Keyword
keyword = st.sidebar.text_input("Keyword in Title (optional)")

# 3) Apply Filters
filtered = df.copy()

if selected_year != "(All)":
    filtered = filtered[filtered["Year"] == selected_year]

if symbol != "(All)":
    filtered = filtered[filtered["Stock_symbol"] == symbol]

if sentiment_range:
    filtered = filtered[
        (filtered["sentiment_score_signed"] >= sentiment_range[0]) &
        (filtered["sentiment_score_signed"] <= sentiment_range[1])
    ]

if keyword:
    filtered = filtered[
        filtered["Article_title"].str.contains(keyword, case=False, na=False)
    ]

# 4) Annual Sentiment Trend
st.subheader("📊 Annual Sentiment Trend Summary")

if "Year" in df.columns:
    trend = df.groupby("Year")["sentiment_score_signed"].mean()
    st.markdown("### 📈 Historical Average Sentiment Score")
    st.line_chart(trend)

st.markdown("---")

# 5) Summary for Selected Year
if selected_year != "(All)":
    year_df = df[df["Year"] == selected_year]
    title = f"Overall Sentiment Trend for **{selected_year}**"
else:
    year_df = df
    title = f"Overall Trend for **All Years**"

st.markdown(f"### {title}")

if "sentiment_label" in year_df.columns and len(year_df) > 0:
    counts = year_df["sentiment_label"].value_counts()
    total = counts.sum()

    col1, col2 = st.columns([1, 2])

    with col1:
        top_label = counts.index[0]
        pct = (counts.iloc[0] / total) * 100
        st.metric("Dominant Sentiment", top_label, f"{pct:.1f}%")

    with col2:
        summary_table = pd.DataFrame({
            "Sentiment": counts.index,
            "Count": counts.values,
            "Percent": (counts.values / total * 100).round(1),
        })
        st.dataframe(summary_table, hide_index=True)

st.markdown("---")

# 6) Search Results Table
st.subheader(f"🔎 Search Results ({filtered.shape[0]} rows)")

if "Date" in filtered.columns and pd.api.types.is_datetime64_any_dtype(filtered["Date"]):
    filtered = filtered.sort_values("Date", ascending=False)

cols = [
    "Year", "Date", "Stock_symbol", "news_id", "Article_title",
    "sentiment_label", "sentiment_score_signed", "avg_sentiment_score",
    "pos_ratio", "neg_ratio", "article_count", "publisher_count",
    "avg_title_len", "sentiment_category",
]
cols = [c for c in cols if c in filtered.columns]

st.data_editor(filtered[cols].head(200), use_container_width=True, hide_index=True)

st.markdown("---")

# 7) Article Detail Viewer
st.subheader("📰 View News Detail")

if filtered.shape[0] == 0:
    st.warning("No results. Adjust filters.")
else:
    selected_id = st.selectbox("Select news_id", filtered["news_id"].unique())
    row = filtered[filtered["news_id"] == selected_id].iloc[0]

    st.markdown("## 📝 Title")
    st.markdown(f"### {row['Article_title']}")

    st.markdown("---")
    st.markdown("## 🧠 Sentiment Metrics")

    metrics = [
        "sentiment_label", "sentiment_score_signed", "avg_sentiment_score",
        "pos_ratio", "neg_ratio", "sentiment_category",
    ]
    for m in metrics:
        if m in row:
            st.write(f"**{m}:** {row[m]}")

    st.markdown("---")
    st.markdown("## 📌 Metadata")

    meta = [
        "Year", "Date", "Stock_symbol", "Publisher", "Url",
        "sample_news_id", "article_count", "publisher_count", "avg_title_len",
    ]
    for m in meta:
        if m in row:
            st.write(f"**{m}:** {row[m]}")
