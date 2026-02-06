import streamlit as st
import pandas as pd
import os

# Streamlit Config
st.set_page_config(page_title="News Search Engine", layout="wide")

st.title("📡 News Search Engine")
st.write("Search news by year, stock symbol, sentiment, or title keywords.")

# 1) Load CSV Automatically (Optimized with Caching)
# --- Configuration for Data Source ---
DEFAULT_FILENAME = "gold_with_sentiment_sample.csv"


# Use st.cache_data decorator to ensure data loading and processing run only once.
@st.cache_data
def load_data(filename):
    """Loads data from local CSV file and performs initial preprocessing."""

    df = None

    # --- 1. Load data from local CSV ---
    if os.path.exists(filename):
        try:
            df = pd.read_csv(filename)
        except Exception as csv_e:
            st.error(f"❌ Failed to read local CSV file: {csv_e}")
    else:
        st.error(f"❌ Local CSV file not found: '{filename}'")

    if df is None or df.empty:
        st.error("❌ Failed to load data. Please check local file path.")
        st.stop()

    # --- 2. Data Preprocessing: Extract Year (applies to all sources) ---
    if "Date" in df.columns:
        try:
            # Convert 'Date' to datetime objects
            df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
            # Extract the year, using 'Int64' to handle potential NaNs gracefully
            df["Year"] = df["Date"].dt.year.astype('Int64')
        except Exception as e:
            # Displaying an error if date processing fails within the cached function
            st.error(f"Error processing 'Date' column within cache: {e}. Please check the date format.")
            pass

    return df


# Load the data using the cached function
df = load_data(DEFAULT_FILENAME)

# 2) Sidebar Filters

st.sidebar.header("🔍 Search Filters")

# ---- Year Filter (Updated) ----
if "Year" in df.columns:
    # Get unique years and sort them in descending order
    all_years = sorted(df["Year"].dropna().unique().tolist(), reverse=True)
    selected_year = st.sidebar.selectbox(
        "Filter by Year",
        ["(All)"] + all_years
    )
else:
    selected_year = "(All)"

# ---- Stock Symbol Filter ----
if "Stock_symbol" in df.columns:
    symbols = ["(All)"] + sorted(df["Stock_symbol"].dropna().unique().tolist())
    symbol = st.sidebar.selectbox("Filter by Stock Symbol", symbols)
else:
    symbol = "(All)"

# ---- Sentiment Range Filter ----
if "sentiment_score_signed" in df.columns:
    min_score = float(df["sentiment_score_signed"].min())
    max_score = float(df["sentiment_score_signed"].max())

    # Ensure range works even if min/max are the same
    if min_score == max_score:
        sentiment_range = (min_score, max_score)
        st.sidebar.write(f"Sentiment Score Range: [{min_score:.2f}, {max_score:.2f}] (All values are the same)")
    else:
        sentiment_range = st.sidebar.slider(
            "Sentiment Score Range",
            min_score,
            max_score,
            (min_score, max_score)
        )
else:
    sentiment_range = None

# ---- Keyword Filter ----
keyword = st.sidebar.text_input("Keyword in Title (optional)")

# 3) Apply Filters
# The filtering operation itself is still run on every interaction, but now
# it works on the cached DataFrame, which is much faster than re-reading the CSV.
filtered = df.copy()

# Apply Year Filter (if selected)
if selected_year != "(All)" and "Year" in filtered.columns:
    filtered = filtered[filtered["Year"] == selected_year]

# Apply Stock Symbol Filter
if symbol != "(All)":
    filtered = filtered[filtered["Stock_symbol"] == symbol]

# Apply Sentiment Range Filter
if sentiment_range:
    filtered = filtered[
        (filtered["sentiment_score_signed"] >= sentiment_range[0]) &
        (filtered["sentiment_score_signed"] <= sentiment_range[1])
        ]

# Apply Keyword Filter
if keyword and "Article_title" in filtered.columns:
    filtered = filtered[
        filtered["Article_title"].str.contains(keyword, case=False, na=False)
    ]

# 4) Annual Trend Summary (NEW SECTION)
st.subheader("📊 Annual Sentiment Trend Summary")

# --- 4.1) Annual Sentiment Trend Chart (across all years) ---
if "Year" in df.columns and "sentiment_score_signed" in df.columns:
    st.markdown("### 📈 Historical Average Sentiment Score (All Years)")

    # Calculate average sentiment score per year from the full dataset (df)
    annual_trend = df.groupby("Year")["sentiment_score_signed"].mean().reset_index()
    annual_trend.rename(columns={"sentiment_score_signed": "Average Sentiment Score"}, inplace=True)

    # Set Year as index for better line chart display
    annual_trend.set_index("Year", inplace=True)

    st.line_chart(annual_trend, height=300)
    st.markdown("---")

# To provide an 'overall' trend for the selected year, we base the summary on the
# data filtered ONLY by the year, ignoring stock/score/keyword filters.
yearly_summary_df = df.copy()
if selected_year != "(All)" and "Year" in yearly_summary_df.columns:
    yearly_summary_df = yearly_summary_df[yearly_summary_df["Year"] == selected_year]
    summary_title = f"Overall Trend for **{selected_year}**"
elif selected_year == "(All)":
    summary_title = "Overall Trend for **All Years**"
else:
    summary_title = "Overall Trend"

st.markdown(f"### {summary_title}")

if "sentiment_label" in yearly_summary_df.columns:
    sentiment_counts = yearly_summary_df["sentiment_label"].value_counts().sort_values(ascending=False)
    total_articles = sentiment_counts.sum()

    if total_articles > 0:
        col1, col2 = st.columns([1, 2])

        # Col 1: Dominant Trend Highlight
        with col1:
            most_common_sentiment = sentiment_counts.index[0]
            most_common_count = sentiment_counts.iloc[0]
            most_common_percentage = (most_common_count / total_articles) * 100

            st.metric(
                label="Dominant Sentiment",
                value=most_common_sentiment.upper(),
                delta=f"{most_common_percentage:.1f}% of articles"
            )
            st.write(f"Total articles analyzed: {total_articles:,}")

        # Col 2: Detailed Breakdown
        with col2:
            summary_table = pd.DataFrame({
                "Sentiment": sentiment_counts.index,
                "Count": sentiment_counts.values,
                "Percentage": [f"{v / total_articles * 100:.1f}%" for v in sentiment_counts.values]
            })
            st.dataframe(summary_table, hide_index=True)

    else:
        st.warning("No data available to generate the annual summary.")
else:
    st.error("Cannot generate summary: 'sentiment_label' column is missing in the dataset.")

st.markdown("---")

# 5) Display Filtered Results
st.subheader(f"🔎 Search Results ({filtered.shape[0]} rows)")

# Sort by Date (Descending) to show newest articles first
if "Date" in filtered.columns:
    # Check if 'Date' is datetime type (should be due to load_data preprocessing)
    if pd.api.types.is_datetime64_any_dtype(filtered["Date"]):
        filtered = filtered.sort_values(by="Date", ascending=False)
    else:
        st.warning("Date column is not in datetime format for sorting.")

# show a subset of columns in table
cols_to_show = [
    col for col in [
        "Year",  # Added Year column
        "Date",
        "Stock_symbol",
        "news_id",
        "Article_title",
        "sentiment_label",
        "sentiment_score_signed",
        "avg_sentiment_score",
        "pos_ratio",
        "neg_ratio",
        "article_count",
        "publisher_count",
        "avg_title_len",
        "sentiment_category",
    ] if col in filtered.columns
]
# Use st.data_editor for better interactivity (sorting, searching)
st.data_editor(
    filtered[cols_to_show].head(200),
    use_container_width=True,
    hide_index=True
)

# 6) Select Article
st.markdown("---")
st.subheader("📰 View News Detail")

if filtered.shape[0] == 0:
    st.warning("No results. Please adjust the search filters!")
    # NOTE: Do not use st.stop() here as it stops the entire script, not just the section.
else:
    selected_id = st.selectbox("Select news_id", filtered["news_id"].unique())
    row = filtered[filtered["news_id"] == selected_id].iloc[0]

    # 7) Display Title & Basic Info
    st.markdown("## 📝 Title")

    if "Article_title" in df.columns:
        st.markdown(f"### {row['Article_title']}")
    else:
        st.write("No title column in this dataset.")

    st.markdown("---")
    st.markdown("## 🧠 Sentiment Metrics")

    if "sentiment_label" in df.columns:
        st.write(f"**Sentiment label:** {row['sentiment_label']}")
    if "sentiment_score_signed" in df.columns:
        st.write(f"**Signed sentiment score:** {row['sentiment_score_signed']}")
    if "avg_sentiment_score" in df.columns:
        st.write(f"**Avg sentiment (Date, Stock):** {row['avg_sentiment_score']}")
    if "pos_ratio" in df.columns and "neg_ratio" in df.columns:
        st.write(f"**Positive ratio:** {row['pos_ratio']}")
        st.write(f"**Negative ratio:** {row['neg_ratio']}")
    if "sentiment_category" in df.columns:
        st.write(f"**Sentiment category:** {row['sentiment_category']}")

    # 8) Metadata Section
    st.markdown("---")
    st.markdown("## 📌 Metadata")

    metadata_fields = [
        "Year",  # Added Year
        "Date",
        "Stock_symbol",
        "sample_news_id",
        "Publisher",
        "Url",
        "article_count",
        "publisher_count",
        "avg_title_len",
    ]

    for col in metadata_fields:
        if col in df.columns:
            st.write(f"**{col}:** {row[col]}")