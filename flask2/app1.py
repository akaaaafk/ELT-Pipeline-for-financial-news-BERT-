from flask import Flask, render_template, request
import pandas as pd
import pyarrow.dataset as ds
import os
import time

app = Flask(__name__)

DATA_PATH = r"D:\Columbia\Fall2025\5400\project\layer\gold_with_sentiment_sample"

# 1) Load Parquet Folder (PyArrow Dataset)
def load_data(path: str) -> pd.DataFrame:
    """Load either a folder of parquet files or a single parquet/csv."""

    if not os.path.exists(path):
        raise FileNotFoundError(f"Data path not found: {path}")

    try:
        if os.path.isdir(path):
            dataset = ds.dataset(path, format="parquet")
            df = dataset.to_table().to_pandas()
        elif path.endswith(".parquet"):
            df = pd.read_parquet(path)
        elif path.endswith(".csv"):
            df = pd.read_csv(path)
        else:
            raise ValueError("Unsupported data file format.")
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset: {e}")

    # ---- Normalize Date and Year ----
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Year"] = df["Date"].dt.year.astype("Int64")

    # ---- Normalize Stock Symbol ----
    if "Stock_symbol" in df.columns:
        df["Stock_symbol"] = df["Stock_symbol"].astype(str).str.strip().str.upper()

    return df

# Load at startup (with logging)
print("========================================")
print("📂 Loading Parquet dataset...")
print(f"📁 Path: {DATA_PATH}")
print("========================================")

start = time.time()

try:
    if os.path.isdir(DATA_PATH):
        file_list = [f for f in os.listdir(DATA_PATH) if f.endswith(".parquet")]
        print(f"🔍 Found {len(file_list)} parquet files. Loading...")

    DF = load_data(DATA_PATH)

    print("✅ Load completed!")
    print(f"📊 Total rows loaded: {len(DF):,}")
    print(f"⏱ Load time: {time.time() - start:.2f} s")
    print("========================================")

except Exception as e:
    print("❌ Failed to load dataset:", e)
    raise e

# Precompute dropdown options
YEARS = sorted(DF["Year"].dropna().unique().tolist()[::-1]) if "Year" in DF.columns else []
SYMBOLS = sorted(DF["Stock_symbol"].dropna().unique().tolist()) if "Stock_symbol" in DF.columns else []

if "sentiment_score_signed" in DF.columns:
    MIN_SCORE = float(DF["sentiment_score_signed"].min())
    MAX_SCORE = float(DF["sentiment_score_signed"].max())
else:
    MIN_SCORE = MAX_SCORE = None


# Helper functions
def compute_annual_trend(df: pd.DataFrame):
    if "Year" not in df.columns or "sentiment_score_signed" not in df.columns:
        return [], []
    annual = df.groupby("Year")["sentiment_score_signed"].mean().reset_index()
    return annual["Year"].tolist(), annual["sentiment_score_signed"].round(4).tolist()


def compute_yearly_summary(df: pd.DataFrame, selected_year):
    if selected_year not in ["", "all", None] and "Year" in df.columns:
        try:
            df_year = df[df["Year"] == int(selected_year)]
        except:
            df_year = df.copy()
    else:
        df_year = df.copy()

    if "sentiment_label" not in df_year.columns or df_year.empty:
        return {"total_articles": 0, "dominant_label": None, "dominant_pct": None, "summary_rows": []}

    counts = df_year["sentiment_label"].value_counts()
    total = int(counts.sum())
    if total == 0:
        return {"total_articles": 0, "dominant_label": None, "dominant_pct": None, "summary_rows": []}

    dom_label = counts.index[0]
    dom_pct = round(counts.iloc[0] / total * 100, 1)
    summary_rows = [{"Sentiment": label, "Count": int(cnt), "Percentage": f"{cnt/total*100:.1f}%"}
                    for label, cnt in counts.items()]

    return {"total_articles": total, "dominant_label": dom_label, "dominant_pct": dom_pct, "summary_rows": summary_rows}


# Flask Route
@app.route("/", methods=["GET"])
def index():
    selected_year = request.args.get("year", "all")
    selected_symbol = request.args.get("symbol", "all")
    keyword = request.args.get("keyword", "").strip()
    selected_news_id = request.args.get("news_id", "").strip()

    try:
        sent_min = float(request.args.get("sent_min", MIN_SCORE))
        sent_max = float(request.args.get("sent_max", MAX_SCORE))
    except:
        sent_min, sent_max = MIN_SCORE, MAX_SCORE

    # ----------- Filtering -----------
    filtered = DF.copy()

    # ---- Year ----
    if selected_year not in ["", "all", None]:
        try:
            filtered = filtered[filtered["Year"] == int(selected_year)]
        except:
            pass

    # ---- Symbol (normalize input) ----
    if selected_symbol not in ["", "all", None]:
        selected_symbol = selected_symbol.strip().upper()
        filtered = filtered[filtered["Stock_symbol"] == selected_symbol]

    # ---- Sentiment ----
    if MIN_SCORE is not None:
        filtered = filtered[
            (filtered["sentiment_score_signed"] >= sent_min) &
            (filtered["sentiment_score_signed"] <= sent_max)
        ]

    # ---- Keyword ----
    if keyword:
        filtered = filtered[
            filtered["Article_title"].str.contains(keyword, case=False, na=False)
        ]

    # ---- news_id ----
    if selected_news_id:
        filtered = filtered[filtered["news_id"].astype(str) == str(selected_news_id)]

    # ---- Sort ----
    if "Date" in filtered.columns:
        filtered = filtered.sort_values("Date", ascending=False)

    # Columns to show
    cols_to_show = [
        "Year", "Date", "Stock_symbol", "news_id", "Article_title",
        "sentiment_label", "sentiment_score_signed", "avg_sentiment_score",
        "article_count", "publisher_count", "avg_title_len", "sentiment_category",
    ]
    cols_to_show = [c for c in cols_to_show if c in filtered.columns]

    table_rows = filtered[cols_to_show].head(200).to_dict(orient="records")

    # ---- Detail ----
    selected_row = None
    if selected_news_id:
        match = filtered[filtered["news_id"].astype(str) == str(selected_news_id)]
        if not match.empty:
            selected_row = match.iloc[0].to_dict()

    # ---- Trend & Summary ----
    trend_years, trend_scores = compute_annual_trend(DF)
    yearly_summary = compute_yearly_summary(DF, selected_year)
    summary_title = "Overall Trend for All Years" if selected_year == "all" else f"Overall Trend for {selected_year}"

    return render_template(
        "index.html",
        years=YEARS,
        symbols=SYMBOLS,
        min_score=MIN_SCORE,
        max_score=MAX_SCORE,
        selected_year=selected_year,
        selected_symbol=selected_symbol,
        keyword=keyword,
        sent_min=sent_min,
        sent_max=sent_max,
        row_count=len(filtered),
        table_cols=cols_to_show,
        table_rows=table_rows,
        selected_news_id=selected_news_id,
        selected_row=selected_row,
        trend_years=trend_years,
        trend_scores=trend_scores,
        summary_title=summary_title,
        yearly_summary=yearly_summary,
    )


# Run app
if __name__ == "__main__":
    app.run(debug=True)
