from flask import Flask, render_template, request
import pandas as pd
import os

app = Flask(__name__)

# ------------ Config ------------
DEFAULT_FILENAME = "gold_with_sentiment_sample.csv"


def load_data(filename: str) -> pd.DataFrame:
    """Load CSV and preprocess."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"CSV not found: {filename}")

    df = pd.read_csv(filename)

    # Date → datetime & Year
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Year"] = df["Date"].dt.year.astype("Int64")

    return df


# Load once at startup
DF = load_data(DEFAULT_FILENAME)

# Precompute option lists
YEARS = sorted(DF["Year"].dropna().unique().tolist()[::-1]) if "Year" in DF.columns else []
SYMBOLS = sorted(DF["Stock_symbol"].dropna().unique().tolist()) if "Stock_symbol" in DF.columns else []

if "sentiment_score_signed" in DF.columns:
    MIN_SCORE = float(DF["sentiment_score_signed"].min())
    MAX_SCORE = float(DF["sentiment_score_signed"].max())
else:
    MIN_SCORE = MAX_SCORE = None


def compute_annual_trend(df: pd.DataFrame):
    if "Year" not in df.columns or "sentiment_score_signed" not in df.columns:
        return [], []
    annual = (
        df.groupby("Year")["sentiment_score_signed"]
        .mean()
        .reset_index()
        .dropna(subset=["Year"])
    )
    return annual["Year"].astype(int).tolist(), annual["sentiment_score_signed"].round(4).tolist()


def compute_yearly_summary(df: pd.DataFrame, selected_year):
    """Summary only filtered by year."""
    if "Year" in df.columns and selected_year != "all":
        try:
            df_year = df[df["Year"] == int(selected_year)]
        except:
            df_year = df.copy()
    else:
        df_year = df.copy()

    if "sentiment_label" not in df_year.columns or df_year.empty:
        return {
            "total_articles": 0,
            "dominant_label": None,
            "dominant_pct": None,
            "summary_rows": [],
        }

    counts = df_year["sentiment_label"].value_counts().sort_values(ascending=False)
    total = int(counts.sum())
    if total == 0:
        return {
            "total_articles": 0,
            "dominant_label": None,
            "dominant_pct": None,
            "summary_rows": [],
        }

    dom_label = counts.index[0]
    dom_pct = round(counts.iloc[0] / total * 100, 1)

    summary_rows = [
        {"Sentiment": label, "Count": int(cnt), "Percentage": f"{cnt/total*100:.1f}%"}
        for label, cnt in counts.items()
    ]

    return {
        "total_articles": total,
        "dominant_label": dom_label,
        "dominant_pct": dom_pct,
        "summary_rows": summary_rows,
    }


@app.route("/", methods=["GET"])
def index():
    # ----------- Read query params -----------
    selected_year = request.args.get("year", "all")
    selected_symbol = request.args.get("symbol", "all")
    keyword = request.args.get("keyword", "").strip()
    selected_news_id = request.args.get("news_id", "").strip()

    # Allow typing blank
    if selected_year == "":
        selected_year = "all"
    if selected_symbol == "":
        selected_symbol = "all"

    # sentiment range
    try:
        sent_min = float(request.args.get("sent_min", MIN_SCORE))
        sent_max = float(request.args.get("sent_max", MAX_SCORE))
    except:
        sent_min, sent_max = MIN_SCORE, MAX_SCORE

    # ----------- Filtering -----------
    filtered = DF.copy()

    # Year
    if selected_year != "all" and "Year" in filtered.columns:
        try:
            filtered = filtered[filtered["Year"] == int(selected_year)]
        except:
            pass

    # Symbol
    if selected_symbol != "all" and "Stock_symbol" in filtered.columns:
        filtered = filtered[filtered["Stock_symbol"] == selected_symbol]

    # Sentiment
    if MIN_SCORE is not None:
        filtered = filtered[
            (filtered["sentiment_score_signed"] >= sent_min)
            & (filtered["sentiment_score_signed"] <= sent_max)
        ]

    # Keyword
    if keyword and "Article_title" in filtered.columns:
        filtered = filtered[
            filtered["Article_title"].str.contains(keyword, case=False, na=False)
        ]

    # Sort
    if "Date" in filtered.columns and pd.api.types.is_datetime64_any_dtype(filtered["Date"]):
        filtered = filtered.sort_values("Date", ascending=False)

    # columns to show
    cols_to_show = [
        "Year", "Date", "Stock_symbol", "news_id", "Article_title",
        "sentiment_label", "sentiment_score_signed", "avg_sentiment_score",
        "article_count", "publisher_count",
        "avg_title_len", "sentiment_category",
    ]
    cols_to_show = [c for c in cols_to_show if c in filtered.columns]

    table_df = filtered[cols_to_show].head(200)
    table_rows = table_df.to_dict(orient="records")

    # ----------- Detail Section -----------
    selected_row = None
    if selected_news_id and not filtered.empty:
        try:
            nid_type = type(filtered["news_id"].iloc[0])
            nid = nid_type(selected_news_id)
            row = filtered[filtered["news_id"] == nid]
        except:
            row = filtered[filtered["news_id"].astype(str) == selected_news_id]

        if not row.empty:
            selected_row = row.iloc[0].to_dict()

    # ----------- Trend & Summary -----------
    trend_years, trend_scores = compute_annual_trend(DF)
    yearly_summary = compute_yearly_summary(DF, selected_year)
    summary_title = (
        "Overall Trend for All Years"
        if selected_year == "all"
        else f"Overall Trend for {selected_year}"
    )

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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
