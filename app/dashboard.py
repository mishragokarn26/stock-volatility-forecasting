# app/dashboard.py (excerpt with updated Upload Tab)
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from arch import arch_model

st.set_page_config(page_title="Stock Volatility Forecasting Dashboard", layout="wide")
st.title("📈 Stock Volatility Forecasting: GARCH Family Comparison")

# Sidebar controls
st.sidebar.header("Controls")
ticker_label = st.sidebar.selectbox("Select Asset", ["AAPL", "S&P500 (GSPC)"], index=0)
n_days = st.sidebar.slider("Show last N days", min_value=50, max_value=2000, value=250, step=50)

# map label to processed filename prefix
ticker_map = {"AAPL": "AAPL", "S&P500 (GSPC)": "GSPC"}
ticker_key = ticker_map.get(ticker_label, "AAPL")

DATA_DIR = Path("data/processed")

@st.cache_data
def read_csv_flex(path: Path):
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.set_index("Date")
    elif df.columns[0].lower().startswith("unnamed") or df.columns[0].lower() == "date":
        df = df.rename(columns={df.columns[0]: "Date"})
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.set_index("Date")
    else:
        try:
            df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], errors="coerce")
            df = df.set_index(df.columns[0])
            df.index.name = "Date"
        except Exception:
            try:
                df.index = pd.to_datetime(df.index, errors="coerce")
            except Exception:
                pass
    return df

@st.cache_data
def load_all(ticker_key: str):
    roll_df = read_csv_flex(DATA_DIR / "rolling_garch_forecasts.csv")
    eval_df = pd.read_csv(DATA_DIR / "garch_model_eval_summary.csv", index_col=0) if (DATA_DIR / "garch_model_eval_summary.csv").exists() else pd.DataFrame()
    compare_df = pd.read_csv(DATA_DIR / "garch_model_compare.csv", index_col=0) if (DATA_DIR / "garch_model_compare.csv").exists() else pd.DataFrame()
    dm_df = pd.read_csv(DATA_DIR / "garch_diebold_mariano.csv", index_col=0) if (DATA_DIR / "garch_diebold_mariano.csv").exists() else pd.DataFrame()

    realized = pd.DataFrame()
    proc_name = f"{ticker_key}_processed.csv"
    realized_path = DATA_DIR / proc_name
    if realized_path.exists():
        realized = pd.read_csv(realized_path)
        if "Date" in realized.columns:
            realized["Date"] = pd.to_datetime(realized["Date"], errors="coerce")
            realized = realized.set_index("Date")
        if "squared_return" in realized.columns:
            realized["realized_vol"] = (realized["squared_return"] ** 0.5) * 100
        elif "log_return" in realized.columns:
            realized["realized_vol"] = realized["log_return"].abs() * 100

    return roll_df, eval_df, compare_df, dm_df, realized

# Load precomputed data
roll_df, eval_df, compare_df, dm_df, realized = load_all(ticker_key)

# Tabs layout
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Rolling Forecasts", "ML Models", "Evaluation", "Upload New Data", "About"])

# --- Rolling Forecasts Tab ---
with tab1:
    st.subheader("Rolling 1-Day Ahead Volatility Forecasts")
    if not roll_df.empty:
        plot_df = roll_df.tail(n_days)
        fig, ax = plt.subplots(figsize=(12, 6))
        for col in plot_df.columns:
            try:
                ax.plot(plot_df.index, pd.to_numeric(plot_df[col], errors="coerce"), label=col)
            except Exception:
                continue
        if not realized.empty:
            common_idx = plot_df.index.intersection(realized.index)
            if len(common_idx) > 0:
                ax.plot(common_idx, realized.loc[common_idx, "realized_vol"], color="k", linewidth=2, label="Realized Volatility")
        ax.set_title(f"{ticker_label}: Rolling Forecasts (last {n_days} days)")
        ax.set_ylabel("Volatility (%)")
        ax.legend()
        st.pyplot(fig)

        # Download button
        st.download_button(
            "📥 Download Forecasts CSV",
            plot_df.to_csv().encode("utf-8"),
            file_name="rolling_forecasts.csv",
            mime="text/csv"
        )
    else:
        st.info("No rolling forecasts available. Please run the forecast notebook/script.")

# --- ML Models Tab ---
with tab2:
    ml_path = DATA_DIR / "ml_forecasts.csv"
    if ml_path.exists():
        st.subheader("Machine Learning Volatility Forecasts")
        ml_df = pd.read_csv(ml_path, parse_dates=["Date"], index_col="Date")
        ml_df = ml_df.tail(n_days)
        common_idx = ml_df.index.intersection(realized.index)

        fig, ax = plt.subplots(figsize=(12,6))
        for col in ml_df.columns:
            ax.plot(ml_df.index, ml_df[col], label=col)
        if not realized.empty and len(common_idx) > 0:
            ax.plot(common_idx, realized.loc[common_idx, "realized_vol"], color="k", linewidth=2, label="Realized Volatility")
        ax.set_title(f"{ticker_label}: ML Forecasts (last {n_days} days)")
        ax.set_ylabel("Volatility (%)")
        ax.legend()
        st.pyplot(fig)

        st.download_button(
            "📥 Download ML Forecasts CSV",
            ml_df.to_csv().encode("utf-8"),
            file_name="ml_forecasts.csv",
            mime="text/csv"
        )
    else:
        st.info("No ML forecasts available. Please run 05_ml_models.ipynb.")

# --- Evaluation Tab ---
with tab3:
    st.subheader("Model Evaluation Metrics")
    if not eval_df.empty:
        metric_choice = st.selectbox("Sort models by:", ["rmse", "mae"])
        eval_sorted = eval_df.sort_values(metric_choice)
        st.dataframe(eval_sorted.style.format({"rmse": "{:.4f}", "mae": "{:.4f}"}))
    else:
        st.info("Evaluation summary not found (garch_model_eval_summary.csv).")

    st.subheader("Model Fit Comparison (AIC / BIC)")
    if not compare_df.empty:
        metric_choice2 = st.selectbox("Sort fit table by:", ["aic", "bic", "loglik"])
        compare_sorted = compare_df.sort_values(metric_choice2)
        st.dataframe(compare_sorted.style.format({"loglik": "{:.2f}", "aic": "{:.2f}", "bic": "{:.2f}"}))
    else:
        st.info("Model fit comparison file not found (garch_model_compare.csv).")

    st.subheader("Diebold–Mariano Test Results")
    if not dm_df.empty:
        st.dataframe(dm_df.style.format({"DM_stat": "{:.3f}", "pvalue": "{:.4f}"}))
    else:
        st.info("Diebold–Mariano results not found (garch_diebold_mariano.csv).")
# --- Upload New Data Tab ---
with tab4:
    st.subheader("📤 Upload New Stock Data (CSV)")
    uploaded_file = st.file_uploader("Upload a CSV with Date & Adj Close columns", type=["csv"])

    demo_file = os.environ.get("VOLATILITY_DEMO_CSV")
    data_file = uploaded_file if uploaded_file is not None else demo_file
    if data_file is not None:
        if uploaded_file is None:
            st.caption(f"Loaded dataset: {Path(demo_file).name}")
        new_df = pd.read_csv(data_file)
        if "Date" in new_df.columns and "Adj Close" in new_df.columns:
            new_df["Date"] = pd.to_datetime(new_df["Date"], errors="coerce")
            new_df = new_df.set_index("Date").sort_index()

            # Compute returns & realized volatility
            new_df["log_return"] = np.log(new_df["Adj Close"] / new_df["Adj Close"].shift(1)) * 100
            new_df["squared_return"] = new_df["log_return"] ** 2
            new_df["realized_vol"] = np.sqrt(new_df["squared_return"]).rolling(21).mean()

            # --- Charts ---
            st.line_chart(new_df[["Adj Close"]], height=300)
            st.line_chart(new_df[["log_return"]].dropna(), height=300)
            st.line_chart(new_df[["realized_vol"]].dropna(), height=300)

            # --- Quick GARCH(1,1) fit ---
            st.subheader("⚡ Auto-Fit GARCH(1,1) Baseline")
            returns = new_df["log_return"].dropna()
            try:
                model = arch_model(returns, vol="Garch", p=1, q=1, mean="constant")
                res = model.fit(disp="off")
                new_df["garch_vol"] = res.conditional_volatility

                fig, ax = plt.subplots(figsize=(12,6))
                ax.plot(new_df.index, new_df["realized_vol"], label="Realized Volatility", color="k", alpha=0.7)
                ax.plot(new_df.index, new_df["garch_vol"], label="GARCH(1,1) Volatility", color="red")
                ax.set_title("Uploaded Data: GARCH(1,1) vs Realized Volatility")
                ax.set_ylabel("Volatility (%)")
                ax.legend()
                st.pyplot(fig)

                st.text(res.summary())
            except Exception as e:
                st.error(f"GARCH fitting failed: {e}")

            # --- Download processed file ---
            st.download_button(
                "📥 Download Processed File (with GARCH fit)",
                new_df.to_csv().encode("utf-8"),
                file_name="processed_uploaded_data.csv",
                mime="text/csv"
            )
        else:
            st.error("Uploaded file must contain 'Date' and 'Adj Close' columns.")
# --- About Tab ---
with tab5:
    st.markdown("""
    ### About this Project  
    This dashboard compares **statistical models (GARCH, GJR, EGARCH)** with  
    **machine learning models (Random Forest, XGBoost)** for forecasting daily  
    stock volatility.  

    The new **Upload** tab lets you upload custom stock data (CSV with `Date` and `Adj Close`)  
    and instantly view price, returns, and realized volatility.
    """)

# Sidebar footer
st.sidebar.markdown("---")
st.sidebar.write("**Project:** Stock Volatility Forecasting")
st.sidebar.write("**Students:** Gokarn And Pradhyumn")
st.sidebar.write("**Course:** Time Series Analysis")
