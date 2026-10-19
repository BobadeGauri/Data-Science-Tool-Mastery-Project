"""End-to-end retail sales analysis and simple regression baseline."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "retail_sales_raw.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

def main():
    df = pd.read_csv(DATA, parse_dates=["date"])
    print("Raw shape:", df.shape)
    print("Missing values before cleaning:\n", df.isna().sum())

    # Data cleaning: fill categorical gaps with mode, then recompute measures.
    for col in ["region", "channel"]:
        df[col] = df[col].fillna(df[col].mode().iloc[0])
    numeric = ["units_sold", "unit_cost", "unit_price"]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())
    df["revenue"] = df["units_sold"] * df["unit_price"]
    df["cost"] = df["units_sold"] * df["unit_cost"]
    df["profit"] = df["revenue"] - df["cost"]
    df["profit_margin"] = np.where(df["revenue"] > 0, df["profit"] / df["revenue"], 0)
    df["month"] = df["date"].dt.to_period("M").astype(str)
    df.to_csv(OUT / "retail_sales_cleaned.csv", index=False, date_format="%Y-%m-%d")

    # Descriptive statistics and grouped analysis.
    summary = df[["units_sold", "revenue", "cost", "profit", "profit_margin"]].describe().round(2)
    summary.to_csv(OUT / "descriptive_statistics.csv")
    category_summary = df.groupby("category", as_index=False).agg(
        units_sold=("units_sold", "sum"), revenue=("revenue", "sum"), profit=("profit", "sum")
    ).sort_values("revenue", ascending=False)
    category_summary.to_csv(OUT / "category_summary.csv", index=False)
    monthly = df.groupby("month", as_index=False).agg(revenue=("revenue", "sum"), profit=("profit", "sum"))
    monthly.to_csv(OUT / "monthly_summary.csv", index=False)

    sns.set_theme()
    plt.figure(figsize=(8, 5))
    sns.barplot(data=category_summary, x="category", y="revenue")
    plt.title("Revenue by Product Category")
    plt.xlabel("Category"); plt.ylabel("Revenue")
    plt.tight_layout(); plt.savefig(OUT / "revenue_by_category.png", dpi=150); plt.close()

    plt.figure(figsize=(8, 5))
    sns.lineplot(data=monthly, x="month", y="revenue", marker="o")
    plt.title("Monthly Revenue Trend")
    plt.xlabel("Month"); plt.ylabel("Revenue")
    plt.tight_layout(); plt.savefig(OUT / "monthly_revenue.png", dpi=150); plt.close()

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x="units_sold", y="profit", hue="category")
    plt.title("Units Sold vs Profit")
    plt.tight_layout(); plt.savefig(OUT / "units_vs_profit.png", dpi=150); plt.close()

    # Baseline ML model: predict profit from available transaction fields.
    features = ["region", "category", "product", "units_sold", "unit_cost", "unit_price", "channel"]
    X, y = df[features], df["profit"]
    categorical = ["region", "category", "product", "channel"]
    numeric_features = ["units_sold", "unit_cost", "unit_price"]
    preprocessor = ColumnTransformer([
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features)
    ])
    model = Pipeline([("preprocessor", preprocessor),
                     ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": mean_squared_error(y_test, pred) ** 0.5,
        "R2": r2_score(y_test, pred),
        "note": "Illustrative only: dataset is small and synthetic."
    }
    pd.DataFrame([metrics]).to_csv(OUT / "model_metrics.csv", index=False)
    print("Total revenue:", round(df["revenue"].sum(), 2))
    print("Total profit:", round(df["profit"].sum(), 2))
    print("Category summary:\n", category_summary.to_string(index=False))
    print("Model metrics:", metrics)
    print("Outputs saved to", OUT)

if __name__ == "__main__":
    main()
