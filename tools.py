import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def load_inventory_data(file) -> pd.DataFrame:
    df = pd.read_csv(file)

    numeric_cols = [
        "Current_Stock",
        "Weekly_Demand",
        "Lead_Time_Days",
        "Reorder_Point",
        "Unit_Cost",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=numeric_cols).copy()
    return df


def calculate_inventory_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["daily_demand"] = df["Weekly_Demand"] / 7
    df["days_of_inventory"] = df["Current_Stock"] / df["daily_demand"]
    df["lead_time_demand"] = df["daily_demand"] * df["Lead_Time_Days"]
    df["stock_gap"] = df["lead_time_demand"] - df["Current_Stock"]
    df["reorder_needed"] = df["Current_Stock"] < df["Reorder_Point"]
    df["estimated_shortage_units"] = df["stock_gap"].clip(lower=0)
    df["estimated_shortage_cost"] = df["estimated_shortage_units"] * df["Unit_Cost"]

    return df


def classify_inventory_risk(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def get_risk(row):
        if row["Current_Stock"] <= 0:
            return "Urgent"
        if row["days_of_inventory"] < row["Lead_Time_Days"]:
            return "High"
        if row["Current_Stock"] < row["Reorder_Point"]:
            return "Medium"
        return "Low"

    def get_priority_score(row):
        score = 0

        if row["Current_Stock"] <= 0:
            score += 100
        if row["days_of_inventory"] < row["Lead_Time_Days"]:
            score += 50
        if row["Current_Stock"] < row["Reorder_Point"]:
            score += 25

        score += max(row["estimated_shortage_cost"], 0)
        return score

    df["risk_level"] = df.apply(get_risk, axis=1)
    df["priority_score"] = df.apply(get_priority_score, axis=1)

    return df.sort_values(by="priority_score", ascending=False).reset_index(drop=True)


def get_top_risks(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    return df[df["risk_level"].isin(["Urgent", "High", "Medium"])].head(n).copy()


def get_urgent_items(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["risk_level"].isin(["Urgent", "High"])].copy()


def get_reorder_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    result["recommended_reorder_qty"] = (
        (result["lead_time_demand"] + result["Reorder_Point"] - result["Current_Stock"])
        .clip(lower=0)
        .round(0)
    )

    result = result[result["recommended_reorder_qty"] > 0].copy()

    return result[
        [
            "SKU",
            "Product",
            "Current_Stock",
            "Weekly_Demand",
            "Lead_Time_Days",
            "Reorder_Point",
            "risk_level",
            "recommended_reorder_qty",
            "estimated_shortage_cost",
        ]
    ].sort_values(by="estimated_shortage_cost", ascending=False)


def get_summary_metrics(df: pd.DataFrame) -> dict:
    return {
        "total_skus": int(len(df)),
        "urgent_count": int((df["risk_level"] == "Urgent").sum()),
        "high_count": int((df["risk_level"] == "High").sum()),
        "medium_count": int((df["risk_level"] == "Medium").sum()),
        "low_count": int((df["risk_level"] == "Low").sum()),
        "total_shortage_cost": round(float(df["estimated_shortage_cost"].sum()), 2),
    }


def prepare_context_table(df: pd.DataFrame, limit: int = 10) -> str:
    cols = [
        "SKU",
        "Product",
        "Current_Stock",
        "Weekly_Demand",
        "Lead_Time_Days",
        "Reorder_Point",
        "days_of_inventory",
        "lead_time_demand",
        "stock_gap",
        "risk_level",
        "estimated_shortage_cost",
    ]

    small_df = df[cols].head(limit).copy()
    small_df = small_df.round(2)

    return small_df.to_csv(index=False)

def project_future_inventory(df, weeks_ahead=4, growth_rate=0.0):
    """
    Project future demand and inventory using current weekly demand.
    
    growth_rate:
        0.0  = baseline forecast
        0.10 = 10% demand growth scenario
       -0.10 = 10% demand decline scenario
    """

    df = df.copy()

    df["projected_weekly_demand"] = df["Weekly_Demand"] * (1 + growth_rate)
    df["projected_demand_next_4_weeks"] = df["projected_weekly_demand"] * weeks_ahead
    df["projected_inventory_after_4_weeks"] = (
        df["Current_Stock"] - df["projected_demand_next_4_weeks"]
    )

    df["weeks_of_cover"] = np.where(
        df["Weekly_Demand"] > 0,
        df["Current_Stock"] / df["Weekly_Demand"],
        np.inf
    )

    df["projected_stockout_risk"] = np.select(
        [
            df["projected_inventory_after_4_weeks"] < 0,
            df["weeks_of_cover"] < 1,
            df["weeks_of_cover"] < 2
        ],
        [
            "Projected Stockout",
            "High Risk",
            "Medium Risk"
        ],
        default="Low Risk"
    )

    return df


def forecast_demand(df, periods=7):
    """
    Forecast demand using simple linear regression trend.
    Requires columns: sku, date, demand
    """

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    forecasts = []

    for sku, group in df.groupby("sku"):
        group = group.sort_values("date")

        if len(group) < 3:
            continue

        X = np.arange(len(group)).reshape(-1, 1)
        y = group["demand"].values

        model = LinearRegression()
        model.fit(X, y)

        future_X = np.arange(len(group), len(group) + periods).reshape(-1, 1)
        preds = model.predict(future_X)

        preds = np.clip(preds, 0, None)

        forecasts.append({
            "sku": sku,
            "forecast_demand_next_7_days": round(float(preds.sum()), 2),
            "forecast_avg_daily_demand": round(float(preds.mean()), 2)
        })

    return pd.DataFrame(forecasts)

def merge_forecast_with_inventory(inventory_df, forecast_df):
    
    merged_df = inventory_df.merge(forecast_df, on="sku", how="left")

    merged_df["forecast_demand_next_7_days"] = merged_df["forecast_demand_next_7_days"].fillna(0)
    merged_df["forecast_avg_daily_demand"] = merged_df["forecast_avg_daily_demand"].fillna(0)

    merged_df["projected_inventory_after_7_days"] = (
        merged_df["inventory"] - merged_df["forecast_demand_next_7_days"]
    )

    return merged_df

def classify_forecast_risk(df):
    
    df = df.copy()

    df["forecast_risk"] = np.select(
        [
            df["projected_inventory_after_7_days"] < 0,
            df["projected_inventory_after_7_days"] < df["inventory"] * 0.2
        ],
        [
            "Projected Stockout",
            "Low Inventory Risk"
        ],
        default="Safe"
    )

    return df

