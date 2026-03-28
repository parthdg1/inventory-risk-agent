import pandas as pd


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