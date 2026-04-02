import streamlit as st
import matplotlib.pyplot as plt
from agent import InventoryRiskAgent


def convert_df_to_csv(df):
    return df.to_csv(index=False).encode("utf-8")


def get_item_column(df):
    possible_cols = ["sku", "SKU", "item", "item_id", "sku_id", "product_id"]
    for col in possible_cols:
        if col in df.columns:
            return col
    raise ValueError(f"No SKU-like column found. Available columns: {df.columns.tolist()}")


st.set_page_config(page_title="Inventory Risk Agent", layout="wide")

st.title("AI Inventory Risk Agent")
st.write("Upload inventory data, compute shortage risk, and query the agent in natural language.")

uploaded_file = st.file_uploader("Upload inventory CSV", type=["csv"])
use_sample = st.checkbox("Use sample inventory data")

file_to_use = None

if uploaded_file is not None:
    file_to_use = uploaded_file
elif use_sample:
    file_to_use = "sample_inventory.csv"

if file_to_use is not None:
    try:
        agent = InventoryRiskAgent(file_to_use)
        dashboard = agent.get_dashboard_data()

        full_data_csv = convert_df_to_csv(dashboard["full_data"])
        top_risks_csv = convert_df_to_csv(dashboard["top_risks"])
        reorder_csv = convert_df_to_csv(dashboard["reorder_recommendations"])

        st.subheader("Inventory Risk Distribution")

        risk_df = dashboard["full_data"]
        risk_order = ["Urgent", "High", "Medium", "Low", "Unknown"]
        risk_counts = (
            risk_df["risk_level"]
            .fillna("Unknown")
            .value_counts()
            .reindex(risk_order, fill_value=0)
        )

        fig, ax = plt.subplots()
        risk_counts.plot(kind="bar", ax=ax)
        ax.set_title("Inventory Risk Distribution by SKU")
        ax.set_xlabel("Risk Level")
        ax.set_ylabel("Number of SKUs")
        st.pyplot(fig)

        metrics = dashboard["summary_metrics"]

        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Total SKUs", metrics["total_skus"])
        col2.metric("Urgent", metrics["urgent_count"])
        col3.metric("High", metrics["high_count"])
        col4.metric("Medium", metrics["medium_count"])
        col5.metric("Low", metrics["low_count"])
        col6.metric("Est. Shortage Cost", f"${metrics['total_shortage_cost']:,.2f}")

        st.subheader("Executive Summary")

        top_risks_df = dashboard["top_risks"]
        reorder_df = dashboard["reorder_recommendations"]
        forecast_df = dashboard["forecast_view"]

        top_risk_col = get_item_column(top_risks_df)
        top_reorder_col = get_item_column(reorder_df)

        top_risk_skus = ", ".join(top_risks_df[top_risk_col].astype(str).head(3).tolist()) or "None"
        top_reorder_skus = ", ".join(reorder_df[top_reorder_col].astype(str).head(3).tolist()) or "None"

        projected_stockout_count = (
            forecast_df["projected_stockout_risk"] == "Projected Stockout"
        ).sum()

        summary_lines = [
            f"{metrics['urgent_count']} SKUs are currently in the urgent risk category.",
            f"Estimated total shortage cost exposure is ${metrics['total_shortage_cost']:,.2f}.",
            f"Highest-risk SKUs include: {top_risk_skus}.",
            f"Top reorder priorities include: {top_reorder_skus}.",
            f"{projected_stockout_count} SKUs are projected to stock out within the next 4 weeks.",
        ]

        for line in summary_lines:
            st.write(f"- {line}")

        st.subheader("Download Reports")

        dcol1, dcol2, dcol3 = st.columns(3)

        with dcol1:
            st.download_button(
                label="Download Full Risk Table",
                data=full_data_csv,
                file_name="inventory_risk_table.csv",
                mime="text/csv",
            )

        with dcol2:
            st.download_button(
                label="Download Top Risks",
                data=top_risks_csv,
                file_name="top_inventory_risks.csv",
                mime="text/csv",
            )

        with dcol3:
            st.download_button(
                label="Download Reorder Recommendations",
                data=reorder_csv,
                file_name="reorder_recommendations.csv",
                mime="text/csv",
            )

        st.subheader("Full Inventory Risk Table")
        st.dataframe(dashboard["full_data"], use_container_width=True)

        st.subheader("Top Risks")
        st.dataframe(dashboard["top_risks"], use_container_width=True)

        st.subheader("Reorder Recommendations")
        st.dataframe(dashboard["reorder_recommendations"], use_container_width=True)

        st.subheader("Projected Inventory Outlook")
        st.caption("4-week demand projection based on current weekly demand")

        forecast_cols = [
            "SKU",
            "Product",
            "Current_Stock",
            "Weekly_Demand",
            "projected_weekly_demand",
            "projected_demand_next_4_weeks",
            "projected_inventory_after_4_weeks",
            "weeks_of_cover",
            "projected_stockout_risk",
        ]

        st.dataframe(
            dashboard["forecast_view"][forecast_cols],
            use_container_width=True
        )

        st.subheader("Ask the Agent")
        user_question = st.text_input(
            "Try: Which SKUs are urgent? / What should I reorder first? / Give me a planner summary."
        )

        if st.button("Run Agent") and user_question:
            result = agent.route(user_question)

            st.write(f"**Tool used:** {result['tool_used']}")
            st.write("### Agent Response")
            st.write(result["text"])

            if result["data"] is not None:
                st.write("### Supporting Data")
                st.dataframe(result["data"], use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")

else:
    st.info("Upload a CSV or check 'Use sample inventory data' to begin.")