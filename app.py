import streamlit as st
from agent import InventoryRiskAgent

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

        metrics = dashboard["summary_metrics"]

        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Total SKUs", metrics["total_skus"])
        col2.metric("Urgent", metrics["urgent_count"])
        col3.metric("High", metrics["high_count"])
        col4.metric("Medium", metrics["medium_count"])
        col5.metric("Low", metrics["low_count"])
        col6.metric("Est. Shortage Cost", f"${metrics['total_shortage_cost']:,.2f}")

        st.subheader("Full Inventory Risk Table")
        st.dataframe(dashboard["full_data"], use_container_width=True)

        st.subheader("Top Risks")
        st.dataframe(dashboard["top_risks"], use_container_width=True)

        st.subheader("Reorder Recommendations")
        st.dataframe(dashboard["reorder_recommendations"], use_container_width=True)

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