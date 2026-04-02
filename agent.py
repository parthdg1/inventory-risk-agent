from tools import (
    load_inventory_data,
    calculate_inventory_metrics,
    classify_inventory_risk,
    get_top_risks,
    get_urgent_items,
    get_reorder_recommendations,
    get_summary_metrics,
    prepare_context_table,
    project_future_inventory
)
from llm_helper import generate_planner_summary, answer_inventory_question


class InventoryRiskAgent:
    REQUIRED_COLUMNS = ["sku", "inventory", "demand", "lead_time", "unit_cost"]
    
    def __init__(self, file):
        self.raw_df = load_inventory_data(file)
        self.metrics_df = calculate_inventory_metrics(self.raw_df)
        self.risk_df = classify_inventory_risk(self.metrics_df)
        self.forecast_df = project_future_inventory(self.raw_df)

    def route(self, user_question: str) -> dict:
        q = user_question.lower()

        if any(word in q for word in ["urgent", "immediate", "critical"]):
            urgent_df = get_urgent_items(self.risk_df)
            return {
                "tool_used": "get_urgent_items",
                "data": urgent_df,
                "text": f"Found {len(urgent_df)} urgent/high-risk SKUs.",
            }

        if any(word in q for word in ["reorder", "order", "buy", "purchase"]):
            reorder_df = get_reorder_recommendations(self.risk_df)
            return {
                "tool_used": "get_reorder_recommendations",
                "data": reorder_df,
                "text": f"Generated reorder recommendations for {len(reorder_df)} SKUs.",
            }

        if any(word in q for word in ["summary", "summarize", "overview", "planner"]):
            summary_metrics = get_summary_metrics(self.risk_df)
            context_table = prepare_context_table(self.risk_df)
            summary = generate_planner_summary(summary_metrics, context_table)
            return {
                "tool_used": "generate_planner_summary",
                "data": None,
                "text": summary,
            }

        if any(word in q for word in ["risk", "stockout", "top", "shortage"]):
            top_df = get_top_risks(self.risk_df)
            summary_metrics = get_summary_metrics(self.risk_df)
            context_table = prepare_context_table(top_df, limit=10)
            answer = answer_inventory_question(user_question, summary_metrics, context_table)
            return {
                "tool_used": "get_top_risks + llm_answer",
                "data": top_df,
                "text": answer,
            }

        summary_metrics = get_summary_metrics(self.risk_df)
        context_table = prepare_context_table(self.risk_df)
        answer = answer_inventory_question(user_question, summary_metrics, context_table)

        return {
            "tool_used": "general_llm_answer",
            "data": None,
            "text": answer,
        }

    def get_dashboard_data(self):
        return {
            "full_data": self.risk_df,
            "summary_metrics": get_summary_metrics(self.risk_df),
            "top_risks": get_top_risks(self.risk_df),
            "urgent_items": get_urgent_items(self.risk_df),
            "reorder_recommendations": get_reorder_recommendations(self.risk_df),
            "forecast_view": self.forecast_df
        }