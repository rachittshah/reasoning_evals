from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class EvaluationVisualizer:
    """Visualization tools for evaluation results."""
    
    def __init__(self, results: pd.DataFrame):
        """
        Initialize visualizer with results DataFrame.
        
        Args:
            results: DataFrame containing evaluation results
        """
        self.results = results
    
    def plot_accuracy_comparison(
        self,
        by_category: bool = False,
        by_difficulty: bool = False
    ) -> go.Figure:
        """
        Plot accuracy comparison between models.
        
        Args:
            by_category: Whether to break down by category
            by_difficulty: Whether to break down by difficulty
            
        Returns:
            Plotly figure
        """
        if by_category and "category" in self.results.columns:
            fig = px.bar(
                self.results.groupby(["model_name", "category"])["is_correct"].mean().reset_index(),
                x="model_name",
                y="is_correct",
                color="category",
                barmode="group",
                title="Model Accuracy by Category",
                labels={"is_correct": "Accuracy", "model_name": "Model"}
            )
        elif by_difficulty and "difficulty" in self.results.columns:
            fig = px.bar(
                self.results.groupby(["model_name", "difficulty"])["is_correct"].mean().reset_index(),
                x="model_name",
                y="is_correct",
                color="difficulty",
                barmode="group",
                title="Model Accuracy by Difficulty",
                labels={"is_correct": "Accuracy", "model_name": "Model"}
            )
        else:
            fig = px.bar(
                self.results.groupby("model_name")["is_correct"].mean().reset_index(),
                x="model_name",
                y="is_correct",
                title="Model Accuracy Comparison",
                labels={"is_correct": "Accuracy", "model_name": "Model"}
            )
        
        return fig
    
    def plot_reasoning_quality_radar(self) -> go.Figure:
        """
        Create a radar plot of reasoning quality metrics.
        
        Returns:
            Plotly figure
        """
        metrics = ["reasoning_quality", "step_clarity"]
        available_metrics = [m for m in metrics if m in self.results.columns]
        
        fig = go.Figure()
        
        for model in self.results["model_name"].unique():
            model_data = self.results[self.results["model_name"] == model]
            values = [model_data[metric].mean() for metric in available_metrics]
            values.append(values[0])  # Close the polygon
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=available_metrics + [available_metrics[0]],
                name=model,
                fill="toself"
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            title="Reasoning Quality Metrics"
        )
        
        return fig
    
    def plot_latency_boxplot(self) -> go.Figure:
        """
        Create boxplot of model latencies.
        
        Returns:
            Plotly figure
        """
        if "latency" not in self.results.columns:
            raise ValueError("Latency data not available in results")
        
        fig = px.box(
            self.results,
            x="model_name",
            y="latency",
            title="Model Latency Distribution",
            labels={"latency": "Latency (seconds)", "model_name": "Model"}
        )
        
        return fig
    
    def plot_token_usage_bar(self) -> go.Figure:
        """
        Plot token usage comparison.
        
        Returns:
            Plotly figure
        """
        if "tokens_used" not in self.results.columns:
            raise ValueError("Token usage data not available in results")
        
        token_stats = self.results.groupby("model_name").agg({
            "tokens_used": ["mean", "std"]
        }).reset_index()
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name="Mean Token Usage",
            x=token_stats["model_name"],
            y=token_stats["tokens_used"]["mean"],
            error_y=dict(
                type="data",
                array=token_stats["tokens_used"]["std"],
                visible=True
            )
        ))
        
        fig.update_layout(
            title="Token Usage by Model",
            xaxis_title="Model",
            yaxis_title="Tokens Used"
        )
        
        return fig
    
    def generate_summary_report(self) -> str:
        """
        Generate a text summary of the evaluation results.
        
        Returns:
            Markdown formatted summary string
        """
        summary = []
        summary.append("# Evaluation Summary Report\n")
        
        # Overall statistics
        summary.append("## Overall Statistics\n")
        overall_stats = self.results.groupby("model_name").agg({
            "is_correct": ["mean", "count"],
            "reasoning_quality": "mean"
        }).round(3)
        
        summary.append("| Model | Accuracy | Sample Size | Reasoning Quality |")
        summary.append("|-------|----------|-------------|------------------|")
        
        for model in overall_stats.index:
            stats = overall_stats.loc[model]
            summary.append(
                f"| {model} | {stats['is_correct']['mean']:.3f} | "
                f"{stats['is_correct']['count']} | {stats['reasoning_quality']['mean']:.3f} |"
            )
        
        # Category breakdown if available
        if "category" in self.results.columns:
            summary.append("\n## Performance by Category\n")
            category_stats = self.results.groupby(["model_name", "category"])["is_correct"].mean().round(3)
            
            summary.append("| Model | Category | Accuracy |")
            summary.append("|-------|----------|----------|")
            
            for (model, category), accuracy in category_stats.items():
                summary.append(f"| {model} | {category} | {accuracy:.3f} |")
        
        # Error analysis
        summary.append("\n## Error Analysis\n")
        error_cases = self.results[~self.results["is_correct"]].groupby("model_name").size()
        
        summary.append("| Model | Number of Errors |")
        summary.append("|-------|-----------------|")
        
        for model, errors in error_cases.items():
            summary.append(f"| {model} | {errors} |")
        
        return "\n".join(summary)
    
    def save_visualizations(self, output_dir: str) -> None:
        """
        Save all visualizations to files.
        
        Args:
            output_dir: Directory to save visualization files
        """
        # Create plots
        plots = {
            "accuracy": self.plot_accuracy_comparison(),
            "accuracy_by_category": self.plot_accuracy_comparison(by_category=True),
            "accuracy_by_difficulty": self.plot_accuracy_comparison(by_difficulty=True),
            "reasoning_quality": self.plot_reasoning_quality_radar(),
            "latency": self.plot_latency_boxplot(),
            "token_usage": self.plot_token_usage_bar()
        }
        
        # Save each plot
        for name, fig in plots.items():
            try:
                fig.write_html(f"{output_dir}/{name}.html")
                fig.write_image(f"{output_dir}/{name}.png")
            except Exception as e:
                print(f"Error saving {name} plot: {str(e)}")
        
        # Save summary report
        with open(f"{output_dir}/summary_report.md", "w") as f:
            f.write(self.generate_summary_report()) 