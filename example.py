import asyncio
import os

from tasks.stem import STEMTask
from evaluation.pipeline import EvaluationPipeline
from evaluation.visualization import EvaluationVisualizer
from utils.config import config

async def main():
    """Example usage of the reasoning evaluation framework."""
    
    # Initialize task
    task_config = config.get_task_config("stem")
    stem_task = STEMTask(task_config)
    
    # Create evaluation pipeline
    pipeline = EvaluationPipeline()
    
    # Run evaluation with specific models
    models = [
        {
            "provider": "openai",
            "name": "gpt-4",
            "alias": "o1"
        },
        {
            "provider": "openai",
            "name": "gpt-3.5-turbo",
            "alias": "o3-mini-high"
        },
        {
            "provider": "deepseek",
            "name": "deepseek-coder",
            "alias": "r1"
        }
    ]
    
    # Create output directory
    output_dir = "example_results"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Running STEM task evaluation...")
    results = await pipeline.evaluate_task(
        task=stem_task,
        models=models,
        num_examples=5  # Small number for demonstration
    )
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    visualizer = EvaluationVisualizer(results)
    visualizer.save_visualizations(output_dir)
    
    # Print summary report
    print("\nEvaluation Summary:")
    print(visualizer.generate_summary_report())

if __name__ == "__main__":
    asyncio.run(main()) 