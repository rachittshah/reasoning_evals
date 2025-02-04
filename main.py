import argparse
import asyncio
import os
from typing import List, Optional

from tasks.stem import STEMTask
from evaluation.pipeline import EvaluationPipeline
from evaluation.visualization import EvaluationVisualizer
from utils.config import config

async def run_evaluation(
    task_names: List[str],
    output_dir: str,
    num_examples: Optional[int] = None
) -> None:
    """
    Run evaluations for specified tasks.
    
    Args:
        task_names: List of task names to evaluate
        output_dir: Directory to save results
        num_examples: Optional number of examples to generate per task
    """
    pipeline = EvaluationPipeline()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Map task names to task classes
    task_map = {
        "stem": STEMTask
    }
    
    for task_name in task_names:
        if task_name not in task_map:
            print(f"Warning: Task {task_name} not implemented, skipping")
            continue
        
        print(f"\nEvaluating {task_name} task...")
        
        # Initialize task
        task_config = config.get_task_config(task_name)
        task = task_map[task_name](task_config)
        
        # Run evaluation
        results = await pipeline.evaluate_task(
            task=task,
            num_examples=num_examples
        )
        
        # Generate visualizations
        task_output_dir = os.path.join(output_dir, task_name)
        os.makedirs(task_output_dir, exist_ok=True)
        
        visualizer = EvaluationVisualizer(results)
        visualizer.save_visualizations(task_output_dir)
        
        print(f"Results saved to {task_output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Run LLM reasoning evaluations")
    
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["stem"],
        help="Tasks to evaluate (default: stem)"
    )
    
    parser.add_argument(
        "--output-dir",
        default="evaluation_results",
        help="Directory to save results (default: evaluation_results)"
    )
    
    parser.add_argument(
        "--num-examples",
        type=int,
        help="Number of examples to generate per task (overrides config)"
    )
    
    args = parser.parse_args()
    
    # Run evaluations
    asyncio.run(run_evaluation(
        task_names=args.tasks,
        output_dir=args.output_dir,
        num_examples=args.num_examples
    ))

if __name__ == "__main__":
    main() 