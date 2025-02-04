import asyncio
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Type

import pandas as pd
from tqdm import tqdm

from ..api.base import BaseModelClient
from ..api.openai_client import OpenAIClient
from ..api.deepseek_client import DeepSeekClient
from ..tasks.base import BaseTask, TaskExample, TaskResult
from ..utils.config import config

class EvaluationPipeline:
    """Pipeline for running model evaluations on tasks."""
    
    def __init__(self):
        self.config = config
        self.model_clients: Dict[str, BaseModelClient] = {
            "openai": OpenAIClient(),
            "deepseek": DeepSeekClient()
        }
        
        # Create results directory if it doesn't exist
        os.makedirs(self.config.get("evaluation.results_dir", "results"), exist_ok=True)
    
    async def evaluate_task(
        self,
        task: BaseTask,
        models: Optional[List[Dict[str, str]]] = None,
        num_examples: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Evaluate models on a task.
        
        Args:
            task: The task to evaluate
            models: List of model configurations to evaluate
            num_examples: Number of examples to generate (overrides config)
            
        Returns:
            DataFrame containing evaluation results
        """
        # Get models to evaluate
        if models is None:
            models = []
            for provider, provider_config in self.config.get("models", {}).items():
                for model in provider_config.get("models", []):
                    models.append({
                        "provider": provider,
                        "name": model["name"],
                        "alias": model["alias"]
                    })
        
        # Generate examples
        n_examples = num_examples or task.config.get("num_examples", 10)
        examples = await task.generate_examples(n_examples)
        
        # Run evaluations
        results: List[TaskResult] = []
        
        for model in tqdm(models, desc="Evaluating models"):
            provider = model["provider"]
            model_name = model["name"]
            client = self.model_clients[provider]
            
            for example in tqdm(examples, desc=f"Testing {model['alias']}", leave=False):
                try:
                    # Get model response
                    prompt = task.get_prompt(example)
                    response = await client.generate(
                        prompt=prompt,
                        model=model_name,
                        max_tokens=self.config.get("evaluation.max_tokens", 1000),
                        temperature=self.config.get("evaluation.temperature", 0.7)
                    )
                    
                    # Evaluate response
                    result = await task.evaluate_response(
                        example=example,
                        model_response=response.text,
                        model_name=model["alias"]
                    )
                    
                    # Add metadata
                    result.metadata = result.metadata or {}
                    result.metadata.update({
                        "tokens_used": response.tokens_used,
                        "latency": response.latency,
                        "provider": provider,
                        "full_model_name": model_name,
                        "task_name": task.task_name,
                        **example.metadata
                    })
                    
                    results.append(result)
                    
                except Exception as e:
                    print(f"Error evaluating {model['alias']} on example {example.id}: {str(e)}")
                    # Add error result
                    results.append(TaskResult(
                        example_id=example.id,
                        model_name=model["alias"],
                        model_output="",
                        is_correct=False,
                        reasoning_quality=0.0,
                        metrics={},
                        metadata={
                            "error": str(e),
                            "provider": provider,
                            "full_model_name": model_name,
                            "task_name": task.task_name,
                            **example.metadata
                        }
                    ))
        
        # Convert results to DataFrame
        df = self._results_to_dataframe(results)
        
        # Save results
        self._save_results(df, task.task_name)
        
        return df
    
    def _results_to_dataframe(self, results: List[TaskResult]) -> pd.DataFrame:
        """Convert task results to a DataFrame."""
        records = []
        for result in results:
            record = {
                "example_id": result.example_id,
                "model_name": result.model_name,
                "is_correct": result.is_correct,
                "reasoning_quality": result.reasoning_quality,
                **result.metrics
            }
            
            if result.metadata:
                for key, value in result.metadata.items():
                    if key not in record:
                        record[key] = value
            
            records.append(record)
        
        return pd.DataFrame.from_records(records)
    
    def _save_results(self, df: pd.DataFrame, task_name: str) -> None:
        """Save evaluation results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = self.config.get("evaluation.results_dir", "results")
        
        # Save CSV
        csv_path = os.path.join(results_dir, f"{task_name}_{timestamp}.csv")
        df.to_csv(csv_path, index=False)
        
        # Save JSON
        json_path = os.path.join(results_dir, f"{task_name}_{timestamp}.json")
        df.to_json(json_path, orient="records", indent=2)
        
        print(f"Results saved to {csv_path} and {json_path}")
    
    @staticmethod
    def load_results(path: str) -> pd.DataFrame:
        """Load results from a CSV or JSON file."""
        if path.endswith(".csv"):
            return pd.read_csv(path)
        elif path.endswith(".json"):
            return pd.read_json(path)
        else:
            raise ValueError("Unsupported file format. Use CSV or JSON.") 