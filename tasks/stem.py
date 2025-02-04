import json
import random
import uuid
from typing import Any, Dict, List, Optional, Tuple

from ..api.openai_client import OpenAIClient
from .base import BaseTask, TaskExample, TaskResult

class STEMTask(BaseTask):
    """Implementation of STEM problem-solving task."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("stem", config)
        self.categories = config.get("categories", [])
        self.difficulty_levels = config.get("difficulty_levels", [])
        self.judge_model = config.get("evaluation", {}).get("judge_model", "gpt-4")
        self.openai_client = OpenAIClient()
        
        # Load templates
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load problem templates from JSON files."""
        templates = {}
        for category in self.categories:
            try:
                with open(f"data/templates/stem/{category}.json", "r") as f:
                    templates[category] = json.load(f)
            except FileNotFoundError:
                print(f"Warning: No templates found for category {category}")
                templates[category] = []
        return templates
    
    async def generate_examples(self, num_examples: int) -> List[TaskExample]:
        """Generate synthetic STEM problems."""
        examples = []
        for _ in range(num_examples):
            category = random.choice(self.categories)
            difficulty = random.choice(self.difficulty_levels)
            
            # Get template
            template = random.choice(self.templates[category])
            
            # Generate problem using GPT-4
            prompt = f"""
            Generate a {difficulty} {category} problem based on this template:
            {template['structure']}
            
            The problem should:
            1. Be clearly stated
            2. Have a unique correct answer
            3. Require multi-step reasoning
            4. Include all necessary information
            
            Format:
            Problem: [problem text]
            Solution: [detailed step-by-step solution]
            Answer: [final numerical or symbolic answer]
            """
            
            response = await self.openai_client.generate(
                prompt=prompt,
                model="gpt-4",
                max_tokens=500
            )
            
            # Parse response
            lines = response.text.strip().split("\n")
            problem = ""
            solution = ""
            answer = ""
            
            current_section = None
            for line in lines:
                if line.startswith("Problem:"):
                    current_section = "problem"
                elif line.startswith("Solution:"):
                    current_section = "solution"
                elif line.startswith("Answer:"):
                    current_section = "answer"
                elif current_section == "problem":
                    problem += line + "\n"
                elif current_section == "solution":
                    solution += line + "\n"
                elif current_section == "answer":
                    answer += line + "\n"
            
            example = TaskExample(
                id=str(uuid.uuid4()),
                input=problem.strip(),
                expected_output=answer.strip(),
                metadata={
                    "category": category,
                    "difficulty": difficulty,
                    "solution": solution.strip(),
                    "template_id": template.get("id")
                }
            )
            
            # Validate example
            is_valid, error = self.validate_example(example)
            if is_valid:
                examples.append(example)
            else:
                print(f"Skipping invalid example: {error}")
        
        return examples
    
    async def evaluate_response(
        self,
        example: TaskExample,
        model_response: str,
        model_name: str
    ) -> TaskResult:
        """Evaluate a model's response to a STEM problem."""
        # Use GPT-4 as a judge
        judge_prompt = f"""
        Evaluate this response to a STEM problem.
        
        Problem:
        {example.input}
        
        Expected Answer:
        {example.expected_output}
        
        Model's Response:
        {model_response}
        
        Evaluate the following aspects:
        1. Correctness (Is the final answer correct?)
        2. Reasoning Quality (Scale 0-1, how clear and logical is the reasoning?)
        3. Step-by-Step Clarity (Scale 0-1, how well are the steps explained?)
        
        Format your response as JSON:
        {{
            "is_correct": true/false,
            "reasoning_quality": float,
            "step_clarity": float,
            "explanation": "Brief explanation of the evaluation"
        }}
        """
        
        judge_response = await self.openai_client.generate(
            prompt=judge_prompt,
            model=self.judge_model,
            max_tokens=300
        )
        
        try:
            evaluation = json.loads(judge_response.text)
            
            return TaskResult(
                example_id=example.id,
                model_name=model_name,
                model_output=model_response,
                is_correct=evaluation["is_correct"],
                reasoning_quality=evaluation["reasoning_quality"],
                metrics={
                    "step_clarity": evaluation["step_clarity"]
                },
                metadata={
                    "judge_explanation": evaluation["explanation"]
                }
            )
        except Exception as e:
            print(f"Error parsing judge response: {str(e)}")
            return TaskResult(
                example_id=example.id,
                model_name=model_name,
                model_output=model_response,
                is_correct=False,
                reasoning_quality=0.0,
                metrics={},
                metadata={"error": str(e)}
            )
    
    def get_prompt(self, example: TaskExample) -> str:
        """Generate the prompt for a STEM problem."""
        return f"""
        Solve this {example.metadata['category']} problem. Show your work step by step.
        
        Problem:
        {example.input}
        
        Format your response:
        1. First, explain your approach
        2. Show each step of your solution
        3. Clearly state your final answer
        """
    
    def validate_example(self, example: TaskExample) -> Tuple[bool, Optional[str]]:
        """Validate a STEM problem example."""
        if not example.input.strip():
            return False, "Empty problem statement"
        
        if not example.expected_output.strip():
            return False, "Empty expected output"
        
        if not example.metadata.get("solution"):
            return False, "Missing solution in metadata"
        
        if not example.metadata.get("category") in self.categories:
            return False, f"Invalid category: {example.metadata.get('category')}"
        
        if not example.metadata.get("difficulty") in self.difficulty_levels:
            return False, f"Invalid difficulty: {example.metadata.get('difficulty')}"
        
        return True, None 