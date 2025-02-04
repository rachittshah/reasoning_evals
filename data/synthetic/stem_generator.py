import json
import uuid
from typing import Any, Dict, List, Tuple

from .generator import SyntheticDataGenerator

class STEMDataGenerator(SyntheticDataGenerator):
    """Synthetic data generator for STEM problems."""
    
    async def generate_example(self, template: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a STEM problem from a template."""
        # Create a detailed prompt for GPT-4 to generate a problem
        prompt = f"""
        Generate a STEM problem based on this template:
        {json.dumps(template['structure'], indent=2)}
        
        Requirements:
        1. Follow the pattern exactly
        2. Use variables within the specified ranges
        3. Ensure all constraints are met
        4. Include a detailed step-by-step solution
        5. Provide a clear, unambiguous answer
        
        The response should be a JSON object with these fields:
        - problem: The problem statement
        - solution: Detailed step-by-step solution
        - answer: The final answer
        - variables: Dictionary of variables used and their values
        - difficulty: One of [easy, medium, hard]
        """
        
        response = await self.openai_client.generate(
            prompt=prompt,
            model="gpt-4",
            max_tokens=1000
        )
        
        try:
            generated = json.loads(response.text)
            return {
                "id": str(uuid.uuid4()),
                **generated,
                "template_id": template.get("id")
            }
        except Exception as e:
            raise ValueError(f"Error parsing generated example: {str(e)}")
    
    async def validate_example(self, example: Dict[str, Any]) -> Tuple[bool, float, str]:
        """Validate a generated STEM problem."""
        validation_prompt = f"""
        Validate this STEM problem:
        
        Problem: {example['problem']}
        Solution: {example['solution']}
        Answer: {example['answer']}
        
        Evaluate the following aspects and provide a score (0-1) for each:
        1. Clarity: Is the problem clearly stated?
        2. Completeness: Are all necessary details provided?
        3. Correctness: Is the solution mathematically correct?
        4. Step-by-Step: Is the solution well explained?
        5. Answer Format: Is the answer properly formatted?
        
        Return your evaluation as a JSON object with:
        - scores: Dictionary of aspect scores
        - overall_score: Weighted average of scores
        - is_valid: Boolean indicating if the example meets minimum quality
        - feedback: Detailed feedback explaining the scores
        """
        
        response = await self.openai_client.generate(
            prompt=validation_prompt,
            model=self.validator_model,
            max_tokens=500
        )
        
        try:
            validation = json.loads(response.text)
            return (
                validation["is_valid"],
                validation["overall_score"],
                validation["feedback"]
            )
        except Exception as e:
            return False, 0.0, f"Error validating example: {str(e)}"
    
    async def generate_variations_with_difficulty(
        self,
        example: Dict[str, Any],
        num_variations: int,
        target_difficulties: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Generate variations of a STEM problem with specific difficulties.
        
        Args:
            example: Base example to create variations from
            num_variations: Number of variations to generate
            target_difficulties: List of desired difficulty levels
            
        Returns:
            List of variations
        """
        variations = []
        
        for i in range(num_variations):
            target_difficulty = target_difficulties[i % len(target_difficulties)]
            
            prompt = f"""
            Create a {target_difficulty} variation of this STEM problem:
            
            Original problem:
            {example['problem']}
            
            Requirements:
            1. Maintain the same concept but adjust difficulty to {target_difficulty}
            2. Change numbers, variables, or context
            3. Provide complete solution and answer
            4. Follow the same format
            
            Return the variation as a JSON object with the same structure.
            """
            
            try:
                response = await self.openai_client.generate(
                    prompt=prompt,
                    model="gpt-4",
                    max_tokens=1000
                )
                
                variation = json.loads(response.text)
                variation["id"] = str(uuid.uuid4())
                variation["metadata"] = {
                    **example.get("metadata", {}),
                    "variation_type": "difficulty_adjustment",
                    "original_id": example.get("id"),
                    "target_difficulty": target_difficulty
                }
                
                # Validate the variation
                is_valid, quality_score, feedback = await self.validate_example(variation)
                if is_valid and quality_score >= self.quality_threshold:
                    variation["metadata"]["quality_score"] = quality_score
                    variation["metadata"]["validation_feedback"] = feedback
                    variations.append(variation)
                
            except Exception as e:
                print(f"Error generating variation: {str(e)}")
        
        return variations
    
    async def generate_concept_connections(
        self,
        examples: List[Dict[str, Any]],
        num_connections: int
    ) -> List[Dict[str, Any]]:
        """
        Generate problems that connect concepts from multiple examples.
        
        Args:
            examples: List of examples to combine concepts from
            num_connections: Number of connected problems to generate
            
        Returns:
            List of new problems combining concepts
        """
        connections = []
        
        for _ in range(num_connections):
            if len(examples) < 2:
                break
                
            prompt = f"""
            Create a new STEM problem that combines concepts from these examples:
            
            Example 1:
            {examples[0]['problem']}
            
            Example 2:
            {examples[1]['problem']}
            
            Requirements:
            1. Create a problem that requires understanding both concepts
            2. Ensure the combination is natural and meaningful
            3. Provide detailed solution steps
            4. Make the connection explicit in the solution
            
            Return the new problem as a JSON object with the same structure.
            """
            
            try:
                response = await self.openai_client.generate(
                    prompt=prompt,
                    model="gpt-4",
                    max_tokens=1000
                )
                
                connected = json.loads(response.text)
                connected["id"] = str(uuid.uuid4())
                connected["metadata"] = {
                    "connection_type": "concept_integration",
                    "source_examples": [ex.get("id") for ex in examples[:2]]
                }
                
                # Validate the connected problem
                is_valid, quality_score, feedback = await self.validate_example(connected)
                if is_valid and quality_score >= self.quality_threshold:
                    connected["metadata"]["quality_score"] = quality_score
                    connected["metadata"]["validation_feedback"] = feedback
                    connections.append(connected)
                
            except Exception as e:
                print(f"Error generating connection: {str(e)}")
            
            # Rotate examples for next iteration
            examples = examples[1:] + [examples[0]]
        
        return connections 