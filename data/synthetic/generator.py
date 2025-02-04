import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from ...api.openai_client import OpenAIClient
from ...utils.config import config

class SyntheticDataGenerator(ABC):
    """Base class for synthetic data generation."""
    
    def __init__(self, task_name: str, config: Dict[str, Any]):
        """
        Initialize the generator.
        
        Args:
            task_name: Name of the task
            config: Configuration dictionary
        """
        self.task_name = task_name
        self.config = config
        self.openai_client = OpenAIClient()
        self.validator_model = config.get("synthetic_data.validation.validator_model", "gpt-4")
        self.quality_threshold = config.get("synthetic_data.validation.quality_threshold", 0.8)
    
    @abstractmethod
    async def generate_example(self, template: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a single example from a template.
        
        Args:
            template: Template dictionary containing structure and constraints
            
        Returns:
            Generated example dictionary
        """
        pass
    
    @abstractmethod
    async def validate_example(self, example: Dict[str, Any]) -> Tuple[bool, float, str]:
        """
        Validate a generated example.
        
        Args:
            example: The generated example to validate
            
        Returns:
            Tuple of (is_valid, quality_score, feedback)
        """
        pass
    
    async def generate_dataset(
        self,
        num_examples: int,
        templates: List[Dict[str, Any]],
        output_path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate a dataset of examples.
        
        Args:
            num_examples: Number of examples to generate
            templates: List of templates to use
            output_path: Optional path to save the dataset
            
        Returns:
            List of generated examples
        """
        examples = []
        failed_attempts = 0
        max_attempts = num_examples * 2  # Allow for some failures
        
        while len(examples) < num_examples and failed_attempts < max_attempts:
            # Select template (can be random or based on distribution)
            template = templates[len(examples) % len(templates)]
            
            try:
                # Generate example
                example = await self.generate_example(template)
                
                # Validate example
                is_valid, quality_score, feedback = await self.validate_example(example)
                
                if is_valid and quality_score >= self.quality_threshold:
                    example["metadata"] = {
                        "template_id": template.get("id"),
                        "quality_score": quality_score,
                        "validation_feedback": feedback
                    }
                    examples.append(example)
                    print(f"Generated valid example {len(examples)}/{num_examples}")
                else:
                    failed_attempts += 1
                    print(f"Example failed validation: {feedback}")
            
            except Exception as e:
                failed_attempts += 1
                print(f"Error generating example: {str(e)}")
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(examples, f, indent=2)
        
        return examples
    
    async def augment_example(
        self,
        example: Dict[str, Any],
        augmentation_type: str
    ) -> Dict[str, Any]:
        """
        Augment an existing example to create variations.
        
        Args:
            example: The example to augment
            augmentation_type: Type of augmentation to apply
            
        Returns:
            Augmented example
        """
        prompt = f"""
        Augment this example using the {augmentation_type} technique:
        
        Original example:
        {json.dumps(example, indent=2)}
        
        Create a new version that:
        1. Maintains the same core concept and difficulty
        2. Changes the surface details
        3. Has a valid solution
        4. Follows the same format
        
        Return the augmented example in the same JSON format.
        """
        
        response = await self.openai_client.generate(
            prompt=prompt,
            model="gpt-4",
            max_tokens=500
        )
        
        try:
            augmented = json.loads(response.text)
            is_valid, quality_score, feedback = await self.validate_example(augmented)
            
            if is_valid and quality_score >= self.quality_threshold:
                augmented["metadata"] = {
                    **example.get("metadata", {}),
                    "augmentation_type": augmentation_type,
                    "original_id": example.get("id"),
                    "quality_score": quality_score,
                    "validation_feedback": feedback
                }
                return augmented
            else:
                raise ValueError(f"Augmented example failed validation: {feedback}")
        
        except Exception as e:
            raise ValueError(f"Error augmenting example: {str(e)}")
    
    async def generate_variations(
        self,
        example: Dict[str, Any],
        num_variations: int,
        augmentation_types: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple variations of an example.
        
        Args:
            example: Base example to create variations from
            num_variations: Number of variations to generate
            augmentation_types: List of augmentation types to use
            
        Returns:
            List of variations
        """
        variations = []
        
        for _ in range(num_variations):
            augmentation_type = augmentation_types[_ % len(augmentation_types)]
            try:
                variation = await self.augment_example(example, augmentation_type)
                variations.append(variation)
            except Exception as e:
                print(f"Error generating variation: {str(e)}")
        
        return variations 