import argparse
import asyncio
import json
import os
from typing import Dict, List

from ...utils.config import config
from .stem_generator import STEMDataGenerator

async def load_templates(category: str) -> List[Dict]:
    """Load templates for a category."""
    template_path = os.path.join(
        config.get("synthetic_data.templates_dir"),
        "stem",
        f"{category}.json"
    )
    
    try:
        with open(template_path, "r") as f:
            data = json.load(f)
            return data.get("templates", [])
    except FileNotFoundError:
        print(f"No templates found for category: {category}")
        return []

async def generate_stem_dataset(
    categories: List[str],
    num_base_examples: int,
    num_variations: int,
    num_connections: int,
    output_dir: str
) -> None:
    """
    Generate a comprehensive STEM dataset.
    
    Args:
        categories: List of STEM categories
        num_base_examples: Number of base examples per category
        num_variations: Number of variations per example
        num_connections: Number of concept connections to generate
        output_dir: Directory to save generated data
    """
    # Initialize generator
    generator = STEMDataGenerator("stem", config.config)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    all_examples = []
    
    # Generate base examples for each category
    for category in categories:
        print(f"\nGenerating examples for {category}...")
        
        # Load templates
        templates = await load_templates(category)
        if not templates:
            continue
        
        # Generate base examples
        examples = await generator.generate_dataset(
            num_examples=num_base_examples,
            templates=templates,
            output_path=os.path.join(output_dir, f"{category}_base.json")
        )
        
        print(f"Generated {len(examples)} base examples")
        all_examples.extend(examples)
        
        # Generate variations with different difficulties
        all_variations = []
        for example in examples:
            variations = await generator.generate_variations_with_difficulty(
                example=example,
                num_variations=num_variations,
                target_difficulties=["easy", "medium", "hard"]
            )
            all_variations.extend(variations)
        
        print(f"Generated {len(all_variations)} variations")
        
        # Save variations
        if all_variations:
            with open(os.path.join(output_dir, f"{category}_variations.json"), "w") as f:
                json.dump(all_variations, f, indent=2)
    
    # Generate concept connections across categories
    if len(all_examples) >= 2:
        print("\nGenerating concept connections...")
        connections = await generator.generate_concept_connections(
            examples=all_examples,
            num_connections=num_connections
        )
        
        print(f"Generated {len(connections)} concept connections")
        
        # Save connections
        if connections:
            with open(os.path.join(output_dir, "concept_connections.json"), "w") as f:
                json.dump(connections, f, indent=2)
    
    print("\nDataset generation complete!")

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic STEM dataset")
    
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["algebra", "geometry", "calculus", "physics"],
        help="STEM categories to generate examples for"
    )
    
    parser.add_argument(
        "--num-base",
        type=int,
        default=10,
        help="Number of base examples per category"
    )
    
    parser.add_argument(
        "--num-variations",
        type=int,
        default=3,
        help="Number of variations per example"
    )
    
    parser.add_argument(
        "--num-connections",
        type=int,
        default=5,
        help="Number of concept connections to generate"
    )
    
    parser.add_argument(
        "--output-dir",
        default="data/synthetic/generated",
        help="Directory to save generated data"
    )
    
    args = parser.parse_args()
    
    # Run generation
    asyncio.run(generate_stem_dataset(
        categories=args.categories,
        num_base_examples=args.num_base,
        num_variations=args.num_variations,
        num_connections=args.num_connections,
        output_dir=args.output_dir
    ))

if __name__ == "__main__":
    main() 