# Reasoning LLM Evaluation Framework

A comprehensive framework for evaluating reasoning capabilities of Large Language Models (LLMs) across multiple tasks and scenarios.

## Features

- Multi-model evaluation support (OpenAI, DeepSeek)
- Synthetic data generation for diverse reasoning tasks
- Automated evaluation pipeline
- Performance metrics and visualization
- Comprehensive logging and error handling

## Project Structure

```
reasoning_evals/
├── api/                 # API integrations for different LLM providers
├── tasks/              # Task definitions and implementations
├── evaluation/         # Evaluation metrics and scoring
├── utils/              # Utility functions and helpers
├── config/             # Configuration files
├── data/               # Data storage
│   ├── raw/           # Raw input data
│   ├── processed/     # Processed data
│   └── synthetic/     # Generated synthetic data
├── results/            # Evaluation results and visualizations
└── tests/              # Test suite
```

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   ```bash
   cp .env.example .env
   # Add your API keys to .env
   ```

## Usage

1. Configure evaluation settings in `config/config.yaml`
2. Run evaluations:
   ```python
   from reasoning_evals.evaluation import run_evaluation
   
   results = run_evaluation(task_name="math_reasoning")
   ```

## Task Types

1. STEM Problem Solving
2. Logical Reasoning & Puzzle Solving
3. Code Generation & Debugging
4. Decision-Making & Planning

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

MIT License
