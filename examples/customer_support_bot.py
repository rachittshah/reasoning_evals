import asyncio
import json
from typing import Dict, List, Optional
from dataclasses import dataclass
import openai
from pathlib import Path

@dataclass
class CustomerContext:
    user_id: str
    issue_type: str
    previous_interactions: List[Dict]
    account_info: Optional[Dict] = None
    system_info: Optional[Dict] = None

class CustomerSupportBot:
    def __init__(self, api_key: str, templates_path: str):
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.templates = self._load_templates(templates_path)
        self.conversation_history = {}
        
    def _load_templates(self, path: str) -> Dict:
        with open(path, 'r') as f:
            return json.load(f)
    
    def _get_template(self, issue_type: str) -> Dict:
        for template in self.templates['templates']:
            if template['id'] == issue_type:
                return template
        raise ValueError(f"No template found for issue type: {issue_type}")
    
    def _build_system_prompt(self, template: Dict) -> str:
        return f"""You are an expert customer support agent. Follow these guidelines:
        1. Use the template pattern: {template['structure']['pattern']}
        2. Constraints: {', '.join(template['constraints'])}
        3. Maintain a professional, empathetic tone
        4. Ask clarifying questions when needed
        5. Provide step-by-step solutions
        6. Verify understanding with the customer"""
    
    def _build_user_prompt(self, context: CustomerContext, user_message: str) -> str:
        template = self._get_template(context.issue_type)
        
        # Format context based on template structure
        formatted_context = {
            "previous_interactions": "\n".join([
                f"User: {msg['user']}\nAgent: {msg['agent']}"
                for msg in context.previous_interactions[-2:]  # Last 2 interactions
            ]),
            "account_info": json.dumps(context.account_info) if context.account_info else "Not available",
            "system_info": json.dumps(context.system_info) if context.system_info else "Not available"
        }
        
        return f"""Context:
        Previous interactions: {formatted_context['previous_interactions']}
        Account info: {formatted_context['account_info']}
        System info: {formatted_context['system_info']}
        
        Current message: {user_message}"""
    
    async def get_response(self, context: CustomerContext, user_message: str) -> str:
        template = self._get_template(context.issue_type)
        system_prompt = self._build_system_prompt(template)
        user_prompt = self._build_user_prompt(context, user_message)
        
        # Get response from GPT-4
        response = await self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content

async def main():
    # Example usage
    api_key = "your-api-key"
    bot = CustomerSupportBot(
        api_key=api_key,
        templates_path="reasoning_evals/data/templates/customer_support/scenarios.json"
    )
    
    # Example context for a technical issue
    context = CustomerContext(
        user_id="user123",
        issue_type="technical_troubleshooting",
        previous_interactions=[],
        system_info={
            "device_type": "smartphone",
            "os": "iOS 16.2",
            "app_version": "2.1.0"
        }
    )
    
    # Example conversation
    user_messages = [
        "Hi, my app keeps crashing when I try to upload photos. I've already tried restarting the app.",
        "Yes, I checked and the app has permission to access photos.",
        "The storage space is fine, but the app is still crashing."
    ]
    
    for message in user_messages:
        response = await bot.get_response(context, message)
        print(f"\nUser: {message}")
        print(f"Bot: {response}")
        context.previous_interactions.append({
            "user": message,
            "agent": response
        })

if __name__ == "__main__":
    asyncio.run(main()) 