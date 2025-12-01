# Quick Start Guide

## 🚀 Get Started in 3 Minutes

### Step 1: Install Dependencies (30 seconds)

```bash
cd business_agent_system
pip install -r requirements.txt
```

### Step 2: Add Your API Key (1 minute)

**Get your free API key:**
1. Go to https://ai.google.dev/
2. Click "Get API Key"
3. Sign in with your Google account
4. Create a new API key
5. Copy the key

**Add the key using ONE of these methods:**

**Method A - Edit the file (Recommended for beginners)**
```python
# Open main.py and find line ~83
class Config:
    GEMINI_API_KEY = "paste-your-key-here"  # Replace this
```

**Method B - Environment variable (Recommended for production)**
```bash
export GEMINI_API_KEY="your-api-key-here"
```

**Method C - .env file**
```bash
cp .env.example .env
# Edit .env and add your key
nano .env
```

### Step 3: Run It! (1 minute)

```bash
python main.py
```

That's it! You'll see:
- ✅ Deal analysis workflow
- ✅ Performance report generation
- ✅ Iterative negotiation loop
- ✅ Pause/resume session demo

## 📖 Next Steps

### Try the Examples

**1. Analyze a Deal**
```python
import asyncio
from main import BusinessAgentOrchestrator

async def quick_test():
    # Initialize with your API key
    agent = BusinessAgentOrchestrator(api_key="YOUR_KEY_HERE")
    
    # Analyze a deal
    result = await agent.analyze_deal_workflow({
        "deal_id": "TEST-001",
        "price": 100000,
        "estimated_cost": 75000,
        "industry": "technology"
    })
    
    print(result)

asyncio.run(quick_test())
```

**2. Generate Report**
```python
result = await agent.generate_performance_report_workflow({
    "revenue": {"current": 1200000, "previous": 1000000},
    "deals": {"won": 20, "total": 100}
})
print(result)
```

**3. Run Negotiation Loop**
```python
result = await agent.iterative_negotiation_loop({
    "deal_id": "DEAL-002",
    "price": 150000,
    "estimated_cost": 120000
}, max_rounds=5)
print(result)
```

### Explore Advanced Features

```bash
python advanced_features.py
```

This demonstrates:
- MCP (Model Context Protocol) tools
- OpenAPI integration
- Advanced observability
- Context engineering
- Agent evaluation
- A2A (Agent-to-Agent) protocol

### Run Tests

```bash
pip install pytest pytest-asyncio
pytest tests/test_agents.py -v
```

## 🎯 Common Use Cases

### Use Case 1: Evaluate a Real Deal

```python
from main import BusinessAgentOrchestrator
import asyncio

async def evaluate_deal():
    agent = BusinessAgentOrchestrator()
    
    my_deal = {
        "deal_id": "ACME-2024-Q4",
        "client_id": "ACME-CORP",
        "industry": "software",
        "price": 500000,
        "estimated_cost": 350000,
        "payment_terms": "Net 60",
        "duration_months": 36,
        "deal_importance": "high"
    }
    
    result = await agent.analyze_deal_workflow(my_deal)
    
    # Extract key insights
    analysis = result.get("deal_analysis", {})
    print(f"Risk Score: {analysis.get('risk_score')}")
    print(f"Status: {analysis.get('status')}")
    print(f"Recommendations: {analysis.get('recommendations')}")

asyncio.run(evaluate_deal())
```

### Use Case 2: Monthly Performance Report

```python
async def monthly_report():
    agent = BusinessAgentOrchestrator()
    
    this_month_metrics = {
        "revenue": {
            "current": 2500000,
            "previous": 2200000
        },
        "deals": {
            "won": 45,
            "total": 180
        },
        "client_satisfaction": 4.3
    }
    
    report = await agent.generate_performance_report_workflow(this_month_metrics)
    
    print("\n📊 EXECUTIVE SUMMARY")
    print("="*50)
    print(report.get("report", {}).get("executive_summary"))
    print("\n📈 KEY METRICS")
    print("="*50)
    for key, value in report.get("report", {}).get("summary", {}).items():
        print(f"{key}: {value}")

asyncio.run(monthly_report())
```

### Use Case 3: Interactive Negotiation

```python
async def negotiate_deal():
    agent = BusinessAgentOrchestrator()
    
    # Start with initial offer
    initial = {
        "deal_id": "NEGOTIATION-001",
        "price": 200000,
        "estimated_cost": 150000,
        "payment_terms": "Net 90",
        "duration_months": 24
    }
    
    # Run negotiation rounds
    result = await agent.iterative_negotiation_loop(initial, max_rounds=3)
    
    print(f"\n🤝 NEGOTIATION RESULTS")
    print("="*50)
    print(f"Total Rounds: {result['total_rounds']}")
    print(f"Success: {result['negotiation_successful']}")
    print(f"Final Price: ${result['final_offer']['price']:,.2f}")
    
    # Review each round
    for i, round_data in enumerate(result['rounds'], 1):
        print(f"\nRound {i}:")
        print(f"  Offer: ${round_data['offer']['price']:,.2f}")
        print(f"  Status: {round_data['analysis']['status']}")

asyncio.run(negotiate_deal())
```

## 🔧 Customization

### Add Your Own Agent

```python
from main import BusinessAgentOrchestrator
from google import genai

class MyCustomAgent:
    def __init__(self, client: genai.Client):
        self.client = client
        self.name = "CustomAgent"
    
    async def my_task(self, data: dict) -> dict:
        prompt = f"Analyze this: {data}"
        response = self.client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=prompt
        )
        return {"result": response.text}

# Add to orchestrator
agent = BusinessAgentOrchestrator()
agent.my_agent = MyCustomAgent(agent.client)
```

### Add Your Own Tool

```python
from main import BusinessTools

@staticmethod
def my_custom_tool(param1: str, param2: int) -> dict:
    """My custom business tool"""
    return {
        "processed": param1,
        "count": param2,
        "status": "success"
    }

# Add to BusinessTools class
BusinessTools.my_custom_tool = my_custom_tool
```

## 📚 File Structure

```
business_agent_system/
├── main.py                    # Main agent system
├── advanced_features.py       # Advanced concepts demo
├── requirements.txt           # Dependencies
├── config.yaml               # Configuration
├── .env.example              # Environment template
├── README.md                 # Full documentation
├── QUICKSTART.md            # This file
├── agent_system.log         # Runtime logs
└── tests/
    └── test_agents.py       # Unit tests
```

## ❓ Troubleshooting

### "Module not found" error
```bash
pip install google-genai asyncio python-dotenv
```

### "API Key not set" error
Make sure you've set the key using one of the three methods above.

### "Rate limit exceeded" error
You're making too many requests. Add delays between calls:
```python
import time
time.sleep(2)  # Wait 2 seconds between requests
```

### "Connection timeout" error
Check your internet connection. The API requires internet access.

## 💡 Tips

1. **Start Small**: Test with one agent workflow first
2. **Check Logs**: Review `agent_system.log` for details
3. **Monitor Metrics**: Use `agent.get_metrics()` to track performance
4. **Save Sessions**: Use pause/resume for long-running tasks
5. **Read Examples**: All examples in `main.py` are fully functional

## 🆘 Need Help?

1. Read the full [README.md](README.md)
2. Check [Google ADK Documentation](https://ai.google.dev/)
3. Review the example code in `main.py`
4. Run the tests: `pytest tests/`

## 🎉 You're Ready!

You now have a production-ready multi-agent system for business negotiations and reporting. Customize it for your specific needs!

Happy building! 🚀
