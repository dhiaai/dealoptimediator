"""
Business Deal Negotiation & Performance Report Multi-Agent System
Simplified version that works with Google Generative AI SDK

This system includes:
- Multi-agent orchestration (sequential, parallel)
- Custom tools for business operations
- Session management and long-term memory
- Observability (logging, metrics)
"""

import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import asyncio
import time

# Install and import Google Generative AI
try:
    import google.generativeai as genai
except ImportError:
    print("Installing google-generativeai...")
    import subprocess
    subprocess.check_call(["pip", "install", "-q", "google-generativeai"])
    import google.generativeai as genai

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agent_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# =====================================================================
# CONFIGURATION
# =====================================================================

class Config:
    """Configuration for the agent system"""
    GEMINI_API_KEY = ""  # <<< INSERT YOUR GEMINI API KEY HERE >>>
    RATE_LIMIT_DELAY = 4  # Seconds between API calls to avoid rate limits
    
    @classmethod
    def initialize(cls, api_key: Optional[str] = None):
        """Initialize the system with API key"""
        if api_key:
            cls.GEMINI_API_KEY = api_key
        elif not cls.GEMINI_API_KEY:
            cls.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
        
        if not cls.GEMINI_API_KEY:
            raise ValueError("Please provide GEMINI_API_KEY")
        
        genai.configure(api_key=cls.GEMINI_API_KEY)
        logger.info("System initialized successfully")


# =====================================================================
# CUSTOM TOOLS FOR BUSINESS OPERATIONS
# =====================================================================

class BusinessTools:
    """Custom tools for business deal negotiations and reporting"""
    
    @staticmethod
    def analyze_deal_terms(deal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze deal terms and provide recommendations"""
        logger.info(f"Analyzing deal terms: {deal_data.get('deal_id', 'unknown')}")
        
        analysis = {
            "deal_id": deal_data.get("deal_id"),
            "risk_score": 0.0,
            "profitability_score": 0.0,
            "recommendations": [],
            "red_flags": []
        }
        
        # Check pricing
        if "price" in deal_data:
            price = deal_data["price"]
            estimated_cost = deal_data.get("estimated_cost", 0)
            margin = (price - estimated_cost) / price if price > 0 else 0
            
            analysis["profitability_score"] = min(margin * 100, 100)
            
            if margin < 0.15:
                analysis["red_flags"].append("Low profit margin (<15%)")
                analysis["risk_score"] += 30
            
            if margin > 0.40:
                analysis["recommendations"].append("Excellent profit margin - prioritize")
        
        # Check payment terms
        if "payment_terms" in deal_data:
            terms = deal_data["payment_terms"]
            if "net_90" in str(terms).lower() or "90 days" in str(terms).lower():
                analysis["red_flags"].append("Extended payment terms may impact cash flow")
                analysis["risk_score"] += 20
        
        # Check contract duration
        if "duration_months" in deal_data:
            duration = deal_data["duration_months"]
            if duration >= 24:
                analysis["recommendations"].append("Long-term contract - ensure exit clauses")
            if duration >= 36:
                analysis["risk_score"] += 15
        
        analysis["status"] = "approved" if analysis["risk_score"] < 50 else "needs_review"
        
        logger.info(f"Deal analysis complete: {analysis['status']}")
        return analysis
    
    @staticmethod
    def generate_performance_report(metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate performance report from metrics"""
        logger.info("Generating performance report")
        
        report = {
            "report_id": f"RPT-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "generated_at": datetime.now().isoformat(),
            "summary": {},
            "trends": [],
            "action_items": []
        }
        
        # Revenue analysis
        if "revenue" in metrics:
            revenue_data = metrics["revenue"]
            current = revenue_data.get("current", 0)
            previous = revenue_data.get("previous", 0)
            
            growth = ((current - previous) / previous * 100) if previous > 0 else 0
            report["summary"]["revenue_growth"] = round(growth, 2)
            
            if growth > 10:
                report["trends"].append(f"Strong revenue growth of {growth:.1f}%")
            elif growth < 0:
                report["trends"].append(f"Revenue decline of {abs(growth):.1f}%")
                report["action_items"].append("Investigate revenue decline causes")
        
        # Deal conversion rate
        if "deals" in metrics:
            deals_data = metrics["deals"]
            won = deals_data.get("won", 0)
            total = deals_data.get("total", 0)
            
            conversion_rate = (won / total * 100) if total > 0 else 0
            report["summary"]["conversion_rate"] = round(conversion_rate, 2)
            
            if conversion_rate < 20:
                report["action_items"].append("Low conversion rate - review sales process")
        
        # Client satisfaction
        if "client_satisfaction" in metrics:
            satisfaction = metrics["client_satisfaction"]
            report["summary"]["avg_satisfaction"] = satisfaction
            
            if satisfaction < 3.5:
                report["action_items"].append("Client satisfaction below target - conduct surveys")
        
        logger.info(f"Performance report generated: {report['report_id']}")
        return report
    
    @staticmethod
    def calculate_negotiation_strategy(context: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate optimal negotiation strategy"""
        logger.info("Calculating negotiation strategy")
        
        strategy = {
            "approach": "balanced",
            "priorities": [],
            "concession_limits": {},
            "target_outcomes": {}
        }
        
        # Determine approach based on deal importance
        importance = context.get("deal_importance", "medium")
        if importance == "high":
            strategy["approach"] = "collaborative"
            strategy["priorities"].append("Maintain long-term relationship")
        elif importance == "low":
            strategy["approach"] = "competitive"
            strategy["priorities"].append("Maximize immediate gains")
        
        # Set concession limits
        base_price = context.get("base_price", 0)
        if base_price > 0:
            strategy["concession_limits"]["max_discount"] = base_price * 0.15
            strategy["concession_limits"]["target_price"] = base_price * 0.95
        
        # Define target outcomes
        strategy["target_outcomes"]["price"] = context.get("target_price", base_price)
        strategy["target_outcomes"]["payment_terms"] = "Net 30"
        strategy["target_outcomes"]["delivery_time"] = context.get("ideal_delivery", "30 days")
        
        logger.info(f"Strategy calculated: {strategy['approach']}")
        return strategy
    
    @staticmethod
    def fetch_market_data(industry: str, region: str = "global") -> Dict[str, Any]:
        """Fetch market data for context (simulated)"""
        logger.info(f"Fetching market data for {industry} in {region}")
        
        market_data = {
            "industry": industry,
            "region": region,
            "avg_deal_size": 150000,
            "avg_profit_margin": 0.25,
            "competitive_intensity": "high",
            "market_trends": [
                "Digital transformation accelerating",
                "Price sensitivity increasing",
                "Long-term contracts preferred"
            ],
            "benchmark_metrics": {
                "conversion_rate": 0.23,
                "avg_sales_cycle_days": 45,
                "client_retention": 0.85
            }
        }
        
        logger.info("Market data retrieved")
        return market_data


# =====================================================================
# MEMORY MANAGEMENT
# =====================================================================

class BusinessMemoryBank:
    """Long-term memory for business context"""
    
    def __init__(self):
        self.deal_history: Dict[str, List[Dict]] = {}
        self.client_profiles: Dict[str, Dict] = {}
        logger.info("Memory bank initialized")
    
    def store_deal(self, deal_id: str, deal_data: Dict[str, Any]):
        """Store deal information"""
        client = deal_data.get("client_id", "unknown")
        
        if client not in self.deal_history:
            self.deal_history[client] = []
        
        self.deal_history[client].append({
            "deal_id": deal_id,
            "timestamp": datetime.now().isoformat(),
            "data": deal_data
        })
        
        logger.info(f"Deal {deal_id} stored in memory")
    
    def get_client_history(self, client_id: str) -> List[Dict]:
        """Retrieve client deal history"""
        return self.deal_history.get(client_id, [])


# =====================================================================
# SESSION MANAGEMENT
# =====================================================================

class BusinessSessionManager:
    """Manage agent sessions with pause/resume capability"""
    
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        logger.info("Session manager initialized")
    
    def create_session(self, session_id: str, context: Dict[str, Any]) -> str:
        """Create new session"""
        self.sessions[session_id] = {
            "id": session_id,
            "created_at": datetime.now().isoformat(),
            "context": context,
            "state": "active",
            "checkpoints": []
        }
        logger.info(f"Session created: {session_id}")
        return session_id
    
    def save_checkpoint(self, session_id: str, state: Dict[str, Any]):
        """Save session checkpoint for pause/resume"""
        if session_id in self.sessions:
            self.sessions[session_id]["checkpoints"].append({
                "timestamp": datetime.now().isoformat(),
                "state": state
            })
            logger.info(f"Checkpoint saved for session: {session_id}")


# =====================================================================
# HELPER FUNCTION TO CALL GEMINI WITH RETRY
# =====================================================================

def call_gemini_with_retry(prompt: str, max_retries: int = 3) -> str:
    """Call Gemini API with retry logic and rate limiting"""
    
    # List available models and use the first one that supports generateContent
    available_models = [
        "models/gemini-2.5-flash-lite"
    ]
    
    for attempt in range(max_retries):
        for model_name in available_models:
            try:
                logger.info(f"Attempting to use model: {model_name}")
                model = genai.GenerativeModel(model_name)
                
                # Add rate limiting delay
                time.sleep(Config.RATE_LIMIT_DELAY)
                
                response = model.generate_content(prompt)
                return response.text
                
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"Attempt {attempt + 1} with {model_name} failed: {error_msg}")
                
                # If it's a 404, try next model
                if "404" in error_msg or "not found" in error_msg:
                    continue
                
                # If it's a rate limit, wait longer
                if "429" in error_msg or "quota" in error_msg.lower():
                    wait_time = Config.RATE_LIMIT_DELAY * (attempt + 2)
                    logger.info(f"Rate limit hit, waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    break  # Try again with first model
                
                # For other errors, wait a bit and retry
                time.sleep(2)
    
    # If all retries failed, return a fallback response
    logger.error("All API calls failed, using fallback response")
    return "Unable to generate AI response due to API limitations. Using rule-based analysis only."


# =====================================================================
# AGENT DEFINITIONS
# =====================================================================

class DealAnalyzerAgent:
    """Agent specialized in analyzing deal terms"""
    
    def __init__(self):
        self.name = "DealAnalyzer"
        self.tools = BusinessTools()
        logger.info(f"{self.name} agent initialized")
    
    async def analyze(self, deal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze deal and provide recommendations"""
        logger.info(f"{self.name}: Starting deal analysis")
        
        # Perform analysis using tools
        analysis = self.tools.analyze_deal_terms(deal_data)
        
        # Use LLM for enhanced insights
        prompt = f"""Analyze this business deal and provide strategic recommendations:

Deal Data:
{json.dumps(deal_data, indent=2)}

Initial Analysis:
{json.dumps(analysis, indent=2)}

Provide a concise analysis covering:
1. Key strengths and weaknesses
2. Negotiation leverage points
3. Risk mitigation strategies
4. Final recommendation (approve/reject/negotiate)

Keep your response under 200 words."""
        
        llm_response = await asyncio.to_thread(call_gemini_with_retry, prompt)
        analysis["llm_insights"] = llm_response
        
        logger.info(f"{self.name}: Analysis complete")
        return analysis


class NegotiationAgent:
    """Agent specialized in negotiation strategy"""
    
    def __init__(self):
        self.name = "Negotiator"
        self.tools = BusinessTools()
        logger.info(f"{self.name} agent initialized")
    
    async def strategize(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Develop negotiation strategy"""
        logger.info(f"{self.name}: Developing strategy")
        
        # Calculate base strategy
        strategy = self.tools.calculate_negotiation_strategy(context)
        
        # Get market context
        industry = context.get("industry", "technology")
        market_data = self.tools.fetch_market_data(industry)
        
        # Use LLM for strategic advice
        prompt = f"""Develop a negotiation strategy for this deal:

Context: {json.dumps(context, indent=2)}
Base Strategy: {json.dumps(strategy, indent=2)}
Market Data: {json.dumps(market_data, indent=2)}

Provide a concise strategy covering:
1. Opening position
2. Key concessions to offer/request
3. Timing strategy

Keep your response under 200 words."""
        
        llm_response = await asyncio.to_thread(call_gemini_with_retry, prompt)
        strategy["detailed_plan"] = llm_response
        
        logger.info(f"{self.name}: Strategy developed")
        return strategy


class ReportingAgent:
    """Agent specialized in performance reporting"""
    
    def __init__(self):
        self.name = "Reporter"
        self.tools = BusinessTools()
        logger.info(f"{self.name} agent initialized")
    
    async def generate_report(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        logger.info(f"{self.name}: Generating report")
        
        # Generate base report
        report = self.tools.generate_performance_report(metrics)
        
        # Use LLM for executive summary
        prompt = f"""Create an executive summary for this performance report:

Metrics: {json.dumps(metrics, indent=2)}
Report Data: {json.dumps(report, indent=2)}

Provide:
1. Executive summary (3-4 key points)
2. Top 3 strategic recommendations
3. Priority action items

Keep your response under 200 words."""
        
        llm_response = await asyncio.to_thread(call_gemini_with_retry, prompt)
        report["executive_summary"] = llm_response
        
        logger.info(f"{self.name}: Report generated")
        return report


# =====================================================================
# ORCHESTRATOR
# =====================================================================

class BusinessAgentOrchestrator:
    """Main orchestrator for the multi-agent system"""
    
    def __init__(self, api_key: Optional[str] = None):
        # Initialize configuration
        Config.initialize(api_key)
        
        # Initialize components
        self.memory = BusinessMemoryBank()
        self.session_manager = BusinessSessionManager()
        
        # Initialize agents
        self.deal_analyzer = DealAnalyzerAgent()
        self.negotiator = NegotiationAgent()
        self.reporter = ReportingAgent()
        
        # Metrics for observability
        self.metrics = {
            "tasks_completed": 0,
            "total_execution_time": 0
        }
        
        logger.info("Business Agent Orchestrator initialized")
    
    async def analyze_deal_workflow(self, deal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Complete workflow for deal analysis and negotiation"""
        start_time = datetime.now()
        
        session_id = f"session-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.session_manager.create_session(session_id, {"deal_data": deal_data})
        
        results = {}
        
        try:
            # Step 1: Parallel - Fetch market data and analyze deal
            logger.info("Step 1: Parallel analysis and market research")
            
            async def get_market():
                return await asyncio.to_thread(
                    self.negotiator.tools.fetch_market_data,
                    deal_data.get("industry", "technology")
                )
            
            market_data, deal_analysis = await asyncio.gather(
                get_market(),
                self.deal_analyzer.analyze(deal_data)
            )
            
            results["market_data"] = market_data
            results["deal_analysis"] = deal_analysis
            
            self.session_manager.save_checkpoint(session_id, {"step": 1, "results": results})
            
            # Step 2: Sequential - Develop negotiation strategy
            logger.info("Step 2: Developing negotiation strategy")
            
            negotiation_context = {
                **deal_data,
                "analysis": deal_analysis,
                "market": market_data
            }
            
            strategy = await self.negotiator.strategize(negotiation_context)
            results["negotiation_strategy"] = strategy
            
            self.session_manager.save_checkpoint(session_id, {"step": 2, "results": results})
            
            # Step 3: Store in memory
            logger.info("Step 3: Storing in long-term memory")
            
            self.memory.store_deal(
                deal_data.get("deal_id", "unknown"),
                {**deal_data, "analysis": deal_analysis, "strategy": strategy}
            )
            
            # Update metrics
            execution_time = (datetime.now() - start_time).total_seconds()
            self.metrics["tasks_completed"] += 1
            self.metrics["total_execution_time"] += execution_time
            
            results["session_id"] = session_id
            results["execution_time_seconds"] = execution_time
            results["status"] = "completed"
            
            logger.info(f"Deal workflow completed in {execution_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error in deal workflow: {str(e)}")
            results["status"] = "error"
            results["error"] = str(e)
        
        return results
    
    async def generate_performance_report_workflow(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Workflow for generating performance reports"""
        start_time = datetime.now()
        
        logger.info("Starting performance report workflow")
        
        try:
            report = await self.reporter.generate_report(metrics)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                "report": report,
                "execution_time_seconds": execution_time,
                "status": "completed"
            }
            
            logger.info(f"Performance report workflow completed in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error in report workflow: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get system metrics for observability"""
        return {
            **self.metrics,
            "avg_execution_time": (
                self.metrics["total_execution_time"] / self.metrics["tasks_completed"]
                if self.metrics["tasks_completed"] > 0 else 0
            ),
            "active_sessions": len(self.session_manager.sessions),
            "stored_deals": sum(len(deals) for deals in self.memory.deal_history.values())
        }


# =====================================================================
# USAGE EXAMPLES
# =====================================================================

async def example_deal_analysis():
    """Example: Analyze a business deal"""
    
    orchestrator = BusinessAgentOrchestrator()
    
    deal_data = {
        "deal_id": "DEAL-2024-001",
        "client_id": "CLIENT-ABC",
        "industry": "technology",
        "price": 250000,
        "estimated_cost": 180000,
        "payment_terms": "Net 60",
        "duration_months": 24,
        "deal_importance": "high"
    }
    
    result = await orchestrator.analyze_deal_workflow(deal_data)
    
    print("\n" + "="*70)
    print("DEAL ANALYSIS RESULTS")
    print("="*70)
    print(json.dumps(result, indent=2, default=str))
    
    return result


async def example_performance_report():
    """Example: Generate performance report"""
    
    orchestrator = BusinessAgentOrchestrator()
    
    metrics = {
        "revenue": {
            "current": 1250000,
            "previous": 1100000
        },
        "deals": {
            "won": 15,
            "total": 72
        },
        "client_satisfaction": 4.2
    }
    
    result = await orchestrator.generate_performance_report_workflow(metrics)
    
    print("\n" + "="*70)
    print("PERFORMANCE REPORT")
    print("="*70)
    print(json.dumps(result, indent=2, default=str))
    
    return result


async def main():
    """Main execution function"""
    
    print("\n" + "="*70)
    print("BUSINESS AGENT SYSTEM - Google Generative AI")
    print("="*70)
    print("\nFeatures:")
    print("✓ Multi-agent orchestration (parallel, sequential)")
    print("✓ Custom business tools")
    print("✓ Session management")
    print("✓ Long-term memory")
    print("✓ Rate limiting to avoid quota issues")
    print("\n" + "="*70)
    
    if not Config.GEMINI_API_KEY and not os.getenv("GEMINI_API_KEY"):
        print("\n⚠️  WARNING: GEMINI_API_KEY not set!")
        print("Set your API key:")
        print("1. Edit Config.GEMINI_API_KEY in this file")
        print("2. Set environment: export GEMINI_API_KEY='your-key'")
        print("\nGet your API key: https://ai.google.dev/")
        return
    
    try:
        print("\n\n1️⃣  Running Deal Analysis Example...")
        await example_deal_analysis()
        
        print("\n\n2️⃣  Running Performance Report Example...")
        await example_performance_report()
        
        print("\n\n✅ All examples completed!")
        print("\nCheck 'agent_system.log' for detailed logs.")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        print(f"\n❌ Error: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())
