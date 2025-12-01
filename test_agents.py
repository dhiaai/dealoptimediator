"""
Unit tests for Business Agent System
Run with: pytest tests/test_agents.py
"""

import pytest
import asyncio
from unittest.mock import Mock, patch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import (
    BusinessAgentOrchestrator,
    DealAnalyzerAgent,
    NegotiationAgent,
    ReportingAgent,
    CoordinatorAgent,
    BusinessTools,
    BusinessMemoryBank,
    BusinessSessionManager,
    Config
)


@pytest.fixture
def mock_genai_client():
    """Mock Gemini client for testing"""
    client = Mock()
    client.models.generate_content = Mock(return_value=Mock(text="Mock LLM response"))
    return client


@pytest.fixture
def orchestrator():
    """Create orchestrator instance for testing"""
    with patch('main.genai.Client'):
        Config.GEMINI_API_KEY = "test_key"
        return BusinessAgentOrchestrator(api_key="test_key")


class TestBusinessTools:
    """Test custom business tools"""
    
    def test_analyze_deal_terms_profitable(self):
        """Test deal analysis with profitable terms"""
        deal_data = {
            "deal_id": "TEST-001",
            "price": 100000,
            "estimated_cost": 70000,
            "payment_terms": "Net 30",
            "duration_months": 12
        }
        
        result = BusinessTools.analyze_deal_terms(deal_data)
        
        assert result["deal_id"] == "TEST-001"
        assert result["profitability_score"] > 0
        assert result["status"] == "approved"
        assert len(result["recommendations"]) > 0
    
    def test_analyze_deal_terms_risky(self):
        """Test deal analysis with risky terms"""
        deal_data = {
            "deal_id": "TEST-002",
            "price": 100000,
            "estimated_cost": 95000,  # Low margin
            "payment_terms": "Net 90",  # Extended terms
            "duration_months": 48  # Long duration
        }
        
        result = BusinessTools.analyze_deal_terms(deal_data)
        
        assert result["risk_score"] > 50
        assert result["status"] == "needs_review"
        assert len(result["red_flags"]) > 0
    
    def test_generate_performance_report(self):
        """Test performance report generation"""
        metrics = {
            "revenue": {"current": 1200000, "previous": 1000000},
            "deals": {"won": 20, "total": 100},
            "client_satisfaction": 4.5
        }
        
        result = BusinessTools.generate_performance_report(metrics)
        
        assert "report_id" in result
        assert result["summary"]["revenue_growth"] == 20.0
        assert result["summary"]["conversion_rate"] == 20.0
        assert result["summary"]["avg_satisfaction"] == 4.5
    
    def test_calculate_negotiation_strategy_high_importance(self):
        """Test strategy calculation for high importance deal"""
        context = {
            "deal_importance": "high",
            "base_price": 200000,
            "target_price": 190000
        }
        
        result = BusinessTools.calculate_negotiation_strategy(context)
        
        assert result["approach"] == "collaborative"
        assert "max_discount" in result["concession_limits"]
        assert result["target_outcomes"]["price"] == 190000
    
    def test_fetch_market_data(self):
        """Test market data fetching"""
        result = BusinessTools.fetch_market_data("technology", "north_america")
        
        assert result["industry"] == "technology"
        assert result["region"] == "north_america"
        assert "avg_deal_size" in result
        assert "benchmark_metrics" in result


class TestMemoryBank:
    """Test memory management"""
    
    def test_store_and_retrieve_deal(self):
        """Test storing and retrieving deal history"""
        memory = BusinessMemoryBank()
        
        deal_data = {
            "client_id": "CLIENT-001",
            "price": 150000,
            "status": "won"
        }
        
        memory.store_deal("DEAL-001", deal_data)
        history = memory.get_client_history("CLIENT-001")
        
        assert len(history) == 1
        assert history[0]["deal_id"] == "DEAL-001"
        assert history[0]["data"]["price"] == 150000
    
    def test_client_profile_management(self):
        """Test client profile updates"""
        memory = BusinessMemoryBank()
        
        profile = {
            "name": "Acme Corp",
            "industry": "technology",
            "lifetime_value": 500000
        }
        
        memory.update_client_profile("CLIENT-001", profile)
        
        assert "CLIENT-001" in memory.client_profiles
        assert memory.client_profiles["CLIENT-001"]["industry"] == "technology"
    
    def test_relevant_context_retrieval(self):
        """Test context retrieval"""
        memory = BusinessMemoryBank()
        
        # Store multiple deals
        for i in range(10):
            memory.store_deal(
                f"DEAL-{i:03d}",
                {"client_id": f"CLIENT-{i % 3}", "price": 100000 + i * 10000}
            )
        
        context = memory.get_relevant_context("test query", limit=5)
        
        assert len(context) <= 5


class TestSessionManager:
    """Test session management"""
    
    def test_create_session(self):
        """Test session creation"""
        manager = BusinessSessionManager()
        
        context = {"deal_id": "DEAL-001"}
        session_id = manager.create_session("session-001", context)
        
        assert session_id == "session-001"
        assert "session-001" in manager.sessions
        assert manager.sessions["session-001"]["state"] == "active"
    
    def test_save_checkpoint(self):
        """Test checkpoint saving"""
        manager = BusinessSessionManager()
        manager.create_session("session-001", {})
        
        state = {"step": 1, "data": "test"}
        manager.save_checkpoint("session-001", state)
        
        checkpoints = manager.sessions["session-001"]["checkpoints"]
        assert len(checkpoints) == 1
        assert checkpoints[0]["state"] == state
    
    def test_pause_resume_session(self):
        """Test pause and resume functionality"""
        manager = BusinessSessionManager()
        manager.create_session("session-001", {})
        
        # Save checkpoint
        state = {"progress": 50}
        manager.save_checkpoint("session-001", state)
        
        # Pause
        manager.pause_session("session-001")
        assert manager.sessions["session-001"]["state"] == "paused"
        
        # Resume
        restored_state = manager.resume_session("session-001")
        assert manager.sessions["session-001"]["state"] == "active"
        assert restored_state["state"]["progress"] == 50


@pytest.mark.asyncio
class TestAgents:
    """Test agent functionality (with mocked LLM calls)"""
    
    async def test_deal_analyzer_agent(self, mock_genai_client):
        """Test deal analyzer agent"""
        agent = DealAnalyzerAgent(mock_genai_client)
        
        deal_data = {
            "deal_id": "TEST-001",
            "price": 150000,
            "estimated_cost": 100000
        }
        
        result = await agent.analyze(deal_data)
        
        assert "llm_insights" in result
        assert result["deal_id"] == "TEST-001"
    
    async def test_negotiation_agent(self, mock_genai_client):
        """Test negotiation agent"""
        agent = NegotiationAgent(mock_genai_client)
        
        context = {
            "deal_importance": "high",
            "base_price": 200000,
            "industry": "technology"
        }
        
        result = await agent.strategize(context)
        
        assert "detailed_plan" in result
        assert result["approach"] in ["collaborative", "competitive", "balanced"]
    
    async def test_reporting_agent(self, mock_genai_client):
        """Test reporting agent"""
        agent = ReportingAgent(mock_genai_client)
        
        metrics = {
            "revenue": {"current": 1500000, "previous": 1200000},
            "deals": {"won": 25, "total": 100}
        }
        
        result = await agent.generate_report(metrics)
        
        assert "executive_summary" in result
        assert "report_id" in result
    
    async def test_coordinator_agent(self, mock_genai_client):
        """Test coordinator agent"""
        agent = CoordinatorAgent(mock_genai_client)
        
        result = await agent.coordinate(
            "Analyze and negotiate a deal",
            {"deal_id": "TEST-001"}
        )
        
        assert "execution_plan" in result
        assert result["task"] == "Analyze and negotiate a deal"


@pytest.mark.asyncio
class TestOrchestrator:
    """Test orchestrator functionality"""
    
    async def test_parallel_execution(self, orchestrator):
        """Test parallel task execution"""
        async def task1():
            await asyncio.sleep(0.1)
            return {"result": 1}
        
        async def task2():
            await asyncio.sleep(0.1)
            return {"result": 2}
        
        tasks = [(task1,), (task2,)]
        results = await orchestrator.execute_parallel(tasks)
        
        assert len(results) == 2
        assert results[0]["result"] == 1
        assert results[1]["result"] == 2
    
    async def test_sequential_execution(self, orchestrator):
        """Test sequential task execution with context passing"""
        async def task1(context):
            return {"value": 10}
        
        async def task2(context):
            return {"value": context.get("value", 0) + 5}
        
        tasks = [(task1, {}), (task2, {})]
        results = await orchestrator.execute_sequential(tasks)
        
        assert len(results) == 2
        assert results[1]["value"] == 15
    
    async def test_loop_execution(self, orchestrator):
        """Test loop execution with termination condition"""
        counter = {"value": 0}
        
        async def increment():
            counter["value"] += 1
            return {"iteration": counter["value"]}
        
        def should_stop(result):
            return result["iteration"] >= 3
        
        results = await orchestrator.execute_loop(increment, should_stop, max_iterations=10)
        
        assert len(results) == 3
        assert counter["value"] == 3
    
    def test_metrics_collection(self, orchestrator):
        """Test metrics collection"""
        orchestrator.metrics["tasks_completed"] = 5
        orchestrator.metrics["total_execution_time"] = 25.0
        
        metrics = orchestrator.get_metrics()
        
        assert metrics["tasks_completed"] == 5
        assert metrics["avg_execution_time"] == 5.0


@pytest.mark.asyncio
class TestWorkflows:
    """Test complete workflows (integration tests with mocks)"""
    
    @patch('main.genai.Client')
    async def test_deal_analysis_workflow(self, mock_client):
        """Test complete deal analysis workflow"""
        Config.GEMINI_API_KEY = "test_key"
        orchestrator = BusinessAgentOrchestrator(api_key="test_key")
        
        deal_data = {
            "deal_id": "WORKFLOW-001",
            "client_id": "CLIENT-001",
            "industry": "technology",
            "price": 250000,
            "estimated_cost": 180000,
            "payment_terms": "Net 30",
            "duration_months": 24
        }
        
        # Mock the LLM calls
        mock_response = Mock()
        mock_response.text = "Mock analysis complete"
        mock_client.return_value.models.generate_content.return_value = mock_response
        
        result = await orchestrator.analyze_deal_workflow(deal_data)
        
        assert result["status"] in ["completed", "error"]
        if result["status"] == "completed":
            assert "market_data" in result
            assert "deal_analysis" in result
            assert "negotiation_strategy" in result
            assert "session_id" in result


def test_config_initialization():
    """Test configuration initialization"""
    test_key = "test_api_key_12345"
    Config.initialize(test_key)
    
    assert Config.GEMINI_API_KEY == test_key


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
