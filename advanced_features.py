"""
Advanced Features for Business Agent System

This module demonstrates additional advanced concepts:
- MCP (Model Context Protocol) integration
- OpenAPI tool integration
- Advanced observability
- Context engineering
- Agent evaluation
- A2A (Agent-to-Agent) protocol
"""

import json
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import hashlib


# =====================================================================
# MCP (Model Context Protocol) Integration
# =====================================================================

class MCPToolManager:
    """
    Manages MCP (Model Context Protocol) tools
    MCP enables standardized communication between AI models and external tools
    """
    
    def __init__(self):
        self.registered_tools: Dict[str, Dict] = {}
        self.tool_schemas: Dict[str, Dict] = {}
    
    def register_mcp_tool(self, tool_name: str, schema: Dict[str, Any], 
                          handler: callable):
        """Register an MCP-compliant tool"""
        self.registered_tools[tool_name] = {
            "handler": handler,
            "schema": schema,
            "registered_at": datetime.now().isoformat()
        }
        self.tool_schemas[tool_name] = schema
        print(f"✓ MCP Tool registered: {tool_name}")
    
    async def execute_mcp_tool(self, tool_name: str, parameters: Dict[str, Any]):
        """Execute an MCP tool with parameters"""
        if tool_name not in self.registered_tools:
            raise ValueError(f"MCP tool not found: {tool_name}")
        
        handler = self.registered_tools[tool_name]["handler"]
        result = await handler(parameters)
        
        return {
            "tool": tool_name,
            "result": result,
            "executed_at": datetime.now().isoformat()
        }
    
    def get_tool_schema(self, tool_name: str) -> Optional[Dict]:
        """Get schema for an MCP tool"""
        return self.tool_schemas.get(tool_name)
    
    def list_available_tools(self) -> List[str]:
        """List all registered MCP tools"""
        return list(self.registered_tools.keys())


# Example MCP Tool Implementation
async def crm_lookup_tool(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Example MCP tool: Look up client data from CRM"""
    client_id = parameters.get("client_id")
    
    # Simulate CRM lookup
    mock_data = {
        "client_id": client_id,
        "name": "Acme Corporation",
        "industry": "Technology",
        "total_deals": 15,
        "lifetime_value": 2500000,
        "satisfaction_score": 4.5,
        "contact": {
            "name": "John Smith",
            "email": "john.smith@acme.com",
            "phone": "+1-555-0123"
        }
    }
    
    return mock_data


# Register the MCP tool
mcp_manager = MCPToolManager()
mcp_manager.register_mcp_tool(
    "crm_lookup",
    schema={
        "name": "crm_lookup",
        "description": "Look up client information from CRM system",
        "parameters": {
            "type": "object",
            "properties": {
                "client_id": {
                    "type": "string",
                    "description": "Unique client identifier"
                }
            },
            "required": ["client_id"]
        }
    },
    handler=crm_lookup_tool
)


# =====================================================================
# OpenAPI Tool Integration
# =====================================================================

class OpenAPIToolManager:
    """
    Manages OpenAPI-based tools
    Enables integration with any REST API that has an OpenAPI spec
    """
    
    def __init__(self):
        self.api_specs: Dict[str, Dict] = {}
        self.api_endpoints: Dict[str, List[Dict]] = {}
    
    def register_openapi_spec(self, service_name: str, spec_url: str, 
                             spec_content: Optional[Dict] = None):
        """Register an OpenAPI specification"""
        self.api_specs[service_name] = {
            "spec_url": spec_url,
            "spec_content": spec_content,
            "registered_at": datetime.now().isoformat()
        }
        
        # Parse endpoints from spec
        if spec_content:
            self._parse_endpoints(service_name, spec_content)
        
        print(f"✓ OpenAPI spec registered: {service_name}")
    
    def _parse_endpoints(self, service_name: str, spec: Dict):
        """Parse endpoints from OpenAPI spec"""
        endpoints = []
        paths = spec.get("paths", {})
        
        for path, methods in paths.items():
            for method, details in methods.items():
                if method in ["get", "post", "put", "delete", "patch"]:
                    endpoints.append({
                        "path": path,
                        "method": method.upper(),
                        "operation_id": details.get("operationId"),
                        "summary": details.get("summary"),
                        "parameters": details.get("parameters", [])
                    })
        
        self.api_endpoints[service_name] = endpoints
    
    def get_available_operations(self, service_name: str) -> List[Dict]:
        """Get available operations for a service"""
        return self.api_endpoints.get(service_name, [])
    
    async def execute_operation(self, service_name: str, operation_id: str, 
                               parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an API operation"""
        # In production, this would make actual HTTP requests
        # Here we simulate the response
        
        return {
            "service": service_name,
            "operation": operation_id,
            "parameters": parameters,
            "result": {"status": "success", "data": {}},
            "executed_at": datetime.now().isoformat()
        }


# Example OpenAPI integration
openapi_manager = OpenAPIToolManager()

# Register a mock market data API
market_data_spec = {
    "openapi": "3.0.0",
    "info": {"title": "Market Data API", "version": "1.0.0"},
    "paths": {
        "/market/{industry}/trends": {
            "get": {
                "operationId": "getIndustryTrends",
                "summary": "Get market trends for an industry",
                "parameters": [
                    {"name": "industry", "in": "path", "required": True}
                ]
            }
        },
        "/market/pricing": {
            "post": {
                "operationId": "getPricingData",
                "summary": "Get pricing benchmarks",
                "parameters": []
            }
        }
    }
}

openapi_manager.register_openapi_spec(
    "market_data_api",
    "https://api.example.com/openapi.json",
    market_data_spec
)


# =====================================================================
# Advanced Observability
# =====================================================================

class ObservabilityManager:
    """
    Advanced observability with metrics, tracing, and monitoring
    """
    
    def __init__(self):
        self.traces: List[Dict] = []
        self.metrics: Dict[str, List[float]] = {}
        self.spans: Dict[str, Dict] = {}
    
    def start_trace(self, trace_id: str, operation: str):
        """Start a new trace"""
        trace = {
            "trace_id": trace_id,
            "operation": operation,
            "start_time": time.time(),
            "spans": []
        }
        self.traces.append(trace)
        return trace_id
    
    def start_span(self, trace_id: str, span_name: str, parent_span: Optional[str] = None):
        """Start a new span within a trace"""
        span_id = hashlib.md5(f"{trace_id}-{span_name}-{time.time()}".encode()).hexdigest()[:8]
        
        span = {
            "span_id": span_id,
            "span_name": span_name,
            "trace_id": trace_id,
            "parent_span": parent_span,
            "start_time": time.time(),
            "end_time": None,
            "attributes": {}
        }
        
        self.spans[span_id] = span
        return span_id
    
    def end_span(self, span_id: str, attributes: Optional[Dict] = None):
        """End a span"""
        if span_id in self.spans:
            self.spans[span_id]["end_time"] = time.time()
            self.spans[span_id]["duration"] = (
                self.spans[span_id]["end_time"] - self.spans[span_id]["start_time"]
            )
            
            if attributes:
                self.spans[span_id]["attributes"].update(attributes)
    
    def record_metric(self, metric_name: str, value: float):
        """Record a metric value"""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        
        self.metrics[metric_name].append(value)
    
    def get_trace(self, trace_id: str) -> Optional[Dict]:
        """Get trace by ID"""
        for trace in self.traces:
            if trace["trace_id"] == trace_id:
                # Attach spans
                trace["spans"] = [
                    span for span in self.spans.values() 
                    if span["trace_id"] == trace_id
                ]
                return trace
        return None
    
    def get_metric_stats(self, metric_name: str) -> Dict[str, float]:
        """Get statistics for a metric"""
        if metric_name not in self.metrics:
            return {}
        
        values = self.metrics[metric_name]
        return {
            "count": len(values),
            "sum": sum(values),
            "avg": sum(values) / len(values) if values else 0,
            "min": min(values) if values else 0,
            "max": max(values) if values else 0
        }
    
    def export_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format"""
        output = []
        
        for metric_name, values in self.metrics.items():
            stats = self.get_metric_stats(metric_name)
            
            output.append(f"# HELP {metric_name} Metric: {metric_name}")
            output.append(f"# TYPE {metric_name} gauge")
            output.append(f"{metric_name}_total {stats['sum']}")
            output.append(f"{metric_name}_count {stats['count']}")
            output.append(f"{metric_name}_avg {stats['avg']}")
            output.append("")
        
        return "\n".join(output)


# =====================================================================
# Context Engineering
# =====================================================================

class ContextCompactor:
    """
    Intelligently compact context to fit within token limits
    Uses summarization and pruning strategies
    """
    
    def __init__(self, max_tokens: int = 30000):
        self.max_tokens = max_tokens
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)"""
        # Rough estimation: 1 token ≈ 4 characters
        return len(text) // 4
    
    def compact_context(self, context: Dict[str, Any], 
                       priority_keys: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Compact context by prioritizing important information
        """
        if priority_keys is None:
            priority_keys = ["deal_id", "client_id", "price", "status"]
        
        # Serialize context
        full_context_str = json.dumps(context, indent=2)
        current_tokens = self.estimate_tokens(full_context_str)
        
        if current_tokens <= self.max_tokens:
            return context  # No compaction needed
        
        # Create compacted version
        compacted = {}
        
        # Keep priority keys
        for key in priority_keys:
            if key in context:
                compacted[key] = context[key]
        
        # Summarize large text fields
        for key, value in context.items():
            if key not in compacted:
                if isinstance(value, str) and len(value) > 500:
                    # Truncate long strings
                    compacted[key] = value[:250] + "..." + value[-250:]
                elif isinstance(value, list) and len(value) > 10:
                    # Keep first and last items for lists
                    compacted[key] = value[:5] + ["..."] + value[-5:]
                elif isinstance(value, dict):
                    # Recursively compact nested dicts
                    compacted[key] = self._compact_dict(value)
                else:
                    compacted[key] = value
        
        # Add metadata about compaction
        compacted["_compacted"] = True
        compacted["_original_size_estimate"] = current_tokens
        
        return compacted
    
    def _compact_dict(self, d: Dict, max_keys: int = 10) -> Dict:
        """Compact a dictionary by limiting keys"""
        if len(d) <= max_keys:
            return d
        
        keys = list(d.keys())[:max_keys]
        compacted = {k: d[k] for k in keys}
        compacted["_truncated"] = f"{len(d) - max_keys} more keys"
        
        return compacted


# =====================================================================
# Agent Evaluation
# =====================================================================

@dataclass
class EvaluationMetrics:
    """Metrics for agent evaluation"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    latency: float
    cost: float
    user_satisfaction: float


class AgentEvaluator:
    """
    Evaluate agent performance against benchmarks
    """
    
    def __init__(self):
        self.evaluation_results: List[Dict] = []
    
    def evaluate_deal_analysis(self, predicted: Dict, ground_truth: Dict) -> EvaluationMetrics:
        """Evaluate deal analysis accuracy"""
        
        # Calculate accuracy
        correct_predictions = 0
        total_predictions = 0
        
        if "status" in predicted and "status" in ground_truth:
            total_predictions += 1
            if predicted["status"] == ground_truth["status"]:
                correct_predictions += 1
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        # Calculate other metrics (simplified)
        metrics = EvaluationMetrics(
            accuracy=accuracy,
            precision=accuracy,  # Simplified
            recall=accuracy,
            f1_score=accuracy,
            latency=predicted.get("execution_time_seconds", 0),
            cost=0.0,  # Would calculate based on token usage
            user_satisfaction=0.0  # Would gather from user feedback
        )
        
        return metrics
    
    def evaluate_negotiation_strategy(self, strategy: Dict, 
                                     outcome: Dict) -> EvaluationMetrics:
        """Evaluate negotiation strategy effectiveness"""
        
        # Check if strategy led to successful outcome
        success = outcome.get("deal_accepted", False)
        
        # Calculate price optimization
        target_price = strategy.get("target_outcomes", {}).get("price", 0)
        actual_price = outcome.get("final_price", 0)
        
        price_accuracy = 1 - abs(target_price - actual_price) / target_price if target_price > 0 else 0
        
        metrics = EvaluationMetrics(
            accuracy=1.0 if success else 0.0,
            precision=price_accuracy,
            recall=price_accuracy,
            f1_score=price_accuracy,
            latency=outcome.get("execution_time_seconds", 0),
            cost=0.0,
            user_satisfaction=0.0
        )
        
        return metrics
    
    def run_benchmark_suite(self, agent_function: callable, 
                           test_cases: List[Dict]) -> Dict[str, Any]:
        """Run a suite of benchmark tests"""
        results = []
        
        for test_case in test_cases:
            input_data = test_case["input"]
            expected_output = test_case["expected"]
            
            # Run agent
            start_time = time.time()
            actual_output = agent_function(input_data)
            execution_time = time.time() - start_time
            
            # Evaluate
            metrics = self.evaluate_deal_analysis(actual_output, expected_output)
            
            results.append({
                "test_case_id": test_case.get("id"),
                "metrics": metrics,
                "execution_time": execution_time
            })
        
        # Aggregate results
        avg_metrics = {
            "avg_accuracy": sum(r["metrics"].accuracy for r in results) / len(results),
            "avg_latency": sum(r["execution_time"] for r in results) / len(results),
            "total_tests": len(results),
            "passed_tests": sum(1 for r in results if r["metrics"].accuracy > 0.8)
        }
        
        return {
            "summary": avg_metrics,
            "detailed_results": results
        }


# =====================================================================
# A2A (Agent-to-Agent) Protocol
# =====================================================================

class A2AMessage:
    """Message format for agent-to-agent communication"""
    
    def __init__(self, sender: str, receiver: str, message_type: str, 
                 content: Any, priority: str = "normal"):
        self.message_id = hashlib.md5(
            f"{sender}-{receiver}-{time.time()}".encode()
        ).hexdigest()[:12]
        self.sender = sender
        self.receiver = receiver
        self.message_type = message_type
        self.content = content
        self.priority = priority
        self.timestamp = datetime.now().isoformat()
        self.status = "pending"
    
    def to_dict(self) -> Dict:
        return {
            "message_id": self.message_id,
            "sender": self.sender,
            "receiver": self.receiver,
            "message_type": self.message_type,
            "content": self.content,
            "priority": self.priority,
            "timestamp": self.timestamp,
            "status": self.status
        }


class A2AProtocol:
    """
    Agent-to-Agent communication protocol
    Enables direct communication between agents
    """
    
    def __init__(self):
        self.message_queue: List[A2AMessage] = []
        self.message_history: List[A2AMessage] = []
        self.agent_registry: Dict[str, Any] = {}
    
    def register_agent(self, agent_name: str, agent_instance: Any):
        """Register an agent for A2A communication"""
        self.agent_registry[agent_name] = {
            "instance": agent_instance,
            "registered_at": datetime.now().isoformat(),
            "message_count": 0
        }
        print(f"✓ Agent registered for A2A: {agent_name}")
    
    def send_message(self, sender: str, receiver: str, message_type: str, 
                    content: Any, priority: str = "normal") -> str:
        """Send a message from one agent to another"""
        
        if receiver not in self.agent_registry:
            raise ValueError(f"Receiver agent not registered: {receiver}")
        
        message = A2AMessage(sender, receiver, message_type, content, priority)
        self.message_queue.append(message)
        
        print(f"📨 Message sent: {sender} → {receiver} ({message.message_id})")
        return message.message_id
    
    async def deliver_messages(self):
        """Deliver pending messages to receiving agents"""
        
        # Sort by priority
        priority_order = {"high": 0, "normal": 1, "low": 2}
        self.message_queue.sort(key=lambda m: priority_order.get(m.priority, 1))
        
        delivered = []
        
        for message in self.message_queue:
            receiver = self.agent_registry.get(message.receiver)
            
            if receiver:
                # Deliver message to agent
                agent_instance = receiver["instance"]
                
                # Call agent's message handler if available
                if hasattr(agent_instance, "handle_message"):
                    await agent_instance.handle_message(message)
                
                message.status = "delivered"
                receiver["message_count"] += 1
                delivered.append(message)
                
                print(f"✓ Message delivered: {message.message_id}")
        
        # Move delivered messages to history
        for message in delivered:
            self.message_queue.remove(message)
            self.message_history.append(message)
    
    def get_message_history(self, agent_name: str, limit: int = 10) -> List[Dict]:
        """Get message history for an agent"""
        messages = [
            m.to_dict() for m in self.message_history
            if m.sender == agent_name or m.receiver == agent_name
        ]
        return messages[-limit:]
    
    def get_agent_stats(self, agent_name: str) -> Dict:
        """Get communication statistics for an agent"""
        if agent_name not in self.agent_registry:
            return {}
        
        sent = sum(1 for m in self.message_history if m.sender == agent_name)
        received = sum(1 for m in self.message_history if m.receiver == agent_name)
        
        return {
            "agent_name": agent_name,
            "messages_sent": sent,
            "messages_received": received,
            "total_messages": sent + received
        }


# =====================================================================
# Example Usage of Advanced Features
# =====================================================================

async def demonstrate_advanced_features():
    """Demonstrate all advanced features"""
    
    print("\n" + "="*70)
    print("ADVANCED FEATURES DEMONSTRATION")
    print("="*70)
    
    # 1. MCP Tools
    print("\n1️⃣  MCP Tool Integration")
    print("-" * 70)
    result = await mcp_manager.execute_mcp_tool("crm_lookup", {"client_id": "CLIENT-001"})
    print(f"CRM Lookup Result: {json.dumps(result, indent=2)}")
    
    # 2. OpenAPI Integration
    print("\n2️⃣  OpenAPI Integration")
    print("-" * 70)
    operations = openapi_manager.get_available_operations("market_data_api")
    print(f"Available Operations: {len(operations)}")
    for op in operations:
        print(f"  - {op['method']} {op['path']}: {op['summary']}")
    
    # 3. Advanced Observability
    print("\n3️⃣  Advanced Observability")
    print("-" * 70)
    obs = ObservabilityManager()
    trace_id = obs.start_trace("trace-001", "deal_analysis")
    span_id = obs.start_span(trace_id, "fetch_market_data")
    time.sleep(0.1)  # Simulate work
    obs.end_span(span_id, {"status": "success", "records": 100})
    obs.record_metric("deal_analysis_duration", 0.15)
    print(f"Trace created: {trace_id}")
    print(f"Metrics: {obs.get_metric_stats('deal_analysis_duration')}")
    
    # 4. Context Engineering
    print("\n4️⃣  Context Engineering")
    print("-" * 70)
    compactor = ContextCompactor(max_tokens=1000)
    large_context = {
        "deal_id": "DEAL-001",
        "description": "A" * 5000,  # Very long description
        "history": list(range(100)),  # Long list
        "details": {"nested": {"data": "x" * 1000}}
    }
    compacted = compactor.compact_context(large_context)
    print(f"Original size estimate: {compactor.estimate_tokens(json.dumps(large_context))} tokens")
    print(f"Compacted size estimate: {compactor.estimate_tokens(json.dumps(compacted))} tokens")
    print(f"Compaction ratio: {len(json.dumps(compacted)) / len(json.dumps(large_context)):.2%}")
    
    # 5. Agent Evaluation
    print("\n5️⃣  Agent Evaluation")
    print("-" * 70)
    evaluator = AgentEvaluator()
    predicted = {"status": "approved", "risk_score": 25, "execution_time_seconds": 1.2}
    ground_truth = {"status": "approved", "risk_score": 30}
    metrics = evaluator.evaluate_deal_analysis(predicted, ground_truth)
    print(f"Evaluation Metrics:")
    print(f"  Accuracy: {metrics.accuracy:.2%}")
    print(f"  Latency: {metrics.latency:.2f}s")
    
    # 6. A2A Protocol
    print("\n6️⃣  Agent-to-Agent Protocol")
    print("-" * 70)
    a2a = A2AProtocol()
    
    # Register mock agents
    class MockAgent:
        def __init__(self, name):
            self.name = name
        async def handle_message(self, message):
            print(f"  [{self.name}] Received: {message.message_type}")
    
    a2a.register_agent("Analyzer", MockAgent("Analyzer"))
    a2a.register_agent("Negotiator", MockAgent("Negotiator"))
    
    # Send messages
    a2a.send_message("Analyzer", "Negotiator", "deal_analysis_complete", 
                    {"deal_id": "DEAL-001", "status": "approved"})
    
    # Deliver messages
    await a2a.deliver_messages()
    
    stats = a2a.get_agent_stats("Analyzer")
    print(f"Agent Stats: {json.dumps(stats, indent=2)}")
    
    print("\n" + "="*70)
    print("✅ All advanced features demonstrated successfully!")
    print("="*70)


if __name__ == "__main__":
    import asyncio
    asyncio.run(demonstrate_advanced_features())
