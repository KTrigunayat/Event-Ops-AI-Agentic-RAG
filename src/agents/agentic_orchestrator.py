import os
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import google.generativeai as genai
from enum import Enum

class IntentType(Enum):
    """Types of user intents"""
    LOOKUP = "lookup"  # Simple information retrieval
    DECISION = "decision"  # Complex problem solving
    COORDINATION = "coordination"  # Multi-party coordination
    EMERGENCY = "emergency"  # Crisis management

@dataclass
class UserQuery:
    """Represents a user query with context"""
    query: str
    user_id: str
    event_id: str
    timestamp: str
    urgency: str  # "low", "medium", "high", "critical"
    context: Dict[str, Any] = None

@dataclass
class IntentAnalysis:
    """Result of intent analysis"""
    intent_type: IntentType
    confidence: float
    entities: Dict[str, Any]
    keywords: List[str]
    reasoning: str

@dataclass
class AgentAction:
    """Represents an action the agent can take"""
    action_type: str
    parameters: Dict[str, Any]
    reasoning: str
    priority: str

class AgenticOrchestrator:
    """Main agentic orchestrator using Gemini 1.5 Flash"""
    
    def __init__(self, api_key: str = None, model: str = "gemini-1.5-flash"):
        self.api_key = api_key or os.getenv("GEMINI_FLASH_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini Flash API key not provided")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model)
        self.logger = logging.getLogger(__name__)
        
        # Safety configurations
        self.safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            }
        ]
        
        self.logger.info(f"Initialized Agentic Orchestrator with model: {model}")
    
    def analyze_intent(self, query: UserQuery) -> IntentAnalysis:
        """
        Analyze user intent to determine the type of response needed
        
        Args:
            query: UserQuery object
            
        Returns:
            IntentAnalysis with classification results
        """
        prompt = f"""
You are an expert event operations coordinator analyzing on-ground personnel queries.

USER QUERY: "{query.query}"
USER ID: {query.user_id}
EVENT ID: {query.event_id}
URGENCY: {query.urgency}
CONTEXT: {query.context or {}}

Analyze this query and determine:
1. Intent Type - Choose ONE: {', '.join([intent.value for intent in IntentType])}
2. Confidence (0.0-1.0)
3. Key entities (people, places, times, items mentioned)
4. Important keywords
5. Reasoning for classification

INTENT DEFINITIONS:
- LOOKUP: Simple information request (Who is vendor? What time is setup?)
- DECISION: Complex problem requiring planning (How handle delay? What backup plan?)
- COORDINATION: Multiple parties need to be involved (Coordinate with venue staff)
- EMERGENCY: Critical issue requiring immediate action (Power outage, medical emergency)

Return your analysis as JSON:
{{
    "intent_type": "decision",
    "confidence": 0.85,
    "entities": {{"vendor": "catering", "time": "6:00 PM", "issue": "traffic"}},
    "keywords": ["catering", "stuck", "traffic", "delay"],
    "reasoning": "User reports a delay issue requiring contingency planning"
}}
"""
        
        try:
            response = self.model.generate_content(
                prompt,
                safety_settings=self.safety_settings
            )
            
            # Parse response
            import json
            result = json.loads(response.text)
            
            return IntentAnalysis(
                intent_type=IntentType(result["intent_type"]),
                confidence=float(result["confidence"]),
                entities=result["entities"],
                keywords=result["keywords"],
                reasoning=result["reasoning"]
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing intent: {e}")
            # Fallback to simple classification
            return self._fallback_intent_analysis(query)
    
    def _fallback_intent_analysis(self, query: UserQuery) -> IntentAnalysis:
        """Fallback intent analysis using keyword patterns"""
        query_lower = query.query.lower()
        
        # Emergency keywords
        emergency_keywords = ["emergency", "crisis", "accident", "injury", "fire", "power outage", "medical"]
        if any(keyword in query_lower for keyword in emergency_keywords):
            return IntentAnalysis(
                intent_type=IntentType.EMERGENCY,
                confidence=0.7,
                entities={},
                keywords=[kw for kw in emergency_keywords if kw in query_lower],
                reasoning="Emergency keywords detected"
            )
        
        # Decision keywords
        decision_keywords = ["what if", "how to", "plan", "backup", "alternative", "delay", "stuck", "problem"]
        if any(keyword in query_lower for keyword in decision_keywords):
            return IntentAnalysis(
                intent_type=IntentType.DECISION,
                confidence=0.6,
                entities={},
                keywords=[kw for kw in decision_keywords if kw in query_lower],
                reasoning="Decision-making keywords detected"
            )
        
        # Coordination keywords
        coordination_keywords = ["coordinate", "contact", "inform", "notify", "team", "staff", "vendor"]
        if any(keyword in query_lower for keyword in coordination_keywords):
            return IntentAnalysis(
                intent_type=IntentType.COORDINATION,
                confidence=0.6,
                entities={},
                keywords=[kw for kw in coordination_keywords if kw in query_lower],
                reasoning="Coordination keywords detected"
            )
        
        # Default to lookup
        return IntentAnalysis(
            intent_type=IntentType.LOOKUP,
            confidence=0.5,
            entities={},
            keywords=[],
            reasoning="Default classification as lookup query"
        )
    
    def plan_actions(self, query: UserQuery, intent: IntentAnalysis, 
                    retrieved_context: List[Dict[str, Any]]) -> List[AgentAction]:
        """
        Plan actions based on intent and retrieved context
        
        Args:
            query: User query
            intent: Intent analysis result
            retrieved_context: Context from vector database
            
        Returns:
            List of AgentAction objects
        """
        context_text = "\n".join([ctx.get("content", "") for ctx in retrieved_context[:5]])
        
        prompt = f"""
You are an expert event operations manager planning actions for an on-ground issue.

USER QUERY: "{query.query}"
INTENT TYPE: {intent.intent_type.value}
CONFIDENCE: {intent.confidence}
KEY ENTITIES: {intent.entities}
KEYWORDS: {intent.keywords}

RETRIEVED CONTEXT:
{context_text}

Plan specific actions to address this query. Each action should be:
1. Specific and actionable
2. Within operational scope (no financial commitments)
3. Prioritized by urgency and importance
4. Based on the retrieved context

Available action types:
- get_vendor_contact: Get contact information for a vendor
- suggest_pivot_plan: Suggest alternative arrangements
- coordinate_staff: Coordinate with event staff
- check_timeline: Check event timeline for conflicts
- notify_stakeholder: Notify relevant stakeholders
- document_issue: Document the issue for follow-up

Return as JSON array:
[
    {{
        "action_type": "get_vendor_contact",
        "parameters": {{"vendor": "catering", "reason": "traffic delay"}},
        "reasoning": "Need to contact catering vendor about delay",
        "priority": "high"
    }}
]

SAFETY CONSTRAINTS:
- Never commit to financial expenditures
- Always suggest human approval for major decisions
- Prioritize safety and compliance
- Consider backup options
"""
        
        try:
            response = self.model.generate_content(
                prompt,
                safety_settings=self.safety_settings
            )
            
            import json
            actions_data = json.loads(response.text)
            
            actions = []
            for action_data in actions_data:
                action = AgentAction(
                    action_type=action_data["action_type"],
                    parameters=action_data["parameters"],
                    reasoning=action_data["reasoning"],
                    priority=action_data["priority"]
                )
                actions.append(action)
            
            return actions
            
        except Exception as e:
            self.logger.error(f"Error planning actions: {e}")
            return self._fallback_action_planning(query, intent)
    
    def _fallback_action_planning(self, query: UserQuery, intent: IntentAnalysis) -> List[AgentAction]:
        """Fallback action planning based on intent type"""
        actions = []
        
        if intent.intent_type == IntentType.EMERGENCY:
            actions.append(AgentAction(
                action_type="notify_stakeholder",
                parameters={"urgency": "critical", "message": query.query},
                reasoning="Emergency requires immediate notification",
                priority="critical"
            ))
        
        elif intent.intent_type == IntentType.DECISION:
            actions.append(AgentAction(
                action_type="suggest_pivot_plan",
                parameters={"issue": query.query, "options": ["delay", "alternative", "cancel"]},
                reasoning="Decision-making requires planning alternatives",
                priority="high"
            ))
        
        elif intent.intent_type == IntentType.COORDINATION:
            actions.append(AgentAction(
                action_type="coordinate_staff",
                parameters={"task": query.query, "team": "operations"},
                reasoning="Coordination requires team involvement",
                priority="medium"
            ))
        
        else:  # LOOKUP
            actions.append(AgentAction(
                action_type="document_issue",
                parameters={"query": query.query, "category": "information_request"},
                reasoning="Information request needs documentation",
                priority="low"
            ))
        
        return actions
    
    def generate_response(self, query: UserQuery, intent: IntentAnalysis,
                         actions: List[AgentAction], retrieved_context: List[Dict[str, Any]]) -> str:
        """
        Generate final response to user
        
        Args:
            query: User query
            intent: Intent analysis
            actions: Planned actions
            retrieved_context: Retrieved context
            
        Returns:
            Formatted response string
        """
        context_text = "\n".join([ctx.get("content", "") for ctx in retrieved_context[:3]])
        actions_text = "\n".join([f"- {action.action_type}: {action.reasoning}" for action in actions])
        
        prompt = f"""
You are a helpful event operations assistant responding to on-ground personnel.

USER QUERY: "{query.query}"
INTENT: {intent.intent_type.value}

RELEVANT INFORMATION:
{context_text}

PLANNED ACTIONS:
{actions_text}

Generate a clear, actionable response that:
1. Directly addresses the user's question
2. Provides specific next steps
3. Includes relevant information from context
4. Mentions any follow-up actions
5. Maintains a professional but helpful tone

IMPORTANT:
- Be concise and specific
- Include contact information when available
- Suggest timelines for actions
- Never make financial commitments
- Always prioritize safety and compliance

Response:
"""
        
        try:
            response = self.model.generate_content(
                prompt,
                safety_settings=self.safety_settings
            )
            
            return response.text.strip()
            
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return self._fallback_response(query, intent, actions)
    
    def _fallback_response(self, query: UserQuery, intent: IntentAnalysis, actions: List[AgentAction]) -> str:
        """Fallback response generation"""
        if intent.intent_type == IntentType.EMERGENCY:
            return f"🚨 EMERGENCY: I've noted your issue '{query.query}'. Emergency protocols are being activated. Please contact your event coordinator immediately."
        
        elif intent.intent_type == IntentType.DECISION:
            return f"🤔 DECISION NEEDED: For '{query.query}', I'm analyzing options. I'll suggest alternative plans for your review. Please stand by for recommendations."
        
        elif intent.intent_type == IntentType.COORDINATION:
            return f"👥 COORDINATION: I'll help coordinate '{query.query}' with the relevant team members. You'll receive updates shortly."
        
        else:  # LOOKUP
            return f"📋 INFORMATION: I'm looking up information about '{query.query}'. I'll provide the details you need shortly."
    
    def process_query(self, query: UserQuery, retrieved_context: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Complete query processing pipeline
        
        Args:
            query: User query
            retrieved_context: Context from vector database
            
        Returns:
            Complete processing result
        """
        if retrieved_context is None:
            retrieved_context = []
        
        try:
            # Step 1: Analyze intent
            intent = self.analyze_intent(query)
            self.logger.info(f"Intent analyzed: {intent.intent_type.value} (confidence: {intent.confidence})")
            
            # Step 2: Plan actions
            actions = self.plan_actions(query, intent, retrieved_context)
            self.logger.info(f"Planned {len(actions)} actions")
            
            # Step 3: Generate response
            response = self.generate_response(query, intent, actions, retrieved_context)
            self.logger.info("Response generated")
            
            return {
                "query": query.query,
                "intent": {
                    "type": intent.intent_type.value,
                    "confidence": intent.confidence,
                    "reasoning": intent.reasoning,
                    "entities": intent.entities,
                    "keywords": intent.keywords
                },
                "actions": [
                    {
                        "type": action.action_type,
                        "parameters": action.parameters,
                        "reasoning": action.reasoning,
                        "priority": action.priority
                    }
                    for action in actions
                ],
                "response": response,
                "context_used": len(retrieved_context),
                "processing_status": "success"
            }
            
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            return {
                "query": query.query,
                "processing_status": "error",
                "error": str(e),
                "response": "I apologize, but I encountered an error processing your request. Please try again or contact your event coordinator."
            }
