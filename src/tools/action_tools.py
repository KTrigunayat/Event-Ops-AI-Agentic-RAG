import os
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import json

from ..database.endee_vector_db import EndeeVectorDatabase

@dataclass
class ToolResult:
    """Result from executing an action tool"""
    success: bool
    data: Any
    message: str
    tool_name: str
    execution_time: float

class ActionTools:
    """Collection of action tools for the agentic orchestrator"""
    
    def __init__(self, database: EndeeVectorDatabase):
        self.database = database
        self.logger = logging.getLogger(__name__)
        
        # Tool registry
        self.tools = {
            "get_vendor_contact": self.get_vendor_contact,
            "suggest_pivot_plan": self.suggest_pivot_plan,
            "coordinate_staff": self.coordinate_staff,
            "check_timeline": self.check_timeline,
            "notify_stakeholder": self.notify_stakeholder,
            "document_issue": self.document_issue
        }
    
    def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> ToolResult:
        """
        Execute a specific action tool
        
        Args:
            tool_name: Name of the tool to execute
            parameters: Parameters for the tool
            
        Returns:
            ToolResult with execution outcome
        """
        start_time = datetime.now()
        
        if tool_name not in self.tools:
            return ToolResult(
                success=False,
                data=None,
                message=f"Unknown tool: {tool_name}",
                tool_name=tool_name,
                execution_time=0.0
            )
        
        try:
            self.logger.info(f"Executing tool: {tool_name} with parameters: {parameters}")
            
            # Execute the tool
            result_data = self.tools[tool_name](parameters)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return ToolResult(
                success=True,
                data=result_data,
                message=f"Tool {tool_name} executed successfully",
                tool_name=tool_name,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Error executing tool {tool_name}: {e}")
            
            return ToolResult(
                success=False,
                data=None,
                message=f"Error executing {tool_name}: {str(e)}",
                tool_name=tool_name,
                execution_time=execution_time
            )
    
    def get_vendor_contact(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get contact information for a specific vendor
        
        Args:
            parameters: {"vendor": "vendor_name", "reason": "contact_reason"}
            
        Returns:
            Vendor contact information
        """
        vendor_name = parameters.get("vendor", "").lower()
        reason = parameters.get("reason", "general inquiry")
        
        if not vendor_name:
            raise ValueError("Vendor name is required")
        
        # Search for vendor information in database
        query_text = f"vendor contact {vendor_name} phone email"
        search_results = self.database.search_similar_chunks(
            query_text, limit=5, similarity_threshold=0.3
        )
        
        # Filter for vendor information chunks
        vendor_chunks = [
            chunk for chunk in search_results 
            if "vendor" in chunk.get("metadata", {}).get("category", "").lower() or
               "vendor" in chunk.get("content", "").lower()
        ]
        
        if not vendor_chunks:
            return {
                "vendor": vendor_name,
                "contact_found": False,
                "message": f"No contact information found for vendor: {vendor_name}",
                "suggestion": "Please check the event brief or contact event coordinator"
            }
        
        # Extract contact information
        contact_info = {}
        for chunk in vendor_chunks:
            content = chunk.get("content", "").lower()
            
            # Look for phone numbers
            import re
            phones = re.findall(r'(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})', content)
            if phones and "phone" not in contact_info:
                contact_info["phone"] = phones[0][0] + phones[0][1] + "-" + phones[0][2] + "-" + phones[0][3]
            
            # Look for emails
            emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', content)
            if emails and "email" not in contact_info:
                contact_info["email"] = emails[0]
            
            # Look for contact person names
            if "contact" in content or "manager" in content:
                lines = content.split('\n')
                for line in lines:
                    if any(word in line.lower() for word in ["contact", "manager", "representative"]):
                        contact_info["contact_person"] = line.strip()
                        break
        
        return {
            "vendor": vendor_name,
            "contact_found": bool(contact_info),
            "contact_info": contact_info,
            "reason": reason,
            "source_chunks": len(vendor_chunks),
            "recommended_action": f"Contact {vendor_name} regarding: {reason}"
        }
    
    def suggest_pivot_plan(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Suggest alternative plans for handling issues
        
        Args:
            parameters: {"issue": "description", "options": ["option1", "option2"]}
            
        Returns:
            Pivot plan suggestions
        """
        issue = parameters.get("issue", "")
        options = parameters.get("options", [])
        
        if not issue:
            raise ValueError("Issue description is required")
        
        # Search for relevant contingency plans
        query_text = f"contingency backup plan {issue}"
        search_results = self.database.search_similar_chunks(
            query_text, limit=5, similarity_threshold=0.3
        )
        
        # Filter for contingency and crisis protocol chunks
        contingency_chunks = [
            chunk for chunk in search_results
            if any(keyword in chunk.get("metadata", {}).get("category", "").lower()
                   for keyword in ["contingency", "crisis", "protocol", "backup"])
        ]
        
        # Generate pivot suggestions
        suggestions = []
        
        # Add suggestions from database
        for chunk in contingency_chunks:
            suggestions.append({
                "source": "event_brief",
                "suggestion": chunk.get("content", "")[:200] + "...",
                "confidence": chunk.get("similarity", 0.0),
                "category": chunk.get("metadata", {}).get("category", "General")
            })
        
        # Add generic suggestions based on issue type
        issue_lower = issue.lower()
        if "delay" in issue_lower or "late" in issue_lower:
            suggestions.append({
                "source": "generic",
                "suggestion": "Adjust timeline: Move non-critical items back, buffer critical path items",
                "confidence": 0.7,
                "category": "Timeline Management"
            })
        
        if "vendor" in issue_lower:
            suggestions.append({
                "source": "generic", 
                "suggestion": "Activate backup vendor: Check event brief for alternative suppliers",
                "confidence": 0.6,
                "category": "Vendor Management"
            })
        
        if "equipment" in issue_lower or "resource" in issue_lower:
            suggestions.append({
                "source": "generic",
                "suggestion": "Resource reallocation: Check inventory, prioritize critical equipment",
                "confidence": 0.6,
                "category": "Resource Management"
            })
        
        return {
            "issue": issue,
            "pivot_suggestions": suggestions,
            "recommended_options": options[:3],  # Top 3 options
            "contingency_plans_found": len(contingency_chunks),
            "next_steps": [
                "Review suggested options with event coordinator",
                "Assess impact on timeline and budget",
                "Communicate changes to affected parties",
                "Update event documentation"
            ]
        }
    
    def coordinate_staff(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Coordinate with event staff
        
        Args:
            parameters: {"task": "task_description", "team": "team_name"}
            
        Returns:
            Staff coordination plan
        """
        task = parameters.get("task", "")
        team = parameters.get("team", "operations")
        
        if not task:
            raise ValueError("Task description is required")
        
        # Search for staffing information
        query_text = f"staff personnel {team} responsibilities"
        search_results = self.database.search_similar_chunks(
            query_text, limit=5, similarity_threshold=0.3
        )
        
        # Filter for staffing chunks
        staffing_chunks = [
            chunk for chunk in search_results
            if "staff" in chunk.get("metadata", {}).get("category", "").lower() or
               "personnel" in chunk.get("content", "").lower()
        ]
        
        # Extract staff information
        staff_info = {
            "team_members": [],
            "responsibilities": [],
            "contact_info": {}
        }
        
        for chunk in staffing_chunks:
            content = chunk.get("content", "")
            lines = content.split('\n')
            
            for line in lines:
                if any(keyword in line.lower() for keyword in ["responsible", "role", "duty"]):
                    staff_info["responsibilities"].append(line.strip())
                
                # Look for names and roles
                if ":" in line and len(line.split(":")) == 2:
                    role, person = line.split(":", 1)
                    staff_info["team_members"].append({
                        "role": role.strip(),
                        "person": person.strip()
                    })
        
        return {
            "task": task,
            "team": team,
            "coordination_plan": {
                "assigned_team": team,
                "task_description": task,
                "available_staff": staff_info["team_members"],
                "team_responsibilities": staff_info["responsibilities"],
                "coordination_steps": [
                    f"Notify {team} team of task: {task}",
                    "Assign responsible team member",
                    "Set deadline and checkpoints",
                    "Document assignment in event log"
                ]
            },
            "staffing_info_available": len(staffing_chunks) > 0
        }
    
    def check_timeline(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check event timeline for conflicts or issues
        
        Args:
            parameters: {"time": "specific_time", "activity": "activity_name"}
            
        Returns:
            Timeline analysis
        """
        time = parameters.get("time", "")
        activity = parameters.get("activity", "")
        
        # Search for timeline information
        query_text = f"timeline schedule {time} {activity}"
        search_results = self.database.search_similar_chunks(
            query_text, limit=5, similarity_threshold=0.3
        )
        
        # Filter for timeline chunks
        timeline_chunks = [
            chunk for chunk in search_results
            if "timeline" in chunk.get("metadata", {}).get("category", "").lower() or
               "schedule" in chunk.get("content", "").lower()
        ]
        
        # Analyze timeline conflicts
        conflicts = []
        suggestions = []
        
        for chunk in timeline_chunks:
            content = chunk.get("content", "").lower()
            
            # Look for time conflicts
            if time and time in content:
                if "conflict" in content or "overlap" in content:
                    conflicts.append({
                        "time": time,
                        "conflict_description": chunk.get("content", "")[:150],
                        "severity": "high" if "critical" in content else "medium"
                    })
        
        # Generate suggestions
        if conflicts:
            suggestions.append("Review timeline for conflicts and adjust accordingly")
        else:
            suggestions.append("Timeline appears clear for requested time")
        
        return {
            "time": time,
            "activity": activity,
            "timeline_analysis": {
                "conflicts_found": len(conflicts),
                "conflicts": conflicts,
                "timeline_chunks_found": len(timeline_chunks),
                "suggestions": suggestions,
                "recommendation": "Proceed with activity" if not conflicts else "Resolve conflicts before proceeding"
            }
        }
    
    def notify_stakeholder(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Notify relevant stakeholders
        
        Args:
            parameters: {"urgency": "urgency_level", "message": "notification_message"}
            
        Returns:
            Notification plan
        """
        urgency = parameters.get("urgency", "medium")
        message = parameters.get("message", "")
        
        if not message:
            raise ValueError("Message is required")
        
        # Search for communication plan
        query_text = "communication plan notify stakeholders"
        search_results = self.database.search_similar_chunks(
            query_text, limit=3, similarity_threshold=0.3
        )
        
        # Determine notification channels based on urgency
        notification_channels = []
        if urgency == "critical":
            notification_channels = ["phone_call", "sms", "email", "in_person"]
        elif urgency == "high":
            notification_channels = ["phone_call", "email", "sms"]
        elif urgency == "medium":
            notification_channels = ["email", "team_chat"]
        else:
            notification_channels = ["email", "documentation"]
        
        return {
            "urgency": urgency,
            "message": message,
            "notification_plan": {
                "channels": notification_channels,
                "stakeholders": ["event_coordinator", "operations_team", "vendor_contacts"],
                "timeline": "immediate" if urgency in ["critical", "high"] else "within_1_hour",
                "follow_up_required": urgency in ["critical", "high"],
                "documentation": True
            },
            "communication_protocols": len(search_results) > 0
        }
    
    def document_issue(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Document issue for follow-up
        
        Args:
            parameters: {"query": "original_query", "category": "issue_category"}
            
        Returns:
            Documentation record
        """
        query = parameters.get("query", "")
        category = parameters.get("category", "general")
        
        # Create issue record
        issue_record = {
            "issue_id": f"issue_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "category": category,
            "status": "documented",
            "follow_up_required": True,
            "priority": "medium"
        }
        
        # Save to a simple log file (in production, this would be a proper database)
        log_file = "issue_log.json"
        try:
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    issues = json.load(f)
            else:
                issues = []
            
            issues.append(issue_record)
            
            with open(log_file, 'w') as f:
                json.dump(issues, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving issue log: {e}")
        
        return {
            "issue_recorded": True,
            "issue_id": issue_record["issue_id"],
            "category": category,
            "status": "documented_for_follow_up",
            "next_steps": [
                "Issue documented in system",
                "Event coordinator will review",
                "Follow-up actions will be assigned",
                "Resolution will be tracked"
            ]
        }
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tools"""
        return list(self.tools.keys())
    
    def get_tool_description(self, tool_name: str) -> str:
        """Get description of a specific tool"""
        descriptions = {
            "get_vendor_contact": "Retrieve contact information for vendors",
            "suggest_pivot_plan": "Suggest alternative plans and contingencies",
            "coordinate_staff": "Coordinate tasks with event staff",
            "check_timeline": "Check event timeline for conflicts",
            "notify_stakeholder": "Notify relevant stakeholders",
            "document_issue": "Document issues for follow-up"
        }
        return descriptions.get(tool_name, "No description available")
