import os
import logging
from typing import List, Dict, Any, Optional
import requests
import json
from dataclasses import dataclass

from ..chunking.llm_chunker import Chunk

@dataclass
class SemanticChunk:
    """Represents a semantic chunk with event-specific categorization"""
    content: str
    chunk_id: str
    source: str
    category: str  # "Vendor Info", "Timeline", "Crisis Protocol", etc.
    priority: str  # "high", "medium", "low"
    start_index: int
    end_index: int
    metadata: Dict[str, Any]

class EventBriefChunker:
    """Specialized chunker for event briefs with semantic categorization"""
    
    def __init__(self, base_url: str = None, model: str = "tinyllama"):
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = model
        self.logger = logging.getLogger(__name__)
        
        # Event-specific categories
        self.categories = [
            "Vendor Information",
            "Timeline & Schedule", 
            "Crisis Protocol",
            "Budget & Finance",
            "Venue Details",
            "Staffing & Personnel",
            "Equipment & Resources",
            "Safety & Compliance",
            "Communication Plan",
            "Contingency Plans"
        ]
    
    def _generate_semantic_chunking_prompt(self, text: str) -> str:
        """Generate prompt for semantic chunking of event briefs"""
        categories_str = "\n".join([f"- {cat}" for cat in self.categories])
        
        return f"""
You are an expert event coordinator analyzing an event brief. Please break the following text into meaningful semantic chunks.

CATEGORIES to identify:
{categories_str}

For each chunk, provide:
1. Content: The actual text chunk
2. Category: Which of the above categories it belongs to
3. Priority: "high", "medium", or "low" based on operational importance
4. Reasoning: Brief explanation of why this chunk was created this way

GUIDELINES:
- Each chunk should be self-contained and actionable
- Vendor info should include contact details and responsibilities
- Timeline chunks should include specific times and dependencies
- Crisis protocols should be complete with step-by-step instructions
- Keep chunks under 800 characters when possible
- Overlap important information between related chunks

Event Brief Text:
{text}

Return the chunks as a JSON array with this structure:
[
    {{
        "content": "chunk content here",
        "category": "Vendor Information",
        "priority": "high",
        "reasoning": "Contains critical vendor contact information"
    }}
]
"""
    
    def _call_ollama(self, prompt: str) -> str:
        """Make API call to Ollama"""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=60
            )
            response.raise_for_status()
            return response.json().get("response", "")
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error calling Ollama API: {e}")
            raise
    
    def _parse_semantic_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse LLM response to extract semantic chunks"""
        try:
            # Try to extract JSON from response
            start_idx = response.find('[')
            end_idx = response.rfind(']') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx:end_idx]
                chunks_data = json.loads(json_str)
                return chunks_data
            else:
                # Fallback to simple chunking
                return self._fallback_semantic_chunking(response)
                
        except json.JSONDecodeError as e:
            self.logger.warning(f"Failed to parse JSON from LLM response: {e}")
            return self._fallback_semantic_chunking(response)
    
    def _fallback_semantic_chunking(self, text: str) -> List[Dict[str, Any]]:
        """Fallback semantic chunking using keyword patterns"""
        chunks = []
        
        # Define keyword patterns for each category
        category_keywords = {
            "Vendor Information": ["vendor", "supplier", "contractor", "contact", "phone", "email"],
            "Timeline & Schedule": ["timeline", "schedule", "time", "start", "end", "deadline", "duration"],
            "Crisis Protocol": ["crisis", "emergency", "protocol", "contingency", "backup", "plan"],
            "Budget & Finance": ["budget", "cost", "payment", "invoice", "finance", "expense"],
            "Venue Details": ["venue", "location", "address", "facility", "room", "area"],
            "Staffing & Personnel": ["staff", "personnel", "team", "role", "responsibility", "duty"],
            "Equipment & Resources": ["equipment", "resource", "material", "tool", "supply", "inventory"],
            "Safety & Compliance": ["safety", "compliance", "regulation", "permit", "license", "security"],
            "Communication Plan": ["communication", "notify", "inform", "update", "report", "coordinate"],
            "Contingency Plans": ["contingency", "backup", "alternative", "fallback", "plan b"]
        }
        
        # Split text into paragraphs
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        
        for paragraph in paragraphs:
            # Determine category based on keywords
            category = "General Information"
            priority = "medium"
            
            for cat, keywords in category_keywords.items():
                if any(keyword.lower() in paragraph.lower() for keyword in keywords):
                    category = cat
                    # Set priority based on category
                    if cat in ["Crisis Protocol", "Vendor Information", "Timeline & Schedule"]:
                        priority = "high"
                    elif cat in ["Safety & Compliance", "Budget & Finance", "Contingency Plans"]:
                        priority = "medium"
                    else:
                        priority = "low"
                    break
            
            chunks.append({
                "content": paragraph,
                "category": category,
                "priority": priority,
                "reasoning": f"Keyword-based categorization: {category}"
            })
        
        return chunks
    
    def chunk_event_brief(self, text: str, source: str = "event_brief") -> List[SemanticChunk]:
        """
        Chunk event brief text with semantic categorization
        
        Args:
            text: Event brief text to chunk
            source: Source identifier
            
        Returns:
            List of SemanticChunk objects
        """
        if not text or not text.strip():
            return []
        
        self.logger.info(f"Semantic chunking of event brief from source: {source}")
        
        # Generate semantic chunking prompt
        prompt = self._generate_semantic_chunking_prompt(text)
        
        try:
            # Call LLM for semantic analysis
            llm_response = self._call_ollama(prompt)
            chunks_data = self._parse_semantic_response(llm_response)
            
            # Convert to SemanticChunk objects
            semantic_chunks = []
            current_start = 0
            
            for i, chunk_data in enumerate(chunks_data):
                content = chunk_data.get("content", "").strip()
                if not content:
                    continue
                
                # Find the actual position in original text
                try:
                    start_idx = text.find(content, current_start)
                    if start_idx == -1:
                        start_idx = current_start
                    end_idx = start_idx + len(content)
                except:
                    start_idx = current_start
                    end_idx = start_idx + len(content)
                
                semantic_chunk = SemanticChunk(
                    content=content,
                    chunk_id=f"{source}_semantic_{i}",
                    source=source,
                    category=chunk_data.get("category", "General Information"),
                    priority=chunk_data.get("priority", "medium"),
                    start_index=start_idx,
                    end_index=end_idx,
                    metadata={
                        "chunk_method": "semantic_llm_based",
                        "reasoning": chunk_data.get("reasoning", "Semantic chunking"),
                        "model_used": self.model,
                        "chunk_index": i,
                        "event_brief_type": "operational"
                    }
                )
                semantic_chunks.append(semantic_chunk)
                current_start = end_idx
            
            self.logger.info(f"Created {len(semantic_chunks)} semantic chunks")
            return semantic_chunks
            
        except Exception as e:
            self.logger.error(f"Error in semantic chunking: {e}")
            # Fallback to simple semantic chunking
            return self._fallback_semantic_chunking_with_conversion(text, source)
    
    def _fallback_semantic_chunking_with_conversion(self, text: str, source: str) -> List[SemanticChunk]:
        """Convert fallback chunks to SemanticChunk objects"""
        fallback_chunks = self._fallback_semantic_chunking(text)
        semantic_chunks = []
        
        for i, chunk_data in enumerate(fallback_chunks):
            semantic_chunk = SemanticChunk(
                content=chunk_data["content"],
                chunk_id=f"{source}_fallback_{i}",
                source=source,
                category=chunk_data["category"],
                priority=chunk_data["priority"],
                start_index=i * 1000,  # Approximate positions
                end_index=(i + 1) * 1000,
                metadata={
                    "chunk_method": "fallback_semantic",
                    "reasoning": chunk_data["reasoning"],
                    "chunk_index": i
                }
            )
            semantic_chunks.append(semantic_chunk)
        
        return semantic_chunks
    
    def get_chunks_by_category(self, chunks: List[SemanticChunk], category: str) -> List[SemanticChunk]:
        """Filter chunks by category"""
        return [chunk for chunk in chunks if chunk.category == category]
    
    def get_high_priority_chunks(self, chunks: List[SemanticChunk]) -> List[SemanticChunk]:
        """Get high priority chunks"""
        return [chunk for chunk in chunks if chunk.priority == "high"]
    
    def get_categories_summary(self, chunks: List[SemanticChunk]) -> Dict[str, int]:
        """Get summary of chunks by category"""
        category_count = {}
        for chunk in chunks:
            category_count[chunk.category] = category_count.get(chunk.category, 0) + 1
        return category_count
