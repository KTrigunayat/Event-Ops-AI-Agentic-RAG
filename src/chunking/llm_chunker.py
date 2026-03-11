import os
import logging
from typing import List, Dict, Any, Optional
import requests
import json
from dataclasses import dataclass

@dataclass
class Chunk:
    """Represents a text chunk with metadata"""
    content: str
    chunk_id: str
    source: str
    start_index: int
    end_index: int
    metadata: Dict[str, Any]

class OllamaChunker:
    """LLM-based chunking using Ollama TinyLlama model"""
    
    def __init__(self, base_url: str = None, model: str = "tinyllama"):
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = model
        self.logger = logging.getLogger(__name__)
        
    def _generate_chunking_prompt(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> str:
        """Generate prompt for LLM-based chunking"""
        return f"""
Please break the following text into meaningful chunks of approximately {chunk_size} characters.
Each chunk should:
- Be semantically coherent and complete
- Maintain context and readability
- Overlap with previous/next chunks by ~{overlap} characters when possible
- Preserve important information and relationships

Text to chunk:
{text}

Return the chunks as a JSON array with the following structure:
[
    {{
        "content": "chunk content here",
        "reasoning": "brief explanation of why this chunk was created this way"
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
                timeout=30
            )
            response.raise_for_status()
            return response.json().get("response", "")
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error calling Ollama API: {e}")
            raise
    
    def _parse_llm_response(self, response: str) -> List[Dict[str, str]]:
        """Parse LLM response to extract chunks"""
        try:
            # Try to extract JSON from response
            start_idx = response.find('[')
            end_idx = response.rfind(']') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx:end_idx]
                chunks_data = json.loads(json_str)
                return chunks_data
            else:
                # Fallback: split text manually
                return self._fallback_chunking(response)
                
        except json.JSONDecodeError as e:
            self.logger.warning(f"Failed to parse JSON from LLM response: {e}")
            return self._fallback_chunking(response)
    
    def _fallback_chunking(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, str]]:
        """Fallback chunking method if LLM fails"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundaries
            if end < len(text):
                # Look for sentence endings
                for i in range(end, max(start + chunk_size - 200, start), -1):
                    if text[i] in '.!?' and i + 1 < len(text) and text[i + 1] == ' ':
                        end = i + 1
                        break
            
            chunk_content = text[start:end].strip()
            if chunk_content:
                chunks.append({
                    "content": chunk_content,
                    "reasoning": "Fallback chunking - sentence boundary based"
                })
            
            start = end - overlap if end - overlap > start else end
        
        return chunks
    
    def chunk_text(self, text: str, source: str = "unknown", chunk_size: int = 1000, overlap: int = 200) -> List[Chunk]:
        """
        Chunk text using LLM-based approach
        
        Args:
            text: Text to chunk
            source: Source identifier for the text
            chunk_size: Target chunk size in characters
            overlap: Overlap between chunks in characters
            
        Returns:
            List of Chunk objects
        """
        if not text or not text.strip():
            return []
        
        self.logger.info(f"Chunking text of length {len(text)} from source: {source}")
        
        # Generate chunking prompt
        prompt = self._generate_chunking_prompt(text, chunk_size, overlap)
        
        # Call LLM
        try:
            llm_response = self._call_ollama(prompt)
            chunks_data = self._parse_llm_response(llm_response)
            
            # Convert to Chunk objects
            chunks = []
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
                
                chunk = Chunk(
                    content=content,
                    chunk_id=f"{source}_chunk_{i}",
                    source=source,
                    start_index=start_idx,
                    end_index=end_idx,
                    metadata={
                        "chunk_method": "llm_based",
                        "reasoning": chunk_data.get("reasoning", "LLM-based chunking"),
                        "model_used": self.model,
                        "chunk_index": i
                    }
                )
                chunks.append(chunk)
                current_start = end_idx - overlap
            
            self.logger.info(f"Successfully created {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            self.logger.error(f"Error in LLM chunking: {e}")
            # Fallback to simple chunking
            return self._simple_chunking(text, source, chunk_size, overlap)
    
    def _simple_chunking(self, text: str, source: str, chunk_size: int, overlap: int) -> List[Chunk]:
        """Simple fallback chunking method"""
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = start + chunk_size
            if end > len(text):
                end = len(text)
            
            content = text[start:end].strip()
            if content:
                chunk = Chunk(
                    content=content,
                    chunk_id=f"{source}_chunk_{chunk_index}",
                    source=source,
                    start_index=start,
                    end_index=end,
                    metadata={
                        "chunk_method": "simple_fallback",
                        "chunk_index": chunk_index
                    }
                )
                chunks.append(chunk)
                chunk_index += 1
            
            start = end - overlap if end - overlap > start else end
        
        return chunks
