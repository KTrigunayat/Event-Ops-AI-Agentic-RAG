#!/usr/bin/env python3
"""
Endee Ingestion Pipeline - Step A
LLM-based chunking with Ollama TinyLlama, DistilBERT embeddings, and vector storage
"""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.pipeline.ingestion_pipeline import IngestionPipeline

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('ingestion.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def load_config():
    """Load configuration from environment variables"""
    load_dotenv()
    
    return {
        "ollama_base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        "ollama_model": os.getenv("OLLAMA_MODEL", "tinyllama"),
        "endeedb_path": os.getenv("ENDEE_DB_PATH", "./endeedb"),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "distilbert-base-uncased"),
        "chunk_size": int(os.getenv("CHUNK_SIZE", 1000)),
        "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", 200))
    }

def main():
    """Main function to run the ingestion pipeline"""
    print("🚀 Starting Endee Ingestion Pipeline - Step A")
    print("=" * 50)
    
    # Setup
    setup_logging()
    config = load_config()
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize pipeline
        logger.info("Initializing ingestion pipeline...")
        pipeline = IngestionPipeline(config)
        
        # Get pipeline stats
        stats = pipeline.get_pipeline_stats()
        print(f"✅ Pipeline initialized successfully!")
        print(f"   - Chunker model: {stats['chunker_model']}")
        print(f"   - Embedding model: {stats['embedding_model']}")
        print(f"   - Embedding dimension: {stats['embedding_dimension']}")
        print()
        
        # Example usage
        print("📝 Example Usage:")
        print("-" * 30)
        
        # Sample text for demonstration
        sample_text = """
        Artificial intelligence (AI) is intelligence demonstrated by machines, 
        in contrast to the natural intelligence displayed by humans and animals. 
        Leading AI textbooks define the field as the study of "intelligent agents": 
        any device that perceives its environment and takes actions that maximize 
        its chance of successfully achieving its goals. The term "artificial 
        intelligence" had previously been used to describe machines that mimic 
        cognitive functions that humans associate with the human mind, such as 
        "learning" and "problem solving".
        
        AI applications include advanced web search engines, recommendation systems 
        (used by YouTube, Amazon and Netflix), understanding human speech (such as 
        Siri and Alexa), self-driving cars (e.g., Tesla), and competing at the 
        highest level in strategic games (such as chess and Go).
        """
        
        # Process sample text
        print("Processing sample text...")
        result = pipeline.process_text(
            text=sample_text,
            source="sample_ai_text",
            chunk_size=config["chunk_size"],
            overlap=config["chunk_overlap"]
        )
        
        if result["status"] == "success":
            print(f"✅ Successfully processed text!")
            print(f"   - Chunks created: {result['chunks_created']}")
            print(f"   - Chunks stored: {result['chunks_stored']}")
            print(f"   - Embedding dimension: {result['embedding_dimension']}")
        else:
            print(f"❌ Error processing text: {result['error']}")
        
        print()
        
        # Example search
        print("🔍 Example Search:")
        print("-" * 30)
        search_results = pipeline.search_similar_content("machine learning", limit=3)
        
        if search_results:
            print(f"Found {len(search_results)} similar chunks:")
            for i, result in enumerate(search_results, 1):
                print(f"  {i}. Similarity: {result['similarity']:.4f}")
                print(f"     Content: {result['content'][:100]}...")
                print()
        else:
            print("No similar content found.")
        
        print()
        print("🎉 Ingestion Pipeline Step A completed successfully!")
        print("=" * 50)
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
