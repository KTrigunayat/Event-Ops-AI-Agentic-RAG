# Event-Ops AI Agentic RAG System

## Overview
Event-Ops AI is a specialized Agentic RAG application designed to bridge communication gaps between vendors and on-ground event teams. By indexing event-brief documents into a high-performance vector store (Endee), the system provides real-time decision support, solves logistical issues, and suggests actionable plans based on ground realities.

## Architecture
The system follows a modular Agent-over-RAG pattern with autonomous orchestration based on query complexity.

```
src/
├── core/
│   └── event_ops_ai.py         # Main system orchestrator
├── parsers/
│   └── document_parser.py      # PDF/DOCX/TXT document parsing
├── chunking/
│   ├── llm_chunker.py        # LLM-based chunking
│   └── semantic_chunker.py    # Event-specific semantic chunking
├── embedding/
│   ├── distilbert_embedder.py # Fallback embeddings
│   └── advanced_embedder.py  # OpenAI/Gemini embeddings
├── database/
│   └── endee_vector_db.py    # Endee Vector Database
├── retrieval/
│   └── context_retriever.py  # Advanced context retrieval
├── agents/
│   └── agentic_orchestrator.py # Gemini 1.5 Flash agent
├── tools/
│   └── action_tools.py       # Action execution tools
└── pipeline/
    └── ingestion_pipeline.py   # Document ingestion pipeline
```

## Features

### 🎯 Core Capabilities
- **Document Ingestion**: Parse and index PDF, DOCX, TXT files
- **Semantic Chunking**: Event-specific categorization (Vendor Info, Timeline, Crisis Protocol)
- **Advanced Embeddings**: OpenAI/Gemini embeddings with fallback options
- **Agentic Reasoning**: Gemini 1.5 Flash for complex decision-making
- **Action Tools**: Vendor contact, pivot planning, staff coordination
- **Real-time Search**: Context-aware document retrieval
- **Safety Guardrails**: No financial commitments, human approval required

### 🔧 Technical Stack
- **Orchestration**: Python (FastAPI)
- **Vector Store**: Endee Vector Database (JSON-based)
- **LLM**: Gemini 1.5 Flash for reasoning
- **Embeddings**: OpenAI/Gemini with DistilBERT fallback
- **Frontend**: Streamlit for on-ground personnel
- **API**: FastAPI with async support

## Quick Start

### 1. Prerequisites
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull tinyllama
ollama serve

# Get API keys
# OpenAI: https://platform.openai.com/api-keys
# Gemini: https://makersuite.google.com/app/apikey
```

### 2. Installation
```bash
# Clone repository
git clone <repository-url>
cd Event-Ops-AI-Agentic-RAG

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

### 3. Configuration
Update `.env` with your settings:
```env
# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=tinyllama

# Database Configuration
ENDEE_DB_PATH=./endeedb

# Embedding Configuration
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=text-embedding-3-small
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=text-embedding-004

# LLM Configuration
GEMINI_FLASH_API_KEY=your_gemini_flash_api_key_here
GEMINI_FLASH_MODEL=gemini-1.5-flash
```

### 4. Run the System

#### Streamlit Frontend (Recommended for on-ground personnel)
```bash
streamlit run app.py
```
Access at: http://localhost:8501

#### FastAPI Backend (For integration)
```bash
python api.py
```
API Documentation: http://localhost:8000/docs

#### Command Line Interface
```bash
python main.py
```

## Usage Examples

### Document Ingestion
```python
from src.core.event_ops_ai import EventOpsAI

# Initialize system
config = {
    "embedding_provider": "openai",
    "gemini_flash_api_key": "your_api_key"
}
ai_system = EventOpsAI(config)

# Ingest event documents
result = ai_system.ingest_event_documents(
    document_paths=["event_brief.pdf", "vendor_contacts.docx"],
    event_id="conference_2024"
)

print(f"Ingested {result.chunks_created} chunks")
```

### Query Processing
```python
# Process operational query
result = ai_system.process_user_query(
    query="The catering truck is stuck in traffic, what's the plan?",
    user_id="staff_123",
    event_id="conference_2024",
    urgency="high"
)

print(f"AI Response: {result.response}")
print(f"Intent: {result.intent}")
print(f"Actions: {result.actions}")
```

### Document Search
```python
# Search within event documents
results = ai_system.search_event_documents(
    event_id="conference_2024",
    query="catering vendor contact",
    filters={"category": "Vendor Information", "priority": "high"}
)
```

## API Reference

### Core Endpoints

#### POST /query
Process user queries through the agentic system.
```json
{
  "query": "The catering truck is stuck in traffic, what's the plan?",
  "user_id": "staff_123",
  "event_id": "conference_2024",
  "urgency": "high"
}
```

#### POST /ingest
Ingest event documents.
```json
{
  "event_id": "conference_2024",
  "files": ["event_brief.pdf", "vendor_contacts.docx"]
}
```

#### POST /search
Search within event documents.
```json
{
  "event_id": "conference_2024",
  "query": "catering contact",
  "filters": {"category": "Vendor Information"}
}
```

#### GET /events/{event_id}/overview
Get event overview and statistics.

#### GET /system/status
Get detailed system status and health.

### Action Tools

The system includes specialized action tools:

1. **get_vendor_contact**: Retrieve vendor contact information
2. **suggest_pivot_plan**: Suggest alternative arrangements
3. **coordinate_staff**: Coordinate with event staff
4. **check_timeline**: Check for timeline conflicts
5. **notify_stakeholder**: Notify relevant stakeholders
6. **document_issue**: Document issues for follow-up

## Intent Types

The system classifies queries into four intent types:

- **LOOKUP**: Simple information retrieval (Who is vendor? What time is setup?)
- **DECISION**: Complex problem solving (How handle delay? What backup plan?)
- **COORDINATION**: Multi-party coordination (Coordinate with venue staff)
- **EMERGENCY**: Crisis management (Power outage, medical emergency)

## Data Categories

Event briefs are automatically categorized into:

- **Vendor Information**: Contact details, contracts, responsibilities
- **Timeline & Schedule**: Event timing, dependencies, deadlines
- **Crisis Protocol**: Emergency procedures, safety protocols
- **Budget & Finance**: Financial information, payment terms
- **Venue Details**: Location information, facilities
- **Staffing & Personnel**: Team roles, responsibilities
- **Equipment & Resources**: Inventory, equipment lists
- **Safety & Compliance**: Regulations, permits, security
- **Communication Plan**: Notification procedures
- **Contingency Plans**: Backup arrangements, alternatives

## Safety & Compliance

### Built-in Guardrails
- **No Financial Commitments**: Agent cannot authorize expenditures
- **Human Approval Required**: Major decisions need human confirmation
- **Safety First**: Prioritizes safety and compliance
- **Context Awareness**: Responses based on event-specific data

### Data Isolation
- Each event's data is stored separately
- No cross-contamination between events
- Unique event IDs for data separation

## Performance Considerations

### Optimization Features
- **Batch Processing**: Efficient embedding generation
- **Similarity Thresholding**: Filter irrelevant results
- **Category Boosting**: Prioritize relevant categories
- **Caching**: Cache frequently accessed data
- **Async Processing**: Non-blocking API operations

### Scalability
- **Modular Architecture**: Easy to extend components
- **Plugin System**: Add new action tools
- **Multiple Embedding Providers**: Switch between OpenAI/Gemini
- **Database Agnostic**: Easy to switch vector stores

## Troubleshooting

### Common Issues

1. **Ollama Connection Failed**
   - Ensure Ollama server is running: `ollama serve`
   - Check model availability: `ollama list`

2. **API Key Errors**
   - Verify API keys in `.env` file
   - Check API key permissions and quotas

3. **Document Parsing Errors**
   - Ensure files are not corrupted
   - Check file format support (PDF, DOCX, TXT)

4. **Memory Issues**
   - Reduce batch size for embeddings
   - Use smaller documents for testing

5. **Vector Database Issues**
   - Check file permissions for database path
   - Ensure sufficient disk space

### Logging
Check logs for detailed error information:
- Application logs: Console output
- Error logs: Check exception messages
- Performance logs: Processing times and metrics

## Development

### Project Structure
```
Event-Ops-AI-Agentic-RAG/
├── src/                    # Source code
├── app.py                  # Streamlit frontend
├── api.py                  # FastAPI backend
├── main.py                 # CLI interface
├── requirements.txt        # Dependencies
├── .env.example           # Configuration template
└── README.md              # This file
```

### Adding New Features

1. **New Action Tools**: Add to `src/tools/action_tools.py`
2. **New Embedding Providers**: Extend `src/embedding/advanced_embedder.py`
3. **New Document Parsers**: Add to `src/parsers/document_parser.py`
4. **New Intent Types**: Update `src/agents/agentic_orchestrator.py`

### Testing
```bash
# Run basic functionality test
python main.py

# Test API endpoints
curl http://localhost:8000/health

# Test Streamlit app
streamlit run app.py
```

## Future Enhancements

### Planned Features
- **Multi-Agent Systems**: Specialized agents for different domains
- **Real-time Telemetry**: IoT integration for automatic alerts
- **Mobile App**: Native mobile application
- **Advanced Analytics**: Event performance insights
- **Integration APIs**: Connect with existing event management systems

### Scalability Roadmap
- **Distributed Processing**: Handle multiple events simultaneously
- **Cloud Deployment**: Scalable cloud infrastructure
- **Advanced Caching**: Redis integration for performance
- **Load Balancing**: Handle high query volumes

## Compliance Checklist

✅ **Completed Requirements**
- [x] Starred official Endee repository
- [x] Forked repository used as base
- [x] RAG system implementation for event briefs
- [x] Agentic logic for issue solving and decision making
- [x] Document parsing for PDF/DOCX files
- [x] Semantic chunking for event-specific categories
- [x] OpenAI/Gemini embeddings with fallback
- [x] Gemini 1.5 Flash agentic orchestrator
- [x] Intent analysis (Lookup vs Decision tasks)
- [x] Context retrieval from Endee Vector DB
- [x] Reasoning and planning engine
- [x] Action tools implementation
- [x] Response generation system
- [x] Streamlit frontend for on-ground personnel
- [x] Data isolation for unique events
- [x] Safety guardrails (no financial commitments)
- [x] FastAPI orchestration layer

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the API documentation
3. Examine the code comments
4. Check the logs for detailed error information

---

**Event-Ops AI**: Empowering event teams with intelligent decision support.