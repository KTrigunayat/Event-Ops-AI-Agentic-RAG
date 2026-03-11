import streamlit as st
import os
import sys
from pathlib import Path
import json
from datetime import datetime
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.core.event_ops_ai import EventOpsAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state
if 'event_ops_ai' not in st.session_state:
    st.session_state.event_ops_ai = None
if 'current_event_id' not in st.session_state:
    st.session_state.current_event_id = None
if 'query_history' not in st.session_state:
    st.session_state.query_history = []

def initialize_system():
    """Initialize the Event-Ops AI system"""
    try:
        config = {
            "ollama_base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            "ollama_model": os.getenv("OLLAMA_MODEL", "tinyllama"),
            "endeedb_path": os.getenv("ENDEE_DB_PATH", "./endeedb"),
            "embedding_provider": os.getenv("EMBEDDING_PROVIDER", "openai"),
            "gemini_flash_api_key": os.getenv("GEMINI_FLASH_API_KEY"),
            "gemini_flash_model": os.getenv("GEMINI_FLASH_MODEL", "gemini-1.5-flash")
        }
        
        st.session_state.event_ops_ai = EventOpsAI(config)
        return True
    except Exception as e:
        st.error(f"Failed to initialize system: {e}")
        return False

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Event-Ops AI",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🎯 Event-Ops AI Agentic RAG")
    st.markdown("Real-time decision support for event operations teams")
    
    # Initialize system
    if st.session_state.event_ops_ai is None:
        if st.button("🚀 Initialize System", type="primary"):
            with st.spinner("Initializing Event-Ops AI System..."):
                if initialize_system():
                    st.success("✅ System initialized successfully!")
                    st.rerun()
        else:
            st.info("Click 'Initialize System' to start using Event-Ops AI")
            return
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Control Panel")
        
        # System Status
        st.subheader("System Status")
        try:
            status = st.session_state.event_ops_ai.get_system_status()
            if status["system_status"] == "operational":
                st.success("🟢 System Operational")
            else:
                st.error("🔴 System Error")
        except Exception as e:
            st.error(f"❌ Status Check Failed: {e}")
        
        # Event Selection
        st.subheader("Event Management")
        
        # Create new event
        if st.button("📄 Create New Event"):
            st.session_state.show_event_creation = True
        
        # Event selection (would be populated from database)
        event_id = st.text_input("Event ID:", value=st.session_state.current_event_id or "")
        if event_id:
            st.session_state.current_event_id = event_id
        
        # Quick Stats
        if st.session_state.current_event_id:
            st.subheader("Event Stats")
            try:
                overview = st.session_state.event_ops_ai.get_event_overview(st.session_state.current_event_id)
                if overview["status"] == "found":
                    st.metric("Total Chunks", overview["total_chunks"])
                    st.metric("Categories", len(overview["categories"]))
                else:
                    st.warning("No data found for this event")
            except Exception as e:
                st.error(f"Error loading event stats: {e}")
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Query Assistant", "📄 Document Ingestion", "📊 Event Analytics", "🔍 Search"])
    
    with tab1:
        st.header("💬 Query Assistant")
        st.markdown("Ask questions about your event operations")
        
        if not st.session_state.current_event_id:
            st.warning("⚠️ Please select or create an event first")
        else:
            # Query input
            col1, col2 = st.columns([4, 1])
            
            with col1:
                user_query = st.text_input(
                    "Enter your question:",
                    placeholder="e.g., The catering truck is stuck in traffic, what's the plan?",
                    key="query_input"
                )
            
            with col2:
                urgency = st.selectbox(
                    "Urgency:",
                    ["low", "medium", "high", "critical"],
                    index=1,
                    key="urgency_select"
                )
            
            # Submit button
            if st.button("🚀 Submit Query", type="primary") and user_query:
                with st.spinner("Processing your query..."):
                    try:
                        result = st.session_state.event_ops_ai.process_user_query(
                            query=user_query,
                            user_id="streamlit_user",
                            event_id=st.session_state.current_event_id,
                            urgency=urgency
                        )
                        
                        # Store in history
                        st.session_state.query_history.append({
                            "timestamp": datetime.now().isoformat(),
                            "query": user_query,
                            "urgency": urgency,
                            "result": result
                        })
                        
                        # Display response
                        st.success("✅ Query processed successfully!")
                        
                        # Response
                        st.subheader("🤖 AI Response")
                        st.write(result.response)
                        
                        # Intent Analysis
                        if result.intent:
                            st.subheader("🧠 Intent Analysis")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Intent Type", result.intent.get("type", "Unknown"))
                            with col2:
                                st.metric("Confidence", f"{result.intent.get('confidence', 0):.2f}")
                            with col3:
                                st.metric("Context Used", result.context_used)
                        
                        # Actions Taken
                        if result.actions:
                            st.subheader("⚡ Actions Executed")
                            for i, action in enumerate(result.actions):
                                with st.expander(f"Action {i+1}: {action['action']['type']}"):
                                    st.write(f"**Reasoning:** {action['action']['reasoning']}")
                                    st.write(f"**Parameters:** {action['action']['parameters']}")
                                    st.write(f"**Priority:** {action['action']['priority']}")
                                    
                                    if action['execution_result']['success']:
                                        st.success(f"✅ {action['execution_result']['message']}")
                                    else:
                                        st.error(f"❌ {action['execution_result']['message']}")
                        
                        # Processing info
                        st.caption(f"Processed in {result.processing_time:.2f} seconds | Query ID: {result.query_id}")
                        
                    except Exception as e:
                        st.error(f"❌ Error processing query: {e}")
            
            # Query History
            if st.session_state.query_history:
                st.subheader("📜 Query History")
                for i, item in enumerate(reversed(st.session_state.query_history[-5:])):
                    with st.expander(f"Q: {item['query'][:50]}... ({item['urgency']})"):
                        st.write(f"**Time:** {item['timestamp']}")
                        st.write(f"**Response:** {item['result'].response}")
    
    with tab2:
        st.header("📄 Document Ingestion")
        st.markdown("Upload and process event documents")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Upload event documents (PDF, DOCX, TXT):",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            st.subheader("📋 Document Preview")
            for file in uploaded_files:
                st.write(f"📄 {file.name} ({file.size} bytes)")
        
            # Event ID for ingestion
            event_id = st.text_input(
                "Event ID for these documents:",
                placeholder="e.g., conference_2024_spring",
                key="ingestion_event_id"
            )
            
            if st.button("📥 Ingest Documents", type="primary") and uploaded_files and event_id:
                with st.spinner("Processing documents..."):
                    try:
                        # Save uploaded files temporarily
                        temp_paths = []
                        for file in uploaded_files:
                            temp_path = f"temp_{file.name}"
                            with open(temp_path, "wb") as f:
                                f.write(file.getbuffer())
                            temp_paths.append(temp_path)
                        
                        # Ingest documents
                        result = st.session_state.event_ops_ai.ingest_event_documents(
                            document_paths=temp_paths,
                            event_id=event_id
                        )
                        
                        # Clean up temp files
                        for path in temp_paths:
                            os.remove(path)
                        
                        if result.success:
                            st.success(f"✅ {result.message}")
                            st.metric("Chunks Created", result.chunks_created)
                            st.metric("Categories Found", len(result.categories_found))
                            st.metric("Processing Time", f"{result.processing_time:.2f}s")
                            
                            # Update current event
                            st.session_state.current_event_id = event_id
                            
                        else:
                            st.error(f"❌ Ingestion failed: {result.message}")
                            
                    except Exception as e:
                        st.error(f"❌ Error during ingestion: {e}")
    
    with tab3:
        st.header("📊 Event Analytics")
        
        if not st.session_state.current_event_id:
            st.warning("⚠️ Please select an event first")
        else:
            try:
                overview = st.session_state.event_ops_ai.get_event_overview(st.session_state.current_event_id)
                
                if overview["status"] == "found":
                    st.subheader(f"📈 Event Overview: {st.session_state.current_event_id}")
                    
                    # Metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Chunks", overview["total_chunks"])
                    with col2:
                        st.metric("Categories", len(overview["categories"]))
                    with col3:
                        st.metric("Last Updated", overview["last_updated"][:10])
                    
                    # Categories breakdown
                    st.subheader("📂 Categories Breakdown")
                    if overview["categories"]:
                        import pandas as pd
                        cat_df = pd.DataFrame(list(overview["categories"].items()), 
                                           columns=["Category", "Count"])
                        st.bar_chart(cat_df.set_index("Category"))
                    
                    # Priority distribution
                    st.subheader("🎯 Priority Distribution")
                    if overview["priorities"]:
                        priority_df = pd.DataFrame(list(overview["priorities"].items()),
                                                columns=["Priority", "Count"])
                        st.bar_chart(priority_df.set_index("Priority"))
                    
                    # Sample chunks
                    st.subheader("📝 Sample Content")
                    for i, chunk in enumerate(overview["sample_chunks"]):
                        with st.expander(f"Sample {i+1} - {chunk['category']} ({chunk['priority']})"):
                            st.write(chunk['content'])
                
                else:
                    st.warning("⚠️ No data found for this event")
                    
            except Exception as e:
                st.error(f"❌ Error loading analytics: {e}")
    
    with tab4:
        st.header("🔍 Document Search")
        
        if not st.session_state.current_event_id:
            st.warning("⚠️ Please select an event first")
        else:
            search_query = st.text_input(
                "Search within event documents:",
                placeholder="e.g., catering vendor contact information",
                key="search_query"
            )
            
            # Search filters
            col1, col2 = st.columns(2)
            with col1:
                category_filter = st.selectbox(
                    "Filter by Category:",
                    ["All", "Vendor Information", "Timeline & Schedule", "Crisis Protocol", 
                     "Budget & Finance", "Venue Details", "Staffing & Personnel",
                     "Equipment & Resources", "Safety & Compliance", "Communication Plan",
                     "Contingency Plans"],
                    index=0
                )
            
            with col2:
                priority_filter = st.selectbox(
                    "Filter by Priority:",
                    ["All", "high", "medium", "low"],
                    index=0
                )
            
            if st.button("🔍 Search", type="primary") and search_query:
                with st.spinner("Searching documents..."):
                    try:
                        # Build filters
                        filters = {}
                        if category_filter != "All":
                            filters["category"] = category_filter
                        if priority_filter != "All":
                            filters["priority"] = priority_filter
                        
                        # Perform search
                        results = st.session_state.event_ops_ai.search_event_documents(
                            event_id=st.session_state.current_event_id,
                            query=search_query,
                            filters=filters if filters else None
                        )
                        
                        if results:
                            st.success(f"✅ Found {len(results)} matching chunks")
                            
                            for i, chunk in enumerate(results):
                                with st.expander(f"Result {i+1} - Similarity: {chunk.get('similarity', 0):.3f}"):
                                    st.write(f"**Category:** {chunk.get('metadata', {}).get('category', 'Unknown')}")
                                    st.write(f"**Priority:** {chunk.get('metadata', {}).get('priority', 'Unknown')}")
                                    st.write(f"**Content:** {chunk.get('content', '')}")
                        else:
                            st.warning("⚠️ No matching chunks found")
                            
                    except Exception as e:
                        st.error(f"❌ Search error: {e}")

if __name__ == "__main__":
    main()
