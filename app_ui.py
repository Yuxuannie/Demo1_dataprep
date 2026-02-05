"""
Streamlit Chatbox UI for Timing-Aware Data Selection Agent
Conversational interface with continuous interaction and modification support
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import tempfile
from io import BytesIO
from pathlib import Path
import time

# Configure Streamlit page
st.set_page_config(
    page_title="Timing Agent Chat",
    page_icon="[T]",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for chatbox styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.0rem;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 1rem;
        max-width: 80%;
    }
    .user-message {
        background: #e3f2fd;
        border: 1px solid #bbdefb;
        margin-left: auto;
        text-align: right;
    }
    .agent-message {
        background: #f1f8e9;
        border: 1px solid #c8e6c9;
        margin-right: auto;
    }
    .system-message {
        background: #fff3e0;
        border: 1px solid #ffcc02;
        text-align: center;
        font-style: italic;
        color: #e65100;
    }
    .reasoning-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f4e79;
        margin: 0.5rem 0;
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 0.8rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin: 0.3rem;
        text-align: center;
    }
    .chat-input {
        position: fixed;
        bottom: 0;
        width: 100%;
        background: white;
        border-top: 2px solid #1f4e79;
        padding: 1rem;
    }
    .stTextInput > div > div > input {
        font-size: 1.1rem !important;
    }
</style>
""", unsafe_allow_html=True)


def initialize_agent():
    """Initialize timing-aware agent with error handling."""
    try:
        from agent.timing_data_selection_agent import TimingDataSelectionAgent
        from agent.timing_llm_config import initialize_timing_llm, test_ollama_connection

        # Test Ollama connection
        if not test_ollama_connection():
            st.error("[ERROR] Cannot connect to Ollama. Please ensure Ollama is running: `ollama serve`")
            return None

        # Initialize LLM and agent
        llm = initialize_timing_llm()
        agent = TimingDataSelectionAgent(llm=llm, verbose=True)

        return agent

    except Exception as e:
        st.error(f"[ERROR] Failed to initialize agent: {e}")
        return None


def create_download_csv(df, filename="selected_data.csv"):
    """Create downloadable CSV from DataFrame."""
    buffer = BytesIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue()


def display_reasoning_log(reasoning_log):
    """Display reasoning log with proper formatting."""
    st.markdown("### Reasoning Senior Timing Engineer Reasoning")

    for i, log_entry in enumerate(reasoning_log):
        stage = log_entry['stage']
        content = log_entry['content']

        # Create expandable sections for each reasoning stage
        with st.expander(f"**{stage}**", expanded=(i == 0)):
            st.markdown(f"""
            <div class="reasoning-box">
            {content}
            </div>
            """, unsafe_allow_html=True)


def display_visual_dashboard(results, csv_path):
    """Display visual validation dashboard."""
    st.markdown("### Dashboard Visual Validation Dashboard")

    try:
        from plot_utils import TimingVisualizationDashboard

        # Create dashboard
        dashboard = TimingVisualizationDashboard()

        # Load CSV data for dashboard
        df = pd.read_csv(csv_path)

        # Generate plots using correct method name and data structure
        fig, plot_metadata = dashboard.generate_dashboard_plots(
            df=df,
            selected_indices=results.get('result', {}).get('selected_indices', []),
            clusters=results.get('decision', {}).get('clustering', {}).get('labels', []),
            centroids=results.get('decision', {}).get('clustering', {}).get('centroids', []),
            pca_components=results.get('decision', {}).get('pca', {}).get('n_components', 2)
        )

        if fig:
            st.pyplot(fig)

            # Add interpretation
            st.markdown("""
            **Analysis Dashboard Interpretation:**
            - **Left**: PCA cluster visualization showing selected boundary samples (red dots)
            - **Center**: Feature correlation heatmap justifying PCA dimensionality reduction
            - **Right**: Distribution overlay proving selected samples cover critical tail regions
            """)
        else:
            st.warning("[WARNING] Could not generate visualization dashboard")

    except Exception as e:
        st.error(f"[ERROR] Visualization error: {e}")


def add_message_to_chat(role, content, message_type="text", metadata=None):
    """Add message to chat history."""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    message = {
        'role': role,
        'content': content,
        'type': message_type,
        'timestamp': time.time(),
        'metadata': metadata or {}
    }
    st.session_state.chat_history.append(message)


def display_chat_message(message):
    """Display a single chat message."""
    role = message['role']
    content = message['content']
    message_type = message.get('type', 'text')
    metadata = message.get('metadata', {})

    if role == 'user':
        st.markdown(f"""
        <div class="chat-message user-message">
        <strong>[USER] You:</strong><br>
        {content}
        </div>
        """, unsafe_allow_html=True)

    elif role == 'agent':
        st.markdown(f"""
        <div class="chat-message agent-message">
        <strong>[AGENT] Timing Agent:</strong><br>
        {content}
        </div>
        """, unsafe_allow_html=True)

        # Display additional content based on message type
        if message_type == 'analysis_results' and 'results' in metadata:
            display_analysis_results(metadata['results'])

        elif message_type == 'visual_dashboard' and 'results' in metadata and 'csv_path' in metadata:
            display_visual_dashboard(metadata['results'], metadata['csv_path'])

    elif role == 'system':
        st.markdown(f"""
        <div class="chat-message system-message">
        [INFO] {content}
        </div>
        """, unsafe_allow_html=True)


def display_analysis_results(results):
    """Display analysis results in compact chat format."""
    # Key metrics in horizontal layout
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
        <strong>Total</strong><br>
        {results['observation']['total_samples']:,}
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
        <strong>Selected</strong><br>
        {results['result']['n_selected']:,}
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
        <strong>Rate</strong><br>
        {results['parsed_params']['selection_percentage']:.1f}%
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
        <strong>Cost Saving</strong><br>
        {results['result']['expected_cost_reduction']}
        </div>
        """, unsafe_allow_html=True)

    # Reasoning log in expandable sections
    with st.expander("Reasoning View Detailed Reasoning"):
        for log_entry in results['reasoning_log']:
            stage = log_entry['stage']
            content = log_entry['content']
            st.markdown(f"""
            <div class="reasoning-box">
            <strong>{stage}:</strong><br>
            {content}
            </div>
            """, unsafe_allow_html=True)

    # Download button
    if 'selected_df' in results['result']:
        csv_data = create_download_csv(
            results['result']['selected_df'],
            f"timing_selected_{results['result']['n_selected']}_samples.csv"
        )

        st.download_button(
            label="Download Download Selected Data",
            data=csv_data,
            file_name=f"timing_selected_{results['result']['n_selected']}_samples.csv",
            mime="text/csv",
            key=f"download_{len(st.session_state.chat_history)}"
        )


def handle_user_request(user_input, agent, csv_path):
    """Process user request and return agent response."""
    try:
        # Check if this is a modification request
        if any(keyword in user_input.lower() for keyword in ['change', 'modify', 'adjust', 'increase', 'decrease', 'different']):
            # This is a modification request - provide guidance
            response = f"""I understand you want to modify the analysis. Here are some examples of what you can request:

**Percentage Changes:**
- "Change selection percentage to 3%"
- "Increase to 10% for more comprehensive coverage"
- "Reduce to 2% for faster characterization"

**Algorithm Changes:**
- "Use K-means instead of GMM"
- "Try different clustering approach"

**Focus Changes:**
- "Focus on high-variability paths only"
- "Select more boundary cases"
- "Emphasize timing corners"

**Analysis Scope:**
- "Analyze only specific cell types"
- "Focus on critical timing paths"

Please specify what you'd like to modify, and I'll rerun the analysis with your new requirements."""

            return response, None

        # Run the analysis
        results = agent.run_selection(user_input, csv_path)

        # Create response message
        response_lines = []
        response_lines.append(f"[OK] **Analysis Complete!**")
        response_lines.append(f"")
        response_lines.append(f"Selected **{results['result']['n_selected']:,}** samples from **{results['observation']['total_samples']:,}** total")
        response_lines.append(f"Selection rate: **{results['parsed_params']['selection_percentage']:.1f}%**")
        response_lines.append(f"Algorithm: **{results['decision']['clustering']['algorithm'].upper()}** with **{results['decision']['clustering']['n_clusters']}** clusters")
        response_lines.append(f"Expected cost reduction: **{results['result']['expected_cost_reduction']}**")
        response_lines.append(f"")
        response_lines.append(f"Tips You can ask me to modify this analysis or request different parameters!")

        response = "\n".join(response_lines)

        return response, results

    except Exception as e:
        error_response = f"""[ERROR] **Analysis Error:** {str(e)}

Please check:
- Your CSV file is properly formatted
- The request is clear (e.g., "Select 5% of data")
- Ollama is running and connected

Try rephrasing your request or upload a different CSV file."""

        return error_response, None


def main():
    """Main Streamlit chatbox application."""

    # Header
    st.markdown("""
    <div class="main-header">Chat Timing Agent Chat</div>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    if 'csv_path' not in st.session_state:
        st.session_state.csv_path = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'current_results' not in st.session_state:
        st.session_state.current_results = None

    # Sidebar - Settings and Configuration
    with st.sidebar:
        st.markdown("## Setup Setup")

        # Agent initialization
        st.markdown("### [AGENT] Agent Status")
        if st.button("Refresh Initialize Agent", type="primary", use_container_width=True):
            with st.spinner("Initializing timing agent..."):
                agent = initialize_agent()
                if agent:
                    st.session_state.agent = agent
                    add_message_to_chat("system", "Timing agent initialized successfully! Upload a CSV file to begin.")

        if st.session_state.agent:
            st.success("[OK] Agent Ready")
        else:
            st.warning("[!] Please initialize agent")

        st.markdown("---")

        # File upload
        st.markdown("### Data Data Upload")
        uploaded_file = st.file_uploader(
            "Upload Timing CSV",
            type=['csv'],
            help="Upload timing characterization data"
        )

        if uploaded_file:
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(mode='wb', suffix='.csv', delete=False) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    csv_path = tmp_file.name
                    st.session_state.csv_path = csv_path

                # Display file info
                df = pd.read_csv(csv_path)
                st.success(f"[OK] {len(df):,} rows loaded")

                # Add system message if this is a new file
                if len(st.session_state.chat_history) == 0 or not any(
                    msg.get('metadata', {}).get('file_loaded') for msg in st.session_state.chat_history
                ):
                    add_message_to_chat(
                        "system",
                        f"CSV file loaded: {len(df):,} rows, {len(df.columns)} columns. Ready for analysis!",
                        metadata={'file_loaded': True}
                    )

            except Exception as e:
                st.error(f"[ERROR] Error: {e}")

        st.markdown("---")

        # Quick actions
        st.markdown("### Actions Quick Actions")
        if st.session_state.agent and st.session_state.csv_path:

            if st.button("Dashboard Standard 5% Selection", use_container_width=True):
                st.session_state.pending_request = "Select 5% of timing data for Monte Carlo characterization"

            if st.button("Target Conservative 3% Selection", use_container_width=True):
                st.session_state.pending_request = "Select 3% of timing data with focus on boundary cases"

            if st.button("Analysis Comprehensive 8% Analysis", use_container_width=True):
                st.session_state.pending_request = "Select 8% of timing data for comprehensive library characterization"

        # Clear chat
        st.markdown("---")
        if st.button("Clear Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.current_results = None
            st.rerun()

    # Main chat area
    st.markdown("## Chat Chat with Timing Agent")

    # Chat display container
    chat_container = st.container()

    with chat_container:
        # Display welcome message if no chat history
        if not st.session_state.chat_history:
            st.markdown("""
            <div class="chat-message system-message">
            [AGENT] Welcome! I'm your Senior Timing Engineer AI assistant.
            <br><br>
            <strong>Get started:</strong>
            <br>1. Initialize the agent in the sidebar
            <br>2. Upload your timing data CSV
            <br>3. Ask me to analyze your data!
            <br><br>
            <strong>Example requests:</strong>
            <br>• "Select 5% of timing data for library characterization"
            <br>• "Focus on high-variability timing paths"
            <br>• "Use K-means clustering with 8 clusters"
            </div>
            """, unsafe_allow_html=True)

        # Display chat history
        for message in st.session_state.chat_history:
            display_chat_message(message)

    # Handle pending requests from quick actions
    if hasattr(st.session_state, 'pending_request'):
        user_input = st.session_state.pending_request
        del st.session_state.pending_request

        # Add user message
        add_message_to_chat("user", user_input)

        # Process request
        with st.spinner("[AGENT] Analyzing..."):
            response, results = handle_user_request(user_input, st.session_state.agent, st.session_state.csv_path)

        # Add agent response
        if results:
            add_message_to_chat("agent", response, "analysis_results", {"results": results})
            st.session_state.current_results = results
        else:
            add_message_to_chat("agent", response)

        st.rerun()

    # Chat input
    st.markdown("---")
    user_input = st.chat_input(
        placeholder="Ask me to analyze your timing data... (e.g., 'Select 6% using GMM clustering')",
        disabled=not (st.session_state.agent and st.session_state.csv_path)
    )

    if user_input:
        # Add user message to chat
        add_message_to_chat("user", user_input)

        # Process the request
        with st.spinner("[AGENT] Analyzing..."):
            response, results = handle_user_request(user_input, st.session_state.agent, st.session_state.csv_path)

        # Add agent response
        if results:
            add_message_to_chat("agent", response, "analysis_results", {"results": results})
            st.session_state.current_results = results

            # Ask if user wants to see visualizations
            if st.session_state.current_results:
                viz_response = "Would you like to see the visual validation dashboard for this analysis?"
                add_message_to_chat("agent", viz_response)

        else:
            add_message_to_chat("agent", response)

        # Rerun to show new messages
        st.rerun()

    # Visual dashboard toggle
    if st.session_state.current_results and st.session_state.csv_path:
        st.markdown("---")
        col1, col2 = st.columns([1, 1])

        with col1:
            if st.button("Dashboard Show Visual Dashboard", type="secondary", use_container_width=True):
                add_message_to_chat(
                    "agent",
                    "Here's the visual validation dashboard showing cluster analysis and sample distribution:",
                    "visual_dashboard",
                    {"results": st.session_state.current_results, "csv_path": st.session_state.csv_path}
                )
                st.rerun()

        with col2:
            if st.button("Tips Suggest Modifications", use_container_width=True):
                suggestions = """Here are some modifications you could try:

**Different Percentages:**
• "Change to 3% for faster characterization"
• "Increase to 8% for better coverage"

**Algorithm Changes:**
• "Try K-means instead of GMM"
• "Use more clusters for finer granularity"

**Focus Areas:**
• "Focus on high-sigma timing paths"
• "Prioritize boundary cases"
• "Select critical timing corners only"

What would you like to modify?"""

                add_message_to_chat("agent", suggestions)
                st.rerun()


if __name__ == "__main__":
    main()