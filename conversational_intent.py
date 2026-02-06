"""
Intent Classification System for Conversational Q&A vs Execution
Prevents re-running pipeline for follow-up questions
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum

class UserIntent(Enum):
    """User intent categories."""
    EXECUTE_SAMPLING = "execute_sampling"
    QUESTION_ABOUT_RESULTS = "question_about_results"
    MODIFY_PARAMETERS = "modify_parameters"
    EXPLAIN_METHODOLOGY = "explain_methodology"
    REQUEST_VISUALIZATION = "request_visualization"
    GENERAL_HELP = "general_help"

class ConversationalAgent:
    """
    Enhanced agent with conversational state management and intent classification.
    """

    def __init__(self, base_agent):
        self.base_agent = base_agent
        self.conversation_state = {
            'last_execution': None,
            'previous_results': None,
            'selection_log': [],
            'context_history': [],
            'current_dataset_info': None
        }

    def classify_user_intent(self, user_input: str) -> Tuple[UserIntent, Dict[str, Any]]:
        """
        Classify user intent to determine whether to execute pipeline or answer from context.

        Args:
            user_input: User's input text

        Returns:
            Tuple of (intent, extracted_parameters)
        """

        input_lower = user_input.lower().strip()

        # Intent patterns with priorities (most specific first)
        intent_patterns = {
            UserIntent.QUESTION_ABOUT_RESULTS: [
                r'why did you (choose|pick|select)',
                r'why.*(\d+)%',
                r'explain (the|your) (selection|choice|decision)',
                r'how did you (decide|determine)',
                r'what.*reasoning.*behind',
                r'justify.*selection',
                r'(why|how).*samples',
                r'rationale.*for'
            ],

            UserIntent.MODIFY_PARAMETERS: [
                r'change.*to.*(\d+)%',
                r'try.*(\d+)%.*instead',
                r'use.*(\d+).*samples',
                r'increase.*to.*(\d+)',
                r'decrease.*to.*(\d+)',
                r'modify.*percentage',
                r'adjust.*selection'
            ],

            UserIntent.REQUEST_VISUALIZATION: [
                r'show.*plot',
                r'visuali[sz]e.*results',
                r'generate.*dashboard',
                r'plot.*samples',
                r'show.*scatter',
                r'display.*chart'
            ],

            UserIntent.EXPLAIN_METHODOLOGY: [
                r'how does.*work',
                r'explain.*algorithm',
                r'what.*method.*using',
                r'describe.*approach',
                r'methodology'
            ],

            UserIntent.EXECUTE_SAMPLING: [
                r'select.*(\d+)%',
                r'run.*sampling',
                r'perform.*selection',
                r'execute.*analysis',
                r'analyze.*dataset',
                r'sample.*(\d+)',
                r'choose.*samples'
            ],

            UserIntent.GENERAL_HELP: [
                r'help',
                r'what.*can.*do',
                r'how.*use',
                r'commands'
            ]
        }

        # Check patterns in order of priority
        for intent, patterns in intent_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, input_lower)
                if match:
                    # Extract parameters from match
                    params = self._extract_parameters(user_input, match)
                    return intent, params

        # Default to execution if no clear pattern matches
        return UserIntent.EXECUTE_SAMPLING, {}

    def _extract_parameters(self, user_input: str, match: re.Match) -> Dict[str, Any]:
        """Extract parameters from matched intent patterns."""
        params = {}

        # Extract percentages
        percentage_matches = re.findall(r'(\d+)%', user_input)
        if percentage_matches:
            params['percentage'] = int(percentage_matches[0])

        # Extract sample counts
        sample_matches = re.findall(r'(\d+)\s*samples?', user_input, re.IGNORECASE)
        if sample_matches:
            params['sample_count'] = int(sample_matches[0])

        # Extract algorithm names
        algorithm_matches = re.findall(r'(k-means|gmm|clustering)', user_input, re.IGNORECASE)
        if algorithm_matches:
            params['algorithm'] = algorithm_matches[0].lower()

        return params

    def handle_user_input(self, user_input: str, csv_path: str = None) -> Dict[str, Any]:
        """
        Main handler that classifies intent and routes to appropriate response.

        Args:
            user_input: User's input text
            csv_path: Path to CSV file (only needed for execution)

        Returns:
            Dictionary containing response and metadata
        """

        intent, params = self.classify_user_intent(user_input)

        print(f"[INTENT] Classified as: {intent.value}")
        if params:
            print(f"[PARAMS] Extracted: {params}")

        # Route based on intent
        if intent == UserIntent.EXECUTE_SAMPLING:
            return self._handle_execution(user_input, csv_path, params)

        elif intent == UserIntent.QUESTION_ABOUT_RESULTS:
            return self._handle_results_question(user_input, params)

        elif intent == UserIntent.MODIFY_PARAMETERS:
            return self._handle_parameter_modification(user_input, params)

        elif intent == UserIntent.REQUEST_VISUALIZATION:
            return self._handle_visualization_request(params)

        elif intent == UserIntent.EXPLAIN_METHODOLOGY:
            return self._handle_methodology_explanation(user_input)

        else:
            return self._handle_general_help()

    def _handle_execution(self, user_input: str, csv_path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the full sampling pipeline."""
        print("[EXECUTION] Running full sampling pipeline...")

        try:
            # Run the base agent
            results = self.base_agent.run_selection(user_input, csv_path)

            # Update conversation state
            self.conversation_state['last_execution'] = results
            self.conversation_state['previous_results'] = results
            self.conversation_state['selection_log'].append({
                'timestamp': pd.Timestamp.now(),
                'user_input': user_input,
                'results': results,
                'intent': 'execute_sampling'
            })

            return {
                'response_type': 'execution',
                'results': results,
                'message': "✅ Sampling execution completed successfully"
            }

        except Exception as e:
            return {
                'response_type': 'error',
                'error': str(e),
                'message': f"❌ Execution failed: {e}"
            }

    def _handle_results_question(self, user_input: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Answer questions about previous results without re-execution."""
        print("[Q&A] Answering from previous context...")

        if not self.conversation_state['previous_results']:
            return {
                'response_type': 'no_context',
                'message': "❌ No previous results to explain. Please run sampling first."
            }

        # Get previous results
        results = self.conversation_state['previous_results']

        # Generate contextual answer based on question type
        if 'why' in user_input.lower() and any(str(p) in user_input for p in [results['parsed_params'].get('selection_percentage', 0)]):
            # Question about percentage choice
            percentage = results['parsed_params']['selection_percentage']
            reasoning = results.get('reasoning_log', [])

            response = f"""
The {percentage}% selection was determined based on:

**Data-Driven Analysis:**
- Dataset size: {results['observation']['total_samples']:,} samples
- Target efficiency: Balance between coverage and computational cost

**Technical Reasoning:**
"""
            for log_entry in reasoning:
                if log_entry['stage'] in ['THINK', 'DECIDE']:
                    response += f"\n• {log_entry['stage']}: {log_entry['content'][:200]}..."

            response += f"""

**Final Selection:**
- Selected {results['result']['n_selected']:,} samples ({percentage:.1f}%)
- Algorithm: {results['decision']['clustering']['algorithm'].upper()}
- Clusters: {results['decision']['clustering']['n_clusters']}
- Expected cost reduction: {results['result']['expected_cost_reduction']}
"""

        else:
            # General explanation
            response = f"""
**Previous Sampling Results Summary:**

• **Selected:** {results['result']['n_selected']:,} samples from {results['observation']['total_samples']:,} total
• **Method:** {results['decision']['clustering']['algorithm'].upper()} with uncertainty-based sampling
• **Coverage:** {results['decision']['clustering']['n_clusters']} clusters for timing path diversity
• **Business Impact:** {results['result']['expected_cost_reduction']}

**Key Decision Factors:**
"""
            for log_entry in results['reasoning_log']:
                response += f"\n• **{log_entry['stage']}:** {log_entry['content'][:150]}..."

        return {
            'response_type': 'explanation',
            'message': response,
            'context_used': True
        }

    def _handle_parameter_modification(self, user_input: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle requests to modify parameters and re-run."""
        print("[MODIFY] Parameter modification requested...")

        if 'percentage' in params:
            new_percentage = params['percentage']
            message = f"🔄 Modifying selection percentage to {new_percentage}%. Re-running analysis..."

            # Modify the user input for re-execution
            modified_input = f"Select {new_percentage}% of timing data for analysis"

            return {
                'response_type': 'modification',
                'message': message,
                'modified_input': modified_input,
                'requires_re_execution': True
            }

        else:
            return {
                'response_type': 'clarification_needed',
                'message': "❓ Please specify what you'd like to modify (e.g., 'Change to 8%')"
            }

    def _handle_visualization_request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle requests for visualization without re-execution."""
        print("[VISUALIZATION] Generating plots from previous results...")

        if not self.conversation_state['previous_results']:
            return {
                'response_type': 'no_context',
                'message': "❌ No results to visualize. Please run sampling first."
            }

        return {
            'response_type': 'visualization',
            'message': "📊 Generating interactive visualization dashboard...",
            'generate_plots': True
        }

    def _handle_methodology_explanation(self, user_input: str) -> Dict[str, Any]:
        """Explain the methodology without execution."""
        methodology_explanation = """
**Agentic Timing Data Selection Methodology:**

**1. OBSERVE Stage:**
- Analyzes dataset characteristics (correlations, variability, cell types)
- Calculates timing-specific statistics (sigma distributions, delay patterns)
- Identifies data complexity and clustering potential

**2. THINK Stage:**
- Develops custom sampling strategy based on data patterns
- Compares approaches: representative vs boundary sampling
- Considers ML model training requirements and simulation budget

**3. ACT Stage:**
- Executes uncertainty-based sampling (samples far from centroids)
- Applies PCA for dimensionality reduction
- Uses GMM/K-means clustering for timing path grouping
- Validates selection quality and iterates if needed

**Key Principles:**
• **Active Learning**: Focus on high-uncertainty samples for model robustness
• **Timing Domain Expertise**: Understand process variation and corner cases
• **Business Value**: Balance simulation cost vs characterization accuracy
"""

        return {
            'response_type': 'methodology',
            'message': methodology_explanation
        }

    def _handle_general_help(self) -> Dict[str, Any]:
        """Provide general help information."""
        help_text = """
**Available Commands:**

**Execution:**
• "Select 5% of timing data" - Run sampling analysis
• "Analyze dataset with 8% selection" - Custom percentage

**Questions & Analysis:**
• "Why did you pick 30%?" - Explain previous decisions
• "Explain your reasoning" - Show methodology
• "How does uncertainty sampling work?" - Technical details

**Modifications:**
• "Change to 3%" - Modify percentage and re-run
• "Use K-means instead" - Different algorithm

**Visualization:**
• "Show plots" - Generate interactive dashboard
• "Visualize results" - Display sampling patterns

**Context:** I maintain conversation history and can answer follow-up questions without re-running the entire analysis.
"""

        return {
            'response_type': 'help',
            'message': help_text
        }