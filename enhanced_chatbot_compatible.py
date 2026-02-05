"""
Enhanced Terminal Chatbot - Compatible Version
Uses your existing working LLM setup + enhanced timing domain prompts
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Import your existing working components
from agent.data_selection_agent import DataSelectionAgent
from agent.llm_config import initialize_ollama_llm, test_ollama_connection

# Import enhanced prompts and validation
from enhanced_prompts import ENHANCED_SYSTEM_PROMPT
from reasoning_validator import TimingReasoningValidator


class CompatibleEnhancedAgent(DataSelectionAgent):
    """
    Enhanced agent that uses your existing working DataSelectionAgent
    but with enhanced timing domain prompts.
    """

    def __init__(self, llm, verbose: bool = True):
        """Initialize with enhanced system prompt."""
        super().__init__(llm, verbose)

        # Override with enhanced system prompt
        self.system_prompt = ENHANCED_SYSTEM_PROMPT

        # Set enhanced environment parameters
        self._set_enhanced_parameters()

        if verbose:
            print("✓ Enhanced agent initialized with timing domain expertise")
            print("✓ Using your existing working LLM configuration")

    def _set_enhanced_parameters(self):
        """Set enhanced parameters via environment variables."""
        enhanced_params = {
            'LLM_TEMPERATURE': '0.2',     # Lower for consistency
            'LLM_TOP_P': '0.9',          # Focused sampling
            'LLM_TOP_K': '40',           # Precision tuned
            'LLM_NUM_PREDICT': '2500',   # More tokens
            'LLM_REPEAT_PENALTY': '1.1'  # Avoid generic phrases
        }

        applied_count = 0
        for param, value in enhanced_params.items():
            if not os.getenv(param):
                os.environ[param] = value
                applied_count += 1

        if applied_count > 0 and self.verbose:
            print(f"✓ Applied {applied_count} enhanced parameters for timing domain")


def print_header(text):
    """Print header without special characters."""
    print("\n" + "=" * 80)
    print(text)
    print("=" * 80 + "\n")


def print_section(text):
    """Print section header."""
    print("\n" + "-" * 80)
    print(text)
    print("-" * 80 + "\n")


def upload_csv():
    """Get CSV file path from user."""
    while True:
        file_path = input("Enter CSV file path: ").strip()

        if not file_path:
            print("Error: Please enter a file path")
            continue

        file_path = file_path.strip('"\'')

        if not Path(file_path).exists():
            print(f"Error: File not found: {file_path}")
            continue

        if not file_path.endswith('.csv'):
            print("Error: File must be CSV format")
            continue

        try:
            import pandas as pd
            df = pd.read_csv(file_path)
            print(f"\nSuccess: Loaded {len(df)} rows, {len(df.columns)} columns")
            print(f"Columns: {', '.join(df.columns.tolist()[:10])}...")
            return file_path
        except Exception as e:
            print(f"Error reading file: {e}")
            continue


def validate_response_quality(reasoning_log, validator):
    """Validate and display response quality."""
    print_section("ENHANCED REASONING QUALITY ANALYSIS")

    try:
        workflow_results = validator.validate_full_workflow(reasoning_log)

        print(f"Overall Quality Score: {workflow_results['workflow_score']:.3f}")
        print(f"Senior Engineer Level: {'✓' if workflow_results['senior_level_achieved'] else '✗'}")

        print("\nQuality Assessment:")
        for stage, quality in workflow_results['stage_results'].items():
            print(f"\n{stage}:")
            print(f"  Score: {quality.overall_score:.3f}")
            print(f"  Expertise: {quality.expertise_level.name}")
            print(f"  Numbers Cited: {quality.specific_numbers_cited}")
            print(f"  Domain Terms: {quality.domain_concepts_used}")
            print(f"  Active Learning: {'✓' if quality.active_learning_explained else '✗'}")
            print(f"  Business Context: {'✓' if quality.business_impact_mentioned else '✗'}")

        if workflow_results['improvement_suggestions']:
            print("\nSuggestions for Next Iteration:")
            for suggestion in workflow_results['improvement_suggestions'][:3]:
                print(f"  • {suggestion}")

        return workflow_results

    except Exception as e:
        print(f"Quality validation error: {e}")
        return {'workflow_score': 0.0, 'senior_level_achieved': False}


def main():
    """Compatible enhanced chatbot main loop."""

    print_header("ENHANCED TIMING DOMAIN AGENT - Compatible Version")
    print("🎯 Senior timing engineer expertise + your working LLM setup")
    print("🔧 Enhanced prompts + proven configuration")

    # Check Ollama connection (using your existing function)
    print("Checking Ollama connection...")
    if not test_ollama_connection():
        print("\n✗ Cannot connect to Ollama")
        print(f"Base URL: {os.getenv('OLLAMA_BASE_URL')}")
        print("Make sure Ollama is running: ollama serve")
        sys.exit(1)

    print("✓ Ollama connected successfully")

    # Initialize LLM (using your existing working function)
    print("\nInitializing your proven LLM configuration...")
    try:
        llm = initialize_ollama_llm()
        print("✓ LLM initialized successfully")
    except Exception as e:
        print(f"✗ LLM initialization failed: {e}")
        sys.exit(1)

    # Initialize enhanced agent (compatible version)
    print("Initializing enhanced timing domain agent...")
    try:
        agent = CompatibleEnhancedAgent(llm=llm, verbose=True)
        validator = TimingReasoningValidator()
        print("✓ Enhanced agent ready\n")
    except Exception as e:
        print(f"✗ Enhanced agent initialization failed: {e}")
        sys.exit(1)

    # Get CSV file
    print_section("UPLOAD TIMING DATA CSV")
    csv_path = upload_csv()

    # Enhanced chat loop
    print_section("ENHANCED TIMING ANALYSIS")
    print("Request intelligent sample selection with timing domain expertise")
    print("Example: 'Select 8% of timing data for Monte Carlo characterization'")
    print("Type 'exit' to quit\n")

    session_results = []

    while True:
        user_input = input("Timing Engineer: ").strip()

        if not user_input:
            continue

        if user_input.lower() == 'exit':
            if session_results:
                avg_score = sum(r.get('workflow_score', 0) for r in session_results) / len(session_results)
                senior_achieved = sum(r.get('senior_level_achieved', False) for r in session_results)
                print(f"\nSession Summary:")
                print(f"  Analyses completed: {len(session_results)}")
                print(f"  Average quality: {avg_score:.3f}")
                print(f"  Senior level achieved: {senior_achieved}/{len(session_results)}")
            print("\n✓ Enhanced timing analysis complete!")
            break

        # Process with enhanced agent
        print(f"\n🔄 Processing with Enhanced Timing Domain AI...")
        print("(Using enhanced prompts + your working configuration)\n")

        try:
            # Run enhanced analysis
            results = agent.run_selection(user_input, csv_path)

            # Display enhanced results
            print_section("ENHANCED SELECTION RESULTS")

            print(f"Dataset: {results['observation']['total_samples']:,} timing samples")
            print(f"Selected: {results['result']['n_selected']:,} samples")
            print(f"Percentage: {results['parsed_params']['selection_percentage']:.1f}%")
            print(f"Clustering: {results['decision']['clustering']['n_clusters']} clusters")
            print(f"Algorithm: {results['decision']['clustering']['algorithm'].upper()}")
            print(f"PCA: {results['decision']['pca']['n_components']} components")
            print(f"Variance: {results['decision']['pca']['variance_explained']*100:.1f}%")

            # Show enhanced reasoning
            print_section("ENHANCED TIMING EXPERTISE REASONING")

            for log_entry in results['reasoning_log']:
                stage = log_entry['stage']
                content = log_entry['content']
                print(f"\n{stage} (Enhanced with Timing Domain Knowledge):")
                print(content)

            # Validate quality improvement
            quality_results = validate_response_quality(
                results['reasoning_log'],
                validator
            )
            session_results.append(quality_results)

            # Save options
            print_section("SAVE RESULTS")
            save_option = input("Save enhanced results? (samples/quality/both/no): ").strip().lower()

            if save_option in ['samples', 'both']:
                output_path = f"enhanced_timing_samples_{results['result']['n_selected']}.csv"
                results['result']['selected_df'].to_csv(output_path, index=False)
                print(f"✓ Enhanced samples saved: {output_path}")

            if save_option in ['quality', 'both']:
                import json
                quality_path = f"timing_quality_analysis_{len(session_results)}.json"
                with open(quality_path, 'w') as f:
                    json.dump(quality_results, f, indent=2, default=str)
                print(f"✓ Quality analysis saved: {quality_path}")

            # Continue option
            print()
            continue_option = input("Run another enhanced analysis? (yes/no): ").strip().lower()
            if continue_option not in ['yes', 'y']:
                if session_results:
                    avg_score = sum(r.get('workflow_score', 0) for r in session_results) / len(session_results)
                    senior_achieved = sum(r.get('senior_level_achieved', False) for r in session_results)
                    print(f"\nFinal Session Summary:")
                    print(f"  Enhanced analyses: {len(session_results)}")
                    print(f"  Average quality: {avg_score:.3f}")
                    print(f"  Senior level: {senior_achieved}/{len(session_results)}")
                print("\n🎯 Enhanced timing domain analysis complete!")
                break

            print_section("ENHANCED TIMING ANALYSIS")

        except Exception as e:
            print(f"✗ Enhanced processing error: {e}")
            print("The enhanced prompts may need adjustment for your setup")

            # Offer fallback
            fallback = input("Try with original agent? (yes/no): ").strip().lower()
            if fallback in ['yes', 'y']:
                try:
                    original_agent = DataSelectionAgent(llm=llm, verbose=False)
                    results = original_agent.run_selection(user_input, csv_path)
                    print("✓ Fallback to original agent successful")
                except Exception as e2:
                    print(f"✗ Fallback also failed: {e2}")


if __name__ == "__main__":
    main()