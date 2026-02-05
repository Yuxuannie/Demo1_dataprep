"""
Enhanced Terminal-based Chatbot with Timing Domain Expertise
Uses EnhancedDataSelectionAgent with senior engineer level reasoning
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load enhanced environment
load_dotenv()

from enhanced_data_selection_agent import EnhancedDataSelectionAgent
from enhanced_llm_config import initialize_enhanced_ollama_llm
from reasoning_validator import TimingReasoningValidator
from agent.llm_config import test_ollama_connection


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
    print_section("RESPONSE QUALITY ANALYSIS")

    workflow_results = validator.validate_full_workflow(reasoning_log)

    print(f"Overall Workflow Score: {workflow_results['workflow_score']:.3f}")
    print(f"Senior Engineer Level Achieved: {'✓' if workflow_results['senior_level_achieved'] else '✗'}")

    print("\nStage-by-Stage Analysis:")
    for stage, quality in workflow_results['stage_results'].items():
        print(f"\n{stage}:")
        print(f"  Score: {quality.overall_score:.3f}")
        print(f"  Expertise: {quality.expertise_level.name}")
        print(f"  Numbers Cited: {quality.specific_numbers_cited}")
        print(f"  Domain Concepts: {quality.domain_concepts_used}")
        print(f"  Active Learning: {'✓' if quality.active_learning_explained else '✗'}")
        print(f"  Business Impact: {'✓' if quality.business_impact_mentioned else '✗'}")

    if workflow_results['improvement_suggestions']:
        print("\nImprovement Suggestions:")
        for suggestion in workflow_results['improvement_suggestions']:
            print(f"  • {suggestion}")

    return workflow_results


def main():
    """Enhanced chatbot main loop."""

    print_header("ENHANCED DATA SELECTION AGENT - Senior Timing Engineer AI")
    print("Powered by domain-specific prompts and active learning principles")

    # Check Ollama connection
    print("Checking Ollama connection...")
    if not test_ollama_connection():
        print("\nError: Cannot connect to Ollama")
        print(f"Base URL: {os.getenv('OLLAMA_BASE_URL')}")
        print("Make sure Ollama is running: ollama serve")
        sys.exit(1)

    print("✓ Ollama connected")

    # Initialize enhanced LLM
    print("\nInitializing Enhanced LLM with timing domain parameters...")
    try:
        llm = initialize_enhanced_ollama_llm()
        print("✓ Enhanced LLM initialized with optimized parameters")
    except Exception as e:
        print(f"Error initializing enhanced LLM: {e}")
        sys.exit(1)

    # Initialize enhanced agent
    print("Initializing Enhanced Timing Domain Agent...")
    try:
        agent = EnhancedDataSelectionAgent(llm=llm, verbose=False)
        validator = TimingReasoningValidator()
        print("✓ Enhanced Agent and Validator initialized\n")
    except Exception as e:
        print(f"Error initializing enhanced agent: {e}")
        sys.exit(1)

    # Get CSV file
    print_section("UPLOAD TIMING DATA CSV")
    csv_path = upload_csv()

    # Enhanced chat loop
    print_section("ENHANCED TIMING ANALYSIS")
    print("Ask for intelligent Monte Carlo sample selection")
    print("Example: 'Select 8% of timing samples for library characterization'")
    print("Type 'exit' to quit\n")

    session_results = []

    while True:
        user_input = input("Timing Engineer: ").strip()

        if not user_input:
            continue

        if user_input.lower() == 'exit':
            print("\n=== SESSION SUMMARY ===")
            if session_results:
                avg_score = sum(r['workflow_score'] for r in session_results) / len(session_results)
                senior_achieved = sum(r['senior_level_achieved'] for r in session_results)
                print(f"Sessions completed: {len(session_results)}")
                print(f"Average quality score: {avg_score:.3f}")
                print(f"Senior engineer level achieved: {senior_achieved}/{len(session_results)}")
            print("\nGoodbye!")
            break

        # Process enhanced request
        print("\nProcessing with Enhanced Timing Domain AI...")
        print("(Using domain expertise prompts and active learning principles)\n")

        try:
            # Run enhanced agent
            results = agent.run_selection(user_input, csv_path)

            # Display enhanced results
            print_section("ENHANCED SELECTION COMPLETE")

            print(f"Total Timing Samples: {results['observation']['total_samples']:,}")
            print(f"Selected Samples: {results['result']['n_selected']:,}")
            print(f"Selection Percentage: {results['parsed_params']['selection_percentage']:.1f}%")
            print(f"Clusters Created: {results['decision']['clustering']['n_clusters']}")
            print(f"Algorithm Used: {results['decision']['clustering']['algorithm'].upper()}")
            print(f"PCA Components: {results['decision']['pca']['n_components']}")
            print(f"Variance Preserved: {results['decision']['pca']['variance_explained']*100:.1f}%")
            print(f"Enhancement Level: {results['enhancement_level']}")
            print(f"Expected Cost Reduction: {results['result']['expected_cost_reduction']}")

            # Show enhanced reasoning
            print_section("ENHANCED REASONING STAGES")

            for log_entry in results['reasoning_log']:
                stage = log_entry['stage']
                content = log_entry['content']
                print(f"\n{stage} (Enhanced with Domain Expertise):")
                print(content)

            # Validate response quality
            quality_results = validate_response_quality(
                results['reasoning_log'],
                validator
            )
            session_results.append(quality_results)

            # Enhanced save options
            print_section("SAVE ENHANCED RESULTS")
            save_option = input("Save results? (samples/analysis/both/no): ").strip().lower()

            if save_option in ['samples', 'both']:
                output_path = f"enhanced_timing_selection_{results['observation']['total_samples']}_{results['result']['n_selected']}.csv"
                results['result']['selected_df'].to_csv(output_path, index=False)
                print(f"✓ Enhanced samples saved: {output_path}")

            if save_option in ['analysis', 'both']:
                analysis_path = f"timing_analysis_{results['observation']['total_samples']}.json"
                analysis_data = {
                    'workflow_quality': quality_results,
                    'reasoning_log': results['reasoning_log'],
                    'enhancement_summary': results.get('enhancement_level', 'ENHANCED'),
                    'business_impact': results['result']['expected_cost_reduction']
                }
                with open(analysis_path, 'w') as f:
                    json.dump(analysis_data, f, indent=2, default=str)
                print(f"✓ Quality analysis saved: {analysis_path}")

            # Continue option
            print()
            continue_option = input("Run another enhanced analysis? (yes/no): ").strip().lower()
            if continue_option not in ['yes', 'y']:
                print("\n=== SESSION SUMMARY ===")
                if session_results:
                    avg_score = sum(r['workflow_score'] for r in session_results) / len(session_results)
                    senior_achieved = sum(r['senior_level_achieved'] for r in session_results)
                    print(f"Enhanced sessions completed: {len(session_results)}")
                    print(f"Average quality score: {avg_score:.3f}")
                    print(f"Senior engineer level achieved: {senior_achieved}/{len(session_results)}")
                print("\nEnhanced Timing AI session complete!")
                break

            print_section("ENHANCED TIMING ANALYSIS")
            print("Enter your next timing characterization request:")

        except Exception as e:
            print(f"Error in enhanced processing: {e}")
            import traceback
            print(traceback.format_exc())


if __name__ == "__main__":
    main()
