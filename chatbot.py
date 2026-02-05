"""
Terminal-based Chatbot for Timing-Aware Data Selection Agent
Senior timing engineer expertise for Monte Carlo sample selection
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment
load_dotenv()

from agent.timing_data_selection_agent import TimingDataSelectionAgent
from agent.timing_llm_config import initialize_timing_llm, test_ollama_connection


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
            print(f"Columns: {', '.join(df.columns.tolist())}")
            return file_path
        except Exception as e:
            print(f"Error reading file: {e}")
            continue


def main():
    """Timing-aware chatbot main loop."""

    print_header("TIMING-AWARE DATA SELECTION AGENT")
    print("Senior timing engineer expertise for Monte Carlo sample selection")

    # Check Ollama connection
    print("Checking Ollama connection...")
    if not test_ollama_connection():
        print("\nError: Cannot connect to Ollama")
        print(f"Base URL: {os.getenv('OLLAMA_BASE_URL')}")
        print("Make sure Ollama is running: ollama serve")
        sys.exit(1)

    print("Success: Ollama connected")

    # Initialize timing-optimized LLM
    print("\nInitializing timing domain LLM...")
    try:
        llm = initialize_timing_llm()
        print("Success: Timing domain LLM initialized")
    except Exception as e:
        print(f"Error initializing timing LLM: {e}")
        sys.exit(1)

    # Initialize timing-aware agent
    print("Initializing timing-aware agent...")
    try:
        agent = TimingDataSelectionAgent(llm=llm, verbose=False)
        print("Success: Timing-aware agent initialized\n")
    except Exception as e:
        print(f"Error initializing timing agent: {e}")
        sys.exit(1)

    # Get CSV file
    print_section("UPLOAD TIMING DATA CSV")
    csv_path = upload_csv()

    # Chat loop
    print_section("TIMING DOMAIN ANALYSIS")
    print("Request intelligent Monte Carlo sample selection")
    print("Example: 'Select 8% of timing data for library characterization'")
    print("Type 'exit' to quit\n")

    while True:
        user_input = input("Timing Engineer: ").strip()

        if not user_input:
            continue

        if user_input.lower() == 'exit':
            print("\nGoodbye!")
            break

        print("\nProcessing with timing domain expertise...")
        print("(Senior timing engineer reasoning + active learning principles)\n")

        try:
            # Run timing-aware analysis
            results = agent.run_selection(user_input, csv_path)

            # Display results
            print_section("TIMING ANALYSIS COMPLETE")

            print(f"Total Timing Samples: {results['observation']['total_samples']}")
            print(f"Selected Samples: {results['result']['n_selected']}")
            print(f"Selection Percentage: {results['parsed_params']['selection_percentage']:.1f}%")
            print(f"Number of Clusters: {results['decision']['clustering']['n_clusters']}")
            print(f"PCA Components: {results['decision']['pca']['n_components']}")
            print(f"PCA Variance Explained: {results['decision']['pca']['variance_explained']*100:.1f}%")
            print(f"Clustering Algorithm: {results['decision']['clustering']['algorithm'].upper()}")
            print(f"Expected Cost Reduction: {results['result']['expected_cost_reduction']}")

            # Show timing expertise reasoning
            print_section("TIMING DOMAIN REASONING")

            for log_entry in results['reasoning_log']:
                stage = log_entry['stage']
                content = log_entry['content']
                print(f"\n{stage}:")
                print(content)

            # Save results
            print_section("SAVE RESULTS")
            save_option = input("Save selected samples to CSV? (yes/no): ").strip().lower()

            if save_option in ['yes', 'y']:
                output_path = f"timing_selected_samples_{results['observation']['total_samples']}_{results['result']['n_selected']}.csv"
                results['result']['selected_df'].to_csv(output_path, index=False)
                print(f"Saved to: {output_path}")

            # Continue option
            print()
            continue_option = input("Run another timing analysis? (yes/no): ").strip().lower()
            if continue_option not in ['yes', 'y']:
                print("\nTiming analysis session complete!")
                break

            print_section("TIMING DOMAIN ANALYSIS")
            print("Enter your next timing characterization request")
            print("Type 'exit' to quit\n")

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            print(traceback.format_exc())


if __name__ == "__main__":
    main()