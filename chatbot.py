"""
Terminal-based Chatbot for Data Selection Agent
Pure CLI interface - no Flask, no markdown symbols
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment
load_dotenv()

from agent.data_selection_agent import DataSelectionAgent
from agent.llm_config import initialize_ollama_llm, test_ollama_connection


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
        
        # Remove quotes if present
        file_path = file_path.strip('"\'')
        
        if not Path(file_path).exists():
            print(f"Error: File not found: {file_path}")
            continue
        
        if not file_path.endswith('.csv'):
            print("Error: File must be CSV format")
            continue
        
        # Load and verify
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
    """Main chatbot loop."""
    
    print_header("DATA SELECTION AGENT - Terminal Chatbot")
    
    # Check Ollama connection
    print("Checking Ollama connection...")
    if not test_ollama_connection():
        print("\nError: Cannot connect to Ollama")
        print(f"Base URL: {os.getenv('OLLAMA_BASE_URL')}")
        print(f"Make sure Ollama is running: ollama serve")
        sys.exit(1)
    
    print("Success: Ollama connected")
    
    # Initialize LLM
    print("\nInitializing LLM...")
    try:
        llm = initialize_ollama_llm()
        print("Success: LLM initialized")
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        sys.exit(1)
    
    # Initialize agent
    print("Initializing agent...")
    try:
        agent = DataSelectionAgent(llm=llm, verbose=False)
        print("Success: Agent initialized\n")
    except Exception as e:
        print(f"Error initializing agent: {e}")
        sys.exit(1)
    
    # Get CSV file
    print_section("UPLOAD CSV FILE")
    csv_path = upload_csv()
    
    # Chat loop
    print_section("CHAT WITH AGENT")
    print("Type your selection request (e.g., 'Select 8% of training data')")
    print("Type 'exit' to quit\n")
    
    while True:
        # Get user input
        user_input = input("You: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() == 'exit':
            print("\nGoodbye!")
            break
        
        # Process request
        print("\n" + "Processing... (this may take 30-60 seconds)\n")
        
        try:
            # Run agent
            results = agent.run_selection(user_input, csv_path)
            
            # Display results
            print_section("SELECTION COMPLETE")
            
            print(f"Total Samples: {results['observation']['total_samples']}")
            print(f"Selected Samples: {results['result']['n_selected']}")
            print(f"Selection Percentage: {results['parsed_params']['selection_percentage']:.1f}%")
            print(f"Number of Clusters: {results['decision']['clustering']['n_clusters']}")
            print(f"PCA Components: {results['decision']['pca']['n_components']}")
            print(f"PCA Variance Explained: {results['decision']['pca']['variance_explained']*100:.1f}%")
            print(f"Clustering Algorithm: {results['decision']['clustering']['algorithm'].upper()}")
            
            # Show reasoning stages
            print_section("REASONING STAGES")
            
            for log_entry in results['reasoning_log']:
                stage = log_entry['stage']
                content = log_entry['content']
                print(f"\n{stage}:")
                print(content)
            
            # Ask for download
            print_section("SAVE RESULTS")
            save_option = input("Save selected samples to CSV? (yes/no): ").strip().lower()
            
            if save_option in ['yes', 'y']:
                output_path = f"selected_samples_{results['observation']['total_samples']}_{results['result']['n_selected']}.csv"
                results['result']['selected_df'].to_csv(output_path, index=False)
                print(f"Saved to: {output_path}")
            
            # Ask to continue
            print()
            continue_option = input("Run another selection? (yes/no): ").strip().lower()
            if continue_option not in ['yes', 'y']:
                print("\nGoodbye!")
                break
            
            print_section("CHAT WITH AGENT")
            print("Type your next selection request")
            print("Type 'exit' to quit\n")
        
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            print(traceback.format_exc())


if __name__ == "__main__":
    main()
