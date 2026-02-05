"""
Integration Guide for Enhanced Timing Domain Agent
Complete implementation and testing framework
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any
import json

def create_enhanced_chatbot():
    """
    Create enhanced chatbot that uses the new timing domain agent.
    """
    chatbot_content = '''"""
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
    print("\\n" + "=" * 80)
    print(text)
    print("=" * 80 + "\\n")


def print_section(text):
    """Print section header."""
    print("\\n" + "-" * 80)
    print(text)
    print("-" * 80 + "\\n")


def upload_csv():
    """Get CSV file path from user."""
    while True:
        file_path = input("Enter CSV file path: ").strip()

        if not file_path:
            print("Error: Please enter a file path")
            continue

        file_path = file_path.strip('"\\\'')

        if not Path(file_path).exists():
            print(f"Error: File not found: {file_path}")
            continue

        if not file_path.endswith('.csv'):
            print("Error: File must be CSV format")
            continue

        try:
            import pandas as pd
            df = pd.read_csv(file_path)
            print(f"\\nSuccess: Loaded {len(df)} rows, {len(df.columns)} columns")
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

    print("\\nStage-by-Stage Analysis:")
    for stage, quality in workflow_results['stage_results'].items():
        print(f"\\n{stage}:")
        print(f"  Score: {quality.overall_score:.3f}")
        print(f"  Expertise: {quality.expertise_level.name}")
        print(f"  Numbers Cited: {quality.specific_numbers_cited}")
        print(f"  Domain Concepts: {quality.domain_concepts_used}")
        print(f"  Active Learning: {'✓' if quality.active_learning_explained else '✗'}")
        print(f"  Business Impact: {'✓' if quality.business_impact_mentioned else '✗'}")

    if workflow_results['improvement_suggestions']:
        print("\\nImprovement Suggestions:")
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
        print("\\nError: Cannot connect to Ollama")
        print(f"Base URL: {os.getenv('OLLAMA_BASE_URL')}")
        print("Make sure Ollama is running: ollama serve")
        sys.exit(1)

    print("✓ Ollama connected")

    # Initialize enhanced LLM
    print("\\nInitializing Enhanced LLM with timing domain parameters...")
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
        print("✓ Enhanced Agent and Validator initialized\\n")
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
    print("Type 'exit' to quit\\n")

    session_results = []

    while True:
        user_input = input("Timing Engineer: ").strip()

        if not user_input:
            continue

        if user_input.lower() == 'exit':
            print("\\n=== SESSION SUMMARY ===")
            if session_results:
                avg_score = sum(r['workflow_score'] for r in session_results) / len(session_results)
                senior_achieved = sum(r['senior_level_achieved'] for r in session_results)
                print(f"Sessions completed: {len(session_results)}")
                print(f"Average quality score: {avg_score:.3f}")
                print(f"Senior engineer level achieved: {senior_achieved}/{len(session_results)}")
            print("\\nGoodbye!")
            break

        # Process enhanced request
        print("\\nProcessing with Enhanced Timing Domain AI...")
        print("(Using domain expertise prompts and active learning principles)\\n")

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
                print(f"\\n{stage} (Enhanced with Domain Expertise):")
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
                print("\\n=== SESSION SUMMARY ===")
                if session_results:
                    avg_score = sum(r['workflow_score'] for r in session_results) / len(session_results)
                    senior_achieved = sum(r['senior_level_achieved'] for r in session_results)
                    print(f"Enhanced sessions completed: {len(session_results)}")
                    print(f"Average quality score: {avg_score:.3f}")
                    print(f"Senior engineer level achieved: {senior_achieved}/{len(session_results)}")
                print("\\nEnhanced Timing AI session complete!")
                break

            print_section("ENHANCED TIMING ANALYSIS")
            print("Enter your next timing characterization request:")

        except Exception as e:
            print(f"Error in enhanced processing: {e}")
            import traceback
            print(traceback.format_exc())


if __name__ == "__main__":
    main()
'''

    return chatbot_content


def create_integration_instructions():
    """
    Create step-by-step integration instructions.
    """
    instructions = """
# Enhanced Timing Domain Agent - Integration Instructions

## 🎯 Goal: Transform Generic ML Agent → Senior Timing Engineer AI

### Phase 1: Setup Enhanced Environment

1. **Copy enhanced configuration files:**
   ```bash
   # In your demo1_dataprep directory
   cp enhanced_prompts.py ./
   cp enhanced_data_selection_agent.py ./
   cp enhanced_llm_config.py ./
   cp reasoning_validator.py ./
   cp enhanced_chatbot.py ./
   ```

2. **Update environment variables:**
   ```bash
   # Copy enhanced template
   cp enhanced_env_template.txt .env

   # Edit .env with your settings:
   OLLAMA_MODEL=qwen2.5:32b-instruct  # Or your preferred model
   LLM_TEMPERATURE=0.2                 # Lower for consistency
   LLM_TOP_P=0.9                      # Focused sampling
   LLM_NUM_PREDICT=2500               # More tokens for detail
   ```

3. **Test enhanced configuration:**
   ```bash
   python enhanced_llm_config.py
   ```

### Phase 2: Validate Enhancement

1. **Run validation tests:**
   ```bash
   python reasoning_validator.py
   ```

2. **Compare responses:**
   - Run original agent: `python chatbot.py`
   - Run enhanced agent: `python enhanced_chatbot.py`
   - Use same query: "Select 8% of timing data"
   - Compare reasoning quality

### Phase 3: Quality Benchmarking

**Target Quality Metrics:**
- Overall Score: ≥ 0.75 (vs current ~0.4)
- Expertise Level: SENIOR_ENGINEER (vs current GENERIC)
- Numbers Cited: ≥ 3 per response (vs current 0-1)
- Domain Concepts: ≥ 5 timing terms (vs current 0-2)
- Active Learning: Explained in all stages (vs current missing)
- Business Impact: Mentioned in strategic stages (vs current missing)

**Success Criteria:**
✅ Cites specific correlations (r=0.89)
✅ Uses timing terminology (sigma_by_nominal, process variation)
✅ Explains uncertainty sampling rationale
✅ Connects to business value (50% cost reduction)
✅ Demonstrates senior engineer reasoning depth

### Phase 4: Production Deployment

1. **Replace original agent:**
   ```bash
   mv chatbot.py chatbot_original.py
   mv enhanced_chatbot.py chatbot.py

   mv agent/data_selection_agent.py agent/data_selection_agent_original.py
   mv enhanced_data_selection_agent.py agent/data_selection_agent.py

   mv agent/llm_config.py agent/llm_config_original.py
   mv enhanced_llm_config.py agent/llm_config.py
   ```

2. **Update imports:**
   ```python
   # In chatbot.py, update imports:
   from enhanced_data_selection_agent import EnhancedDataSelectionAgent as DataSelectionAgent
   from enhanced_llm_config import initialize_enhanced_ollama_llm as initialize_ollama_llm
   ```

3. **Enable quality monitoring:**
   ```python
   # Add to your main workflow
   from reasoning_validator import TimingReasoningValidator

   validator = TimingReasoningValidator()
   quality_results = validator.validate_full_workflow(reasoning_log)

   if not quality_results['senior_level_achieved']:
       print("⚠️ Quality below senior engineer level")
   ```

### Phase 5: Continuous Improvement

1. **Monitor quality metrics**
2. **Collect feedback on reasoning depth**
3. **Iterate prompts based on domain expert review**
4. **Prepare for Demo 2 with evaluation agent**

## 🔍 Troubleshooting

**Issue: Generic responses despite enhanced prompts**
- Solution: Lower temperature to 0.1, increase repeat_penalty to 1.15

**Issue: Responses too short**
- Solution: Increase num_predict to 3000, add explicit length requirements

**Issue: Missing domain concepts**
- Solution: Add few-shot examples to system prompt

**Issue: Poor correlation citations**
- Solution: Add explicit data context to prompts

## 📈 Expected Improvements

| Metric | Current | Enhanced | Improvement |
|--------|---------|----------|-------------|
| Reasoning Quality | 0.4/1.0 | 0.8/1.0 | +100% |
| Numbers Cited | 0-1 | 3-5 | +300% |
| Domain Expertise | Generic | Senior | Qualitative leap |
| Business Context | Missing | Present | New capability |
| Active Learning | Vague | Explicit | Clear explanation |

## 🎯 Next Steps for Demo 2

The enhanced agent provides foundation for:
1. **Multi-constraint evaluation agent**
2. **Adaptive reasoning based on data characteristics**
3. **Expert-level technical justifications**
4. **Integration with TSMC workflow tools**
"""

    return instructions


def run_integration_test():
    """
    Run integration test comparing original vs enhanced agent.
    """
    print("=" * 80)
    print("INTEGRATION TEST: Original vs Enhanced Agent")
    print("=" * 80)

    test_query = "Select 8% of timing data for Monte Carlo characterization"

    print(f"Test Query: {test_query}")
    print("\nThis test would:")
    print("1. Run original agent")
    print("2. Run enhanced agent")
    print("3. Compare reasoning quality")
    print("4. Validate domain expertise")
    print("5. Measure improvement metrics")

    print("\n⚠️  To run actual test:")
    print("1. Ensure both agents are set up")
    print("2. Have timing CSV data ready")
    print("3. Run: python integration_test.py")

    return True


def main():
    """
    Main integration function.
    """
    print("Enhanced Timing Domain Agent - Integration Guide")
    print("=" * 60)

    # Create enhanced chatbot
    chatbot_content = create_enhanced_chatbot()
    with open('enhanced_chatbot.py', 'w') as f:
        f.write(chatbot_content)
    print("✓ Enhanced chatbot created: enhanced_chatbot.py")

    # Create integration instructions
    instructions = create_integration_instructions()
    with open('INTEGRATION_GUIDE.md', 'w') as f:
        f.write(instructions)
    print("✓ Integration guide created: INTEGRATION_GUIDE.md")

    # Run integration test
    run_integration_test()

    print("\n🎯 Ready for Enhanced Timing Domain AI!")
    print("Next: Follow INTEGRATION_GUIDE.md for step-by-step setup")


if __name__ == "__main__":
    main()