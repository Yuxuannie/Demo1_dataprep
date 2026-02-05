"""
Quick test script to validate timing agent fixes
"""

import os
from pathlib import Path

def test_imports():
    """Test that all imports work correctly."""
    print("Testing imports...")
    try:
        from agent.timing_data_selection_agent import TimingDataSelectionAgent
        from agent.timing_llm_config import initialize_timing_llm, test_ollama_connection
        from timing_prompts import TIMING_SYSTEM_PROMPT
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False

def test_prompt_formatting():
    """Test that the DECIDE prompt formats correctly."""
    print("Testing DECIDE prompt formatting...")
    try:
        from timing_prompts import TIMING_DECIDE_PROMPT

        # Test with sample data
        test_prompt = TIMING_DECIDE_PROMPT.format(
            original_features=13,
            pca_components=7,
            variance_explained=94.3,
            assessment='Excellent',
            clustering_metrics='k=8: K-means inertia=96003, GMM BIC=292521'
        )

        print("✓ DECIDE prompt formats correctly")
        print(f"Prompt length: {len(test_prompt)} characters")
        return True
    except Exception as e:
        print(f"✗ Prompt formatting error: {e}")
        return False

def test_data_file():
    """Test that mock data file exists."""
    print("Testing mock data file...")
    data_path = Path("mock_data/test_data.csv")
    if data_path.exists():
        print("✓ Mock data file found")
        try:
            import pandas as pd
            df = pd.read_csv(data_path)
            print(f"✓ Data loaded: {len(df)} rows, {len(df.columns)} columns")
            return True
        except Exception as e:
            print(f"✗ Data loading error: {e}")
            return False
    else:
        print(f"✗ Mock data file not found at: {data_path}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("TIMING AGENT VALIDATION TESTS")
    print("=" * 60)

    tests = [
        ("Import Test", test_imports),
        ("Prompt Formatting Test", test_prompt_formatting),
        ("Data File Test", test_data_file)
    ]

    results = []
    for name, test_func in tests:
        print(f"\n{name}:")
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append(False)

    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)

    passed = sum(results)
    total = len(results)

    for i, (name, _) in enumerate(tests):
        status = "PASS" if results[i] else "FAIL"
        print(f"{name}: {status}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("\n✓ All tests passed! Agent should work correctly.")
        print("Try running: python chatbot.py")
    else:
        print(f"\n✗ {total-passed} tests failed. Please fix issues before running agent.")

if __name__ == "__main__":
    main()