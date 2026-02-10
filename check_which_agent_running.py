"""
Check Which Agent Is Actually Running
This will determine if the UI is using real or fake intelligence
"""

def check_streamlit_agent_class():
    """Check what the Streamlit app is actually importing"""
    print("🔍 CHECKING STREAMLIT AGENT IMPORT")
    print("=" * 50)

    # Simulate the same imports as app_ui.py
    try:
        from agent.real_timing_agent import RealTimingDataSelectionAgent as TimingDataSelectionAgent
        print(f"✅ Successfully imported RealTimingDataSelectionAgent")
        print(f"   Class: {TimingDataSelectionAgent}")
        print(f"   Module: {TimingDataSelectionAgent.__module__}")

        # Check methods
        agent = TimingDataSelectionAgent(llm=None, verbose=False)

        # Check if it has fake methods
        fake_methods = [
            '_generate_timing_specific_thought',
            '_generate_timing_specific_action'
        ]

        print(f"\n🎭 Checking for fake methods:")
        for method in fake_methods:
            if hasattr(agent, method):
                print(f"   ❌ FAKE METHOD FOUND: {method}")
            else:
                print(f"   ✅ No fake method: {method}")

        # Check if it has real methods
        real_methods = [
            'exploration_engine',
            'evidence_sampler'
        ]

        print(f"\n🧠 Checking for real components:")
        for component in real_methods:
            if hasattr(agent, component):
                print(f"   ✅ Real component: {component}")
            else:
                print(f"   ❌ Missing component: {component}")

        # Test with LLM parameter to see if behavior changes
        print(f"\n🔬 Testing with LLM parameter...")

        class MockLLM:
            def invoke(self, prompt):
                return "Mock LLM response"

        agent_with_llm = TimingDataSelectionAgent(llm=MockLLM(), verbose=False)

        # Check if having an LLM changes the agent behavior
        print(f"   Agent with LLM class: {type(agent_with_llm).__name__}")

        return TimingDataSelectionAgent

    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return None


def check_old_agent_still_exists():
    """Check if the old fake agent is still being used somewhere"""
    print(f"\n🕵️ CHECKING OLD AGENT EXISTENCE")
    print("=" * 50)

    try:
        # Try importing the old agent
        from agent.timing_data_selection_agent import TimingDataSelectionAgent as OldAgent
        print(f"⚠ OLD AGENT STILL EXISTS")
        print(f"   Class: {OldAgent}")
        print(f"   Module: {OldAgent.__module__}")

        # Check if it has fake methods
        old_agent = OldAgent(llm=None, verbose=False)

        fake_methods = [
            '_generate_timing_specific_thought',
            '_generate_timing_specific_action'
        ]

        print(f"\n🎭 Old agent fake methods:")
        for method in fake_methods:
            if hasattr(old_agent, method):
                print(f"   ❌ FAKE METHOD EXISTS: {method}")
                # Try calling it to see what it returns
                try:
                    if method == '_generate_timing_specific_thought':
                        result = old_agent._generate_timing_specific_thought()
                        print(f"      Returns: '{result[:50]}...'")
                except Exception as e:
                    print(f"      Error calling: {e}")
            else:
                print(f"   ✅ Method removed: {method}")

    except ImportError:
        print(f"✅ Old agent import failed - good, it's not accessible")


def trace_actual_execution():
    """Trace which methods are actually being called"""
    print(f"\n🔍 TRACING ACTUAL EXECUTION")
    print("=" * 50)

    from agent.real_timing_agent import RealTimingDataSelectionAgent

    # Create test data
    import pandas as pd
    import numpy as np
    import os

    np.random.seed(999)
    data = np.random.normal([5, 5], 1, (50, 2))
    df = pd.DataFrame(data, columns=['x', 'y'])

    os.makedirs('/tmp/claude', exist_ok=True)
    csv_path = '/tmp/claude/trace_test.csv'
    df.to_csv(csv_path, index=False)

    agent = RealTimingDataSelectionAgent(llm=None, verbose=True)

    # Monkey patch to trace method calls
    original_classify = agent.classify_user_intent
    original_run_selection = agent.run_selection

    def traced_classify(*args, **kwargs):
        print(f"   📞 CALLED: classify_user_intent with args: {args[:1]}")
        result = original_classify(*args, **kwargs)
        print(f"   📤 RETURNED: {result}")
        return result

    def traced_run_selection(*args, **kwargs):
        print(f"   📞 CALLED: run_selection with args: {args[:1]}")
        result = original_run_selection(*args, **kwargs)
        method_used = result.get('selection_method', 'unknown') if isinstance(result, dict) else 'error'
        print(f"   📤 RETURNED: method={method_used}")
        return result

    agent.classify_user_intent = traced_classify
    agent.run_selection = traced_run_selection

    # Test the agent
    print(f"\n🧪 Testing agent execution:")
    try:
        result = agent.run_selection("Select 10% of samples", csv_path)

        # Check what was actually returned
        if isinstance(result, dict):
            keys = list(result.keys())
            print(f"   Result keys: {keys}")

            if 'data_analysis' in result:
                print(f"   ✅ Has data_analysis - likely real intelligence")
            else:
                print(f"   ⚠ No data_analysis - may be fake")

            if 'reasoning_chain' in result:
                reasoning = result['reasoning_chain']
                print(f"   Reasoning: {reasoning}")

        else:
            print(f"   ❌ Unexpected result type: {type(result)}")

    except Exception as e:
        print(f"   ❌ Execution failed: {e}")


def main():
    """Run all checks"""
    print("🔍 CHECKING WHICH AGENT IS ACTUALLY RUNNING")
    print("Investigating if the UI uses real or fake intelligence")
    print("")

    agent_class = check_streamlit_agent_class()
    check_old_agent_still_exists()

    if agent_class:
        trace_actual_execution()

    print(f"\n" + "=" * 60)
    print(f"🎯 INVESTIGATION COMPLETE")
    print(f"Check the results above to see which agent is really running")


if __name__ == "__main__":
    main()