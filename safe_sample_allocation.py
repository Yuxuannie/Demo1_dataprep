"""
Safe Sample Allocation Logic with None-Type Protection
Fixes TypeError: unsupported operand type(s) for +: 'int' and 'NoneType'
"""

import numpy as np
from typing import Dict, List, Any, Optional, Union

def safe_sample_allocation(strategy_results: Dict[str, Any], total_target: int) -> Dict[str, int]:
    """
    Safely allocate samples across different strategies with None-type protection.

    Args:
        strategy_results: Dictionary containing allocation results from LLM strategy
        total_target: Total number of samples to allocate

    Returns:
        Dictionary with safe sample counts for each strategy
    """

    def safe_int(value: Union[int, float, str, None], default: int = 0) -> int:
        """Convert any value to safe integer, defaulting to 0 for None/invalid values."""
        if value is None:
            return default
        try:
            if isinstance(value, str):
                # Extract numbers from string responses like "30 samples" or "None allocated"
                import re
                numbers = re.findall(r'\d+', value)
                return int(numbers[0]) if numbers else default
            return max(0, int(float(value)))  # Ensure non-negative
        except (ValueError, TypeError):
            return default

    # Define all possible allocation strategies
    allocation_strategies = [
        'grid_sampling',
        'uncertainty_sampling',
        'boundary_sampling',
        'sparse_region_exploration',
        'validation_holdout',
        'representative_coverage',
        'corner_case_sampling'
    ]

    # Extract and safely convert all allocations
    safe_allocations = {}
    total_allocated = 0

    for strategy in allocation_strategies:
        raw_value = strategy_results.get(strategy, 0)
        safe_count = safe_int(raw_value, 0)
        safe_allocations[strategy] = safe_count
        total_allocated += safe_count

    # Handle over/under allocation
    if total_allocated > total_target:
        print(f"[WARNING] Over-allocation detected: {total_allocated} > {total_target}")
        # Proportionally reduce all non-zero allocations
        scale_factor = total_target / total_allocated
        for strategy in safe_allocations:
            if safe_allocations[strategy] > 0:
                safe_allocations[strategy] = max(1, int(safe_allocations[strategy] * scale_factor))

        # Recalculate total after scaling
        total_allocated = sum(safe_allocations.values())

    # Handle under-allocation by adding to largest strategy
    if total_allocated < total_target:
        remaining = total_target - total_allocated
        largest_strategy = max(safe_allocations.keys(), key=lambda k: safe_allocations[k])
        safe_allocations[largest_strategy] += remaining
        print(f"[INFO] Added {remaining} samples to {largest_strategy} to reach target")

    # Final validation
    final_total = sum(safe_allocations.values())
    assert final_total == total_target, f"Allocation error: {final_total} != {total_target}"

    return safe_allocations

def execute_agentic_sampling_with_safety(agent_response: str, target_count: int, dataset_size: int) -> Dict[str, Any]:
    """
    Execute sampling with comprehensive safety checks and error recovery.

    Args:
        agent_response: LLM response containing sampling strategy
        target_count: Number of samples to select
        dataset_size: Total dataset size

    Returns:
        Dictionary containing safe execution results
    """

    try:
        # Parse LLM response for allocation numbers
        strategy_allocations = parse_llm_strategy_response(agent_response)

        # Apply safety checks
        safe_allocations = safe_sample_allocation(strategy_allocations, target_count)

        # Execute each sampling strategy safely
        selected_indices = []
        execution_log = []

        for strategy, count in safe_allocations.items():
            if count > 0:
                try:
                    indices = execute_sampling_strategy(strategy, count, dataset_size)
                    selected_indices.extend(indices)
                    execution_log.append(f"✅ {strategy}: {len(indices)} samples")
                except Exception as e:
                    execution_log.append(f"❌ {strategy}: Failed ({e}) - Skipping {count} samples")
                    print(f"[ERROR] Strategy {strategy} failed: {e}")

        # Remove duplicates while preserving order
        unique_indices = list(dict.fromkeys(selected_indices))

        # Ensure we have enough samples (pad with random if necessary)
        if len(unique_indices) < target_count:
            remaining_needed = target_count - len(unique_indices)
            all_indices = set(range(dataset_size))
            available_indices = list(all_indices - set(unique_indices))

            if available_indices:
                additional_indices = np.random.choice(
                    available_indices,
                    min(remaining_needed, len(available_indices)),
                    replace=False
                )
                unique_indices.extend(additional_indices)
                execution_log.append(f"🔄 Random padding: {len(additional_indices)} samples")

        # Trim to exact target if over-selected
        if len(unique_indices) > target_count:
            unique_indices = unique_indices[:target_count]
            execution_log.append(f"✂️ Trimmed to target: {target_count} samples")

        return {
            'selected_indices': unique_indices[:target_count],
            'safe_allocations': safe_allocations,
            'execution_log': execution_log,
            'success': True,
            'final_count': len(unique_indices[:target_count])
        }

    except Exception as e:
        print(f"[CRITICAL ERROR] Sampling execution failed: {e}")
        # Emergency fallback: random sampling
        emergency_indices = np.random.choice(dataset_size, target_count, replace=False)
        return {
            'selected_indices': emergency_indices.tolist(),
            'safe_allocations': {'emergency_random': target_count},
            'execution_log': [f"🚨 Emergency random sampling: {target_count} samples"],
            'success': False,
            'error': str(e),
            'final_count': target_count
        }

def parse_llm_strategy_response(response: str) -> Dict[str, Any]:
    """Parse LLM response to extract sample allocation numbers."""
    import re

    allocations = {}

    # Common patterns for allocation parsing
    patterns = {
        'grid_sampling': r'grid[^:]*:\s*(\d+)|grid.*?(\d+)\s*samples',
        'uncertainty_sampling': r'uncertainty[^:]*:\s*(\d+)|uncertainty.*?(\d+)\s*samples',
        'boundary_sampling': r'boundary[^:]*:\s*(\d+)|boundary.*?(\d+)\s*samples',
        'sparse_region_exploration': r'sparse[^:]*:\s*(\d+)|sparse.*?(\d+)\s*samples',
        'validation_holdout': r'validation[^:]*:\s*(\d+)|holdout.*?(\d+)\s*samples',
        'representative_coverage': r'representative[^:]*:\s*(\d+)|coverage.*?(\d+)\s*samples'
    }

    for strategy, pattern in patterns.items():
        matches = re.findall(pattern, response, re.IGNORECASE)
        if matches:
            # Extract first non-empty match
            for match in matches:
                if isinstance(match, tuple):
                    number = next((m for m in match if m), None)
                else:
                    number = match
                if number:
                    allocations[strategy] = int(number)
                    break

    return allocations

def execute_sampling_strategy(strategy: str, count: int, dataset_size: int) -> List[int]:
    """Execute individual sampling strategy with error handling."""

    if count <= 0:
        return []

    # Ensure count doesn't exceed dataset size
    count = min(count, dataset_size)

    try:
        if strategy == 'grid_sampling':
            # Grid-based sampling
            step = max(1, dataset_size // count)
            return list(range(0, dataset_size, step))[:count]

        elif strategy == 'uncertainty_sampling':
            # High-uncertainty samples (simulated)
            np.random.seed(42)
            scores = np.random.random(dataset_size)
            return np.argsort(-scores)[:count].tolist()

        elif strategy == 'boundary_sampling':
            # Boundary/edge samples
            boundary_indices = list(range(count//2)) + list(range(dataset_size-count//2, dataset_size))
            return boundary_indices[:count]

        else:
            # Default: random sampling for unknown strategies
            return np.random.choice(dataset_size, count, replace=False).tolist()

    except Exception as e:
        print(f"[ERROR] Strategy {strategy} execution failed: {e}")
        return []