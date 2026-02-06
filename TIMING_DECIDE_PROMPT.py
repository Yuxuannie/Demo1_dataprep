"""
Fallback DECIDE prompt for compatibility with existing workflows
This is used when the standard decide() method is called
"""

TIMING_DECIDE_PROMPT = """Based on your strategic analysis, make the final technical decisions.

STRATEGY SUMMARY:
{strategy_summary}

CLUSTERING COMPARISON:
{clustering_metrics}

DECISION REQUIRED:
Select the optimal clustering algorithm and parameters based on the analysis:

1. Algorithm Choice: K-means vs GMM
   - Consider data overlap patterns
   - Evaluate computational efficiency vs accuracy trade-offs

2. Cluster Count: Optimal number for this dataset
   - Balance between coverage and computational cost
   - Consider cell type diversity and feature complexity

3. Final Configuration: Specific parameters
   - Justify choices with quantitative reasoning

Provide your technical decision with brief justification.
Use plain text only."""