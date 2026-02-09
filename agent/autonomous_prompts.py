"""
Autonomous Prompt System for Self-Directed Sampling Agent

This module provides dynamic, context-aware prompts that adapt to the agent's
discoveries and enable truly autonomous exploration without predetermined steps.
"""

from typing import Dict, List, Any, Optional
import json


class AutonomousPromptGenerator:
    """Generates dynamic prompts that adapt to agent's discoveries and context."""

    def __init__(self):
        self.discovery_patterns = {}
        self.failed_approaches = []
        self.successful_patterns = []

    def generate_exploration_prompt(self, context: Dict[str, Any]) -> str:
        """Generate autonomous exploration prompt based on current context."""

        iteration = context.get('iteration', 1)
        knowledge_gaps = context.get('knowledge_gaps', [])
        discoveries = context.get('discoveries', {})
        data_size = context.get('data_size', 'unknown')

        base_prompt = f"""You are an autonomous sampling agent exploring a timing dataset.

CURRENT SITUATION:
- Dataset size: {data_size} samples
- Iteration: {iteration}
- What I know: {self._summarize_discoveries(discoveries)}
- Knowledge gaps: {knowledge_gaps}
- Failed approaches so far: {self.failed_approaches}

YOUR TASK: Decide what to investigate next to improve sampling strategy.

THINK: What specific aspect of this data should I explore to make better sampling decisions?
- What patterns might exist that I haven't discovered?
- What data characteristics would change my sampling approach?
- What assumptions should I validate?

ACTION: Write Python code to investigate your chosen focus area.
- Be specific and targeted in your investigation
- Generate insights that will directly impact sampling decisions
- Use any analysis technique you think will be valuable

Generate ONLY the Python code you want to execute."""

        # Add adaptive guidance based on iteration and context
        if iteration == 1:
            base_prompt += "\n\nFOCUS SUGGESTION: Start with basic data understanding - shape, features, distributions."
        elif iteration <= 3 and not discoveries.get('schema_complete'):
            base_prompt += "\n\nFOCUS SUGGESTION: Complete schema understanding before moving to advanced analysis."
        elif discoveries.get('high_variance_features'):
            base_prompt += f"\n\nFOCUS SUGGESTION: Investigate high-variance features: {discoveries['high_variance_features']}"
        elif iteration > 10:
            base_prompt += "\n\nFOCUS SUGGESTION: You've explored extensively. Focus on synthesizing insights for sampling strategy."

        return base_prompt

    def generate_hypothesis_prompt(self, observations: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate prompt for autonomous hypothesis formation."""

        return f"""You are forming hypotheses about optimal sampling strategies.

OBSERVATIONS FROM YOUR EXPLORATION:
{self._format_observations(observations)}

CURRENT KNOWLEDGE BASE:
{self._summarize_discoveries(context.get('discoveries', {}))}

YOUR TASK: Generate testable hypotheses about sampling strategies.

THINK: Based on what you've observed, what sampling approaches might work best?
- What patterns in the data suggest certain sampling methods?
- What are your theories about which samples would be most informative?
- What specific predictions can you make about algorithm performance?

Generate 3-5 specific, testable hypotheses in this format:
HYPOTHESIS 1: [Your theory about optimal sampling approach]
PREDICTION: [Specific, measurable prediction]
TEST: [How you would test this hypothesis]

HYPOTHESIS 2: [Next theory]
...and so on."""

    def generate_experiment_design_prompt(self, hypothesis: str, available_algorithms: List[str], context: Dict[str, Any]) -> str:
        """Generate prompt for autonomous experiment design."""

        return f"""You are designing an experiment to test a specific hypothesis.

HYPOTHESIS TO TEST: {hypothesis}

AVAILABLE RESOURCES:
- Algorithms: {available_algorithms}
- Previous results: {context.get('previous_results', [])}
- Current best performance: {context.get('best_performance', 'unknown')}

YOUR TASK: Design a specific experiment to test this hypothesis.

THINK: How can I test this hypothesis rigorously?
- Which algorithm(s) should I use and why?
- What parameters should I test?
- How will I measure success?
- What would prove or disprove this hypothesis?

DESIGN: Create a specific experiment plan:
- Algorithm choice and justification
- Parameter settings
- Success metrics
- Expected outcomes

Provide your experiment design as a structured plan."""

    def generate_validation_prompt(self, results: Dict[str, Any], boundaries: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate prompt for autonomous result validation."""

        return f"""You are validating experimental results against quality standards.

EXPERIMENT RESULTS:
{self._format_results(results)}

QUALITY BOUNDARIES:
{boundaries}

VALIDATION HISTORY:
- Previous failures: {context.get('validation_failures', [])}
- Success patterns: {self.successful_patterns}

YOUR TASK: Evaluate these results and decide next steps.

THINK: Do these results meet quality standards? Why or why not?
- Which boundaries are satisfied/violated?
- How do these results compare to previous attempts?
- What patterns do you notice?
- What would you change to improve results?

DECISION: Based on your analysis:
- PASS/FAIL assessment with reasoning
- Specific improvements needed if failed
- Next experimental direction
- What you learned from this attempt

Provide your validation assessment and next steps."""

    def generate_strategy_synthesis_prompt(self, all_results: List[Dict], knowledge_base: Dict, context: Dict[str, Any]) -> str:
        """Generate prompt for autonomous strategy synthesis."""

        return f"""You are synthesizing your exploration into a final sampling strategy.

ALL EXPLORATION RESULTS:
{self._format_all_results(all_results)}

KNOWLEDGE GAINED:
{self._summarize_discoveries(knowledge_base)}

PERFORMANCE PATTERNS:
- Best performing approaches: {context.get('top_performers', [])}
- Failed approaches: {self.failed_approaches}
- Learned patterns: {self.successful_patterns}

YOUR TASK: Create the optimal sampling strategy for this specific dataset.

THINK: What is the best sampling approach given everything you've learned?
- Which algorithm performed best and why?
- What parameters should you use?
- How should you allocate samples?
- What are the key insights that drive this strategy?

SYNTHESIZE: Create a comprehensive sampling strategy:
- Chosen algorithm with full justification
- Optimal parameters based on your experiments
- Sample allocation strategy
- Expected performance and why
- Confidence level in this approach

Provide a complete, evidence-based sampling strategy."""

    def generate_recovery_prompt(self, failure: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate prompt for autonomous failure recovery."""

        return f"""You encountered a failure that needs autonomous recovery.

FAILURE DETAILS:
{failure}

CONTEXT:
- Current iteration: {context.get('iteration', 'unknown')}
- Previous attempts: {context.get('attempts', [])}
- Available alternatives: {context.get('alternatives', [])}

YOUR TASK: Autonomously recover from this failure.

THINK: Why did this approach fail? What can you learn?
- What went wrong specifically?
- What assumptions were incorrect?
- What would you do differently?
- Are there alternative approaches to try?

RECOVER: Design a recovery strategy:
- Root cause analysis
- Alternative approach
- Modified parameters or methods
- How to avoid this failure in future

Generate your recovery plan."""

    def _summarize_discoveries(self, discoveries: Dict[str, Any]) -> str:
        """Summarize discoveries in a readable format."""
        if not discoveries:
            return "No significant discoveries yet"

        summary_parts = []
        for key, value in discoveries.items():
            if isinstance(value, (list, dict)):
                summary_parts.append(f"- {key}: {len(value) if isinstance(value, list) else 'complex structure'}")
            else:
                summary_parts.append(f"- {key}: {value}")

        return "\n".join(summary_parts)

    def _format_observations(self, observations: Dict[str, Any]) -> str:
        """Format observations for prompt inclusion."""
        formatted = []
        for key, value in observations.items():
            if isinstance(value, dict):
                formatted.append(f"{key}: {json.dumps(value, indent=2)}")
            elif isinstance(value, list) and len(value) > 5:
                formatted.append(f"{key}: {len(value)} items (showing first 5: {value[:5]})")
            else:
                formatted.append(f"{key}: {value}")

        return "\n".join(formatted)

    def _format_results(self, results: Dict[str, Any]) -> str:
        """Format experimental results for prompt inclusion."""
        formatted = []
        for key, value in results.items():
            if key in ['silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score']:
                formatted.append(f"- {key}: {value:.4f}")
            elif isinstance(value, float):
                formatted.append(f"- {key}: {value:.3f}")
            else:
                formatted.append(f"- {key}: {value}")

        return "\n".join(formatted)

    def _format_all_results(self, all_results: List[Dict]) -> str:
        """Format all exploration results for synthesis."""
        if not all_results:
            return "No results available"

        summary = f"Total experiments: {len(all_results)}\n\n"

        # Group by algorithm
        by_algorithm = {}
        for result in all_results:
            alg = result.get('algorithm', 'unknown')
            if alg not in by_algorithm:
                by_algorithm[alg] = []
            by_algorithm[alg].append(result)

        for algorithm, results in by_algorithm.items():
            summary += f"{algorithm}:\n"
            for result in results[:3]:  # Show top 3 per algorithm
                score = result.get('composite_score', result.get('silhouette_score', 'unknown'))
                summary += f"  - Score: {score}, Config: {result.get('variant', 'default')}\n"
            summary += "\n"

        return summary

    def update_patterns(self, results: Dict[str, Any], success: bool):
        """Update learned patterns based on results."""
        if success:
            pattern = {
                'algorithm': results.get('algorithm'),
                'parameters': results.get('parameters', {}),
                'performance': results.get('performance', {})
            }
            self.successful_patterns.append(pattern)
        else:
            failure = {
                'approach': results.get('algorithm'),
                'reason': results.get('failure_reason', 'unknown'),
                'parameters': results.get('parameters', {})
            }
            self.failed_approaches.append(failure)


class ContextualPromptAdaptor:
    """Adapts prompts based on agent's learning history and current context."""

    def __init__(self):
        self.learning_history = []
        self.context_patterns = {}

    def adapt_prompt_for_context(self, base_prompt: str, context: Dict[str, Any]) -> str:
        """Adapt base prompt based on current context and learning history."""

        # Add learned context adaptations
        adaptations = []

        # Data size adaptations
        data_size = context.get('data_size', 0)
        if data_size > 100000:
            adaptations.append("NOTE: Large dataset - prioritize scalable algorithms.")
        elif data_size < 1000:
            adaptations.append("NOTE: Small dataset - be careful of overfitting.")

        # Feature dimensionality adaptations
        n_features = context.get('n_features', 0)
        if n_features > 50:
            adaptations.append("NOTE: High-dimensional data - consider dimensionality reduction.")

        # Learning history adaptations
        if len(self.learning_history) > 10:
            recent_failures = [h for h in self.learning_history[-5:] if not h.get('success')]
            if len(recent_failures) >= 3:
                adaptations.append("NOTE: Recent failures detected - try fundamentally different approaches.")

        # Add adaptations to prompt
        if adaptations:
            adapted_prompt = base_prompt + "\n\nCONTEXT AWARENESS:\n" + "\n".join(adaptations)
            return adapted_prompt

        return base_prompt

    def record_learning(self, prompt_type: str, context: Dict[str, Any], outcome: Dict[str, Any]):
        """Record learning from prompt usage for future adaptation."""
        learning_record = {
            'prompt_type': prompt_type,
            'context': context,
            'outcome': outcome,
            'success': outcome.get('success', False)
        }
        self.learning_history.append(learning_record)

        # Maintain history size
        if len(self.learning_history) > 100:
            self.learning_history = self.learning_history[-50:]


# Global instances for use by the agent
autonomous_prompt_generator = AutonomousPromptGenerator()
contextual_adaptor = ContextualPromptAdaptor()


def get_autonomous_exploration_prompt(context: Dict[str, Any]) -> str:
    """Get context-adapted exploration prompt."""
    base_prompt = autonomous_prompt_generator.generate_exploration_prompt(context)
    return contextual_adaptor.adapt_prompt_for_context(base_prompt, context)


def get_autonomous_hypothesis_prompt(observations: Dict[str, Any], context: Dict[str, Any]) -> str:
    """Get context-adapted hypothesis generation prompt."""
    base_prompt = autonomous_prompt_generator.generate_hypothesis_prompt(observations, context)
    return contextual_adaptor.adapt_prompt_for_context(base_prompt, context)


def get_autonomous_experiment_prompt(hypothesis: str, algorithms: List[str], context: Dict[str, Any]) -> str:
    """Get context-adapted experiment design prompt."""
    base_prompt = autonomous_prompt_generator.generate_experiment_design_prompt(hypothesis, algorithms, context)
    return contextual_adaptor.adapt_prompt_for_context(base_prompt, context)


def get_autonomous_validation_prompt(results: Dict[str, Any], boundaries: Dict[str, Any], context: Dict[str, Any]) -> str:
    """Get context-adapted validation prompt."""
    base_prompt = autonomous_prompt_generator.generate_validation_prompt(results, boundaries, context)
    return contextual_adaptor.adapt_prompt_for_context(base_prompt, context)


def get_autonomous_synthesis_prompt(all_results: List[Dict], knowledge_base: Dict, context: Dict[str, Any]) -> str:
    """Get context-adapted strategy synthesis prompt."""
    base_prompt = autonomous_prompt_generator.generate_strategy_synthesis_prompt(all_results, knowledge_base, context)
    return contextual_adaptor.adapt_prompt_for_context(base_prompt, context)


def get_autonomous_recovery_prompt(failure: Dict[str, Any], context: Dict[str, Any]) -> str:
    """Get context-adapted recovery prompt."""
    base_prompt = autonomous_prompt_generator.generate_recovery_prompt(failure, context)
    return contextual_adaptor.adapt_prompt_for_context(base_prompt, context)


def update_prompt_learning(prompt_type: str, context: Dict[str, Any], outcome: Dict[str, Any]):
    """Update prompt system learning from outcomes."""
    autonomous_prompt_generator.update_patterns(outcome, outcome.get('success', False))
    contextual_adaptor.record_learning(prompt_type, context, outcome)