"""
Response Validation and Quality Assessment for Enhanced Timing Agent
Ensures LLM responses demonstrate senior timing engineer expertise
"""

import re
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from enum import Enum


class ExpertiseLevel(Enum):
    GENERIC = 1
    INTERMEDIATE = 2
    SENIOR_ENGINEER = 3
    EXPERT = 4


@dataclass
class ReasoningQuality:
    """Quality assessment metrics for LLM reasoning."""
    expertise_level: ExpertiseLevel
    specific_numbers_cited: int
    domain_concepts_used: int
    business_impact_mentioned: bool
    active_learning_explained: bool
    technical_justification_quality: float  # 0-1 score
    overall_score: float  # 0-1 score
    feedback: List[str]


class TimingReasoningValidator:
    """
    Validates LLM responses for timing domain expertise.

    Checks for:
    - Specific number citations (correlation=0.89 vs "high")
    - Timing domain terminology
    - Active learning concepts
    - Business impact awareness
    - Technical depth vs generic advice
    """

    def __init__(self):
        # Timing domain vocabulary
        self.timing_concepts = [
            'process variation', 'monte carlo', 'timing signoff', 'sigma_by_nominal',
            'lib_sigma', 'nominal_delay', 'setup time', 'hold time', 'skew',
            'process corner', 'pvt', 'cell characterization', 'timing arc',
            'delay variability', 'library development', 'silicon failure',
            'tape-out', 'characterization cost', 'edge case', 'boundary condition'
        ]

        # Active learning vocabulary
        self.active_learning_concepts = [
            'uncertainty sampling', 'active learning', 'model uncertainty',
            'boundary cases', 'high uncertainty', 'far from centroid',
            'uncertain samples', 'edge conditions', 'model confidence',
            'uncertainty score', 'training robustness'
        ]

        # Business impact vocabulary
        self.business_concepts = [
            'cost reduction', 'cost savings', '50% reduction', '10% to 5%',
            'characterization cost', 'silicon failure', 'tape-out delay',
            'signoff accuracy', 'business value', 'risk reduction'
        ]

        # Generic ML phrases to penalize
        self.generic_phrases = [
            'as a best practice', 'generally speaking', 'it is important',
            'machine learning', 'data science', 'typically used',
            'commonly applied', 'standard approach', 'usual method'
        ]

    def validate_observe_response(self, response: str, data_stats: Dict[str, Any]) -> ReasoningQuality:
        """Validate OBSERVE stage response."""
        return self._validate_response(
            response,
            stage='OBSERVE',
            expected_numbers=['correlation', 'samples', 'features'],
            required_concepts=['process variation', 'delay', 'variability'],
            data_context=data_stats
        )

    def validate_think_response(self, response: str, context: Dict[str, Any]) -> ReasoningQuality:
        """Validate THINK stage response."""
        return self._validate_response(
            response,
            stage='THINK',
            expected_numbers=['percentage', 'samples'],
            required_concepts=['uncertainty sampling', 'clustering', 'active learning'],
            data_context=context
        )

    def validate_decide_response(self, response: str, metrics: Dict[str, Any]) -> ReasoningQuality:
        """Validate DECIDE stage response."""
        return self._validate_response(
            response,
            stage='DECIDE',
            expected_numbers=['bic', 'clusters', 'variance'],
            required_concepts=['gmm', 'overlap', 'timing'],
            data_context=metrics
        )

    def validate_act_response(self, response: str, results: Dict[str, Any]) -> ReasoningQuality:
        """Validate ACT stage response."""
        return self._validate_response(
            response,
            stage='ACT',
            expected_numbers=['selected', 'uncertainty', 'cost'],
            required_concepts=['uncertainty sampling', 'signoff', 'business impact'],
            data_context=results
        )

    def _validate_response(
        self,
        response: str,
        stage: str,
        expected_numbers: List[str],
        required_concepts: List[str],
        data_context: Dict[str, Any]
    ) -> ReasoningQuality:
        """Core validation logic."""

        feedback = []
        response_lower = response.lower()

        # 1. Check for specific number citations
        specific_numbers = self._count_specific_numbers(response)
        if specific_numbers < 2:
            feedback.append(f"❌ Need more specific numbers (found {specific_numbers}, expected ≥2)")
        else:
            feedback.append(f"✅ Good number citations ({specific_numbers} found)")

        # 2. Check timing domain concepts
        timing_score = self._score_domain_concepts(response_lower, self.timing_concepts)
        if timing_score < 3:
            feedback.append(f"❌ Insufficient timing domain expertise (score: {timing_score}/10)")
        else:
            feedback.append(f"✅ Good timing domain knowledge (score: {timing_score}/10)")

        # 3. Check active learning concepts
        active_learning_present = self._check_active_learning(response_lower)
        if not active_learning_present:
            feedback.append("❌ Missing active learning explanation")
        else:
            feedback.append("✅ Active learning principle explained")

        # 4. Check business impact
        business_impact = self._check_business_impact(response_lower)
        if not business_impact:
            feedback.append("❌ No business impact mentioned")
        else:
            feedback.append("✅ Business value articulated")

        # 5. Penalize generic phrases
        generic_penalty = self._count_generic_phrases(response_lower)
        if generic_penalty > 0:
            feedback.append(f"⚠️ Generic phrases detected ({generic_penalty} instances)")

        # 6. Check technical justification quality
        tech_quality = self._assess_technical_depth(response, stage)
        if tech_quality < 0.6:
            feedback.append(f"❌ Insufficient technical depth (score: {tech_quality:.2f})")
        else:
            feedback.append(f"✅ Good technical reasoning (score: {tech_quality:.2f})")

        # Calculate overall score
        overall_score = (
            min(specific_numbers / 3, 1.0) * 0.25 +  # Numbers
            min(timing_score / 5, 1.0) * 0.25 +      # Domain knowledge
            (1.0 if active_learning_present else 0.0) * 0.20 +  # Active learning
            (1.0 if business_impact else 0.0) * 0.15 +           # Business
            tech_quality * 0.15 -                                # Technical depth
            (generic_penalty * 0.02)                             # Generic penalty
        )

        overall_score = max(0.0, min(1.0, overall_score))

        # Determine expertise level
        if overall_score >= 0.85:
            expertise = ExpertiseLevel.EXPERT
        elif overall_score >= 0.70:
            expertise = ExpertiseLevel.SENIOR_ENGINEER
        elif overall_score >= 0.50:
            expertise = ExpertiseLevel.INTERMEDIATE
        else:
            expertise = ExpertiseLevel.GENERIC

        return ReasoningQuality(
            expertise_level=expertise,
            specific_numbers_cited=specific_numbers,
            domain_concepts_used=timing_score,
            business_impact_mentioned=business_impact,
            active_learning_explained=active_learning_present,
            technical_justification_quality=tech_quality,
            overall_score=overall_score,
            feedback=feedback
        )

    def _count_specific_numbers(self, response: str) -> int:
        """Count specific numerical citations."""
        patterns = [
            r'r\s*=\s*0\.\d+',           # correlation=0.89
            r'correlation\s*=\s*0\.\d+',  # correlation=0.89
            r'\d+\.\d+%',                 # 85.3%
            r'\d+\,\d+',                  # 21,817
            r'bic\s*=\s*\d+',            # BIC=1234
            r'k\s*=\s*\d+',              # k=10
            r'\d+\s*samples',             # 1745 samples
            r'\d+\s*components',          # 3 components
            r'variance.*\d+\.\d+%'        # variance 93.9%
        ]

        count = 0
        for pattern in patterns:
            matches = re.findall(pattern, response.lower())
            count += len(matches)

        return count

    def _score_domain_concepts(self, response: str, concepts: List[str]) -> int:
        """Score domain concept usage."""
        score = 0
        for concept in concepts:
            if concept in response:
                score += 1
        return score

    def _check_active_learning(self, response: str) -> bool:
        """Check for active learning explanation."""
        return any(concept in response for concept in self.active_learning_concepts)

    def _check_business_impact(self, response: str) -> bool:
        """Check for business impact awareness."""
        return any(concept in response for concept in self.business_concepts)

    def _count_generic_phrases(self, response: str) -> int:
        """Count generic ML phrases."""
        count = 0
        for phrase in self.generic_phrases:
            count += response.count(phrase)
        return count

    def _assess_technical_depth(self, response: str, stage: str) -> float:
        """Assess technical depth based on stage."""
        depth_indicators = {
            'OBSERVE': [
                'correlation coefficient', 'standard deviation', 'percentile',
                'distribution', 'variance ratio', 'process corner'
            ],
            'THINK': [
                'overlapping distributions', 'cluster boundary', 'model uncertainty',
                'representativeness', 'edge case coverage'
            ],
            'DECIDE': [
                'bic comparison', 'model selection', 'algorithm justification',
                'parameter optimization', 'metric interpretation'
            ],
            'ACT': [
                'selection strategy', 'uncertainty ranking', 'sample diversity',
                'coverage analysis', 'robustness improvement'
            ]
        }

        indicators = depth_indicators.get(stage, [])
        found = sum(1 for indicator in indicators if indicator in response.lower())

        return min(found / len(indicators), 1.0) if indicators else 0.5

    def generate_improvement_suggestions(self, quality: ReasoningQuality, stage: str) -> List[str]:
        """Generate specific improvement suggestions."""
        suggestions = []

        if quality.specific_numbers_cited < 3:
            suggestions.append(
                f"Cite specific numbers: correlation coefficients, exact sample counts, "
                f"percentage values, BIC scores, variance percentages"
            )

        if quality.domain_concepts_used < 3:
            suggestions.append(
                f"Use more timing terminology: process variation, timing signoff, "
                f"sigma_by_nominal, characterization cost, boundary conditions"
            )

        if not quality.active_learning_explained:
            suggestions.append(
                f"Explain active learning: why uncertainty sampling works, "
                f"why samples far from centroids improve model robustness"
            )

        if not quality.business_impact_mentioned:
            suggestions.append(
                f"Connect to business value: 50% cost reduction, improved signoff accuracy, "
                f"reduced silicon failure risk"
            )

        if quality.technical_justification_quality < 0.7:
            suggestions.append(
                f"Provide deeper technical justification: algorithm trade-offs, "
                f"why GMM handles timing overlaps, uncertainty ranking rationale"
            )

        return suggestions

    def validate_full_workflow(self, reasoning_log: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate complete reasoning workflow."""
        stage_results = {}
        overall_feedback = []

        for entry in reasoning_log:
            stage = entry['stage']
            content = entry['content']

            if stage == 'OBSERVE':
                quality = self.validate_observe_response(content, {})
            elif stage == 'THINK':
                quality = self.validate_think_response(content, {})
            elif stage == 'DECIDE':
                quality = self.validate_decide_response(content, {})
            elif stage == 'ACT':
                quality = self.validate_act_response(content, {})
            else:
                continue

            stage_results[stage] = quality

            # Add stage-specific feedback
            if quality.overall_score < 0.7:
                suggestions = self.generate_improvement_suggestions(quality, stage)
                overall_feedback.extend([f"{stage}: {s}" for s in suggestions])

        # Calculate workflow score
        if stage_results:
            workflow_score = sum(q.overall_score for q in stage_results.values()) / len(stage_results)
        else:
            workflow_score = 0.0

        # Determine if agent demonstrates senior engineer level
        senior_level_achieved = (
            workflow_score >= 0.75 and
            all(q.expertise_level in [ExpertiseLevel.SENIOR_ENGINEER, ExpertiseLevel.EXPERT]
                for q in stage_results.values())
        )

        return {
            'workflow_score': workflow_score,
            'senior_level_achieved': senior_level_achieved,
            'stage_results': stage_results,
            'improvement_suggestions': overall_feedback,
            'summary': {
                'avg_numbers_cited': sum(q.specific_numbers_cited for q in stage_results.values()) / len(stage_results) if stage_results else 0,
                'avg_domain_concepts': sum(q.domain_concepts_used for q in stage_results.values()) / len(stage_results) if stage_results else 0,
                'active_learning_coverage': sum(q.active_learning_explained for q in stage_results.values()),
                'business_impact_coverage': sum(q.business_impact_mentioned for q in stage_results.values())
            }
        }


def test_enhanced_prompts():
    """Test enhanced prompts with sample responses."""
    validator = TimingReasoningValidator()

    # Sample responses for testing
    generic_response = """
    The dataset has many samples and features. High correlations were found.
    Clustering is a good approach for data selection. We should use machine learning
    best practices to select representative samples.
    """

    enhanced_response = """
    Analysis of 21,817 timing arc samples reveals strong delay-variability correlation
    (r=0.89) between nominal_delay and lib_sigma_delay_late, confirming process variation
    scaling. The sigma_by_nominal range 0.02-0.15 indicates mix of stable and high-variation
    paths critical for timing signoff. PCA to 3 components preserves 93.9% variance while
    eliminating redundancy. Uncertainty sampling targeting samples far from centroids
    captures boundary conditions where model uncertainty is highest, essential for robust
    timing characterization. This approach enables 5% Monte Carlo coverage vs current 10%,
    delivering 50% characterization cost reduction while maintaining signoff accuracy.
    """

    print("=== Testing Generic Response ===")
    generic_quality = validator.validate_observe_response(generic_response, {})
    print(f"Score: {generic_quality.overall_score:.3f}")
    print(f"Level: {generic_quality.expertise_level.name}")
    for feedback in generic_quality.feedback:
        print(feedback)

    print("\n=== Testing Enhanced Response ===")
    enhanced_quality = validator.validate_observe_response(enhanced_response, {})
    print(f"Score: {enhanced_quality.overall_score:.3f}")
    print(f"Level: {enhanced_quality.expertise_level.name}")
    for feedback in enhanced_quality.feedback:
        print(feedback)


if __name__ == "__main__":
    test_enhanced_prompts()