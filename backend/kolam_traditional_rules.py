"""
Kolam Traditional Rule Engine

This module implements cultural authenticity validation and traditional rule enforcement
for kolam pattern recognition and recreation, ensuring compliance with historical,
ethnic, and regional authenticity standards while avoiding cultural appropriation.
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class KolamRegion(Enum):
    """Traditional kolam regions and styles."""
    TAMIL_NADU = "tamil_nadu"
    KERALA = "kerala"
    ANDHRA_PRADESH = "andhra_pradesh"
    KARNATAKA = "karnataka"
    TELANGANA = "telangana"
    NORTH_INDIAN = "north_indian"
    MODERN_FUSION = "modern_fusion"


class KolamType(Enum):
    """Types of kolam patterns."""
    PULLI_KOLAM = "pulli_kolam"  # Dot-based patterns
    NELLI_KOLAM = "nelli_kolam"  # Line-based patterns
    RANGOLI = "rangoli"  # Festival patterns
    MUGGU = "muggu"  # Floor decorations
    ALPANA = "alpana"  # Ritual patterns
    MANDANA = "mandana"  # Wall art
    CHOWK_PURANA = "chowk_purana"  # Sacred geometry


class AuthenticityLevel(Enum):
    """Cultural authenticity levels."""
    HIGHLY_AUTHENTIC = "highly_authentic"
    CULTURALLY_INSPIRED = "culturally_inspired"
    MODERN_INTERPRETATION = "modern_interpretation"
    CULTURALLY_INAPPROPRIATE = "culturally_inappropriate"


@dataclass
class CulturalRule:
    """Represents a traditional kolam rule."""
    rule_id: str
    name: str
    description: str
    region: KolamRegion
    kolam_type: KolamType
    mathematical_properties: Dict[str, any]
    cultural_significance: str
    authenticity_weight: float
    is_mandatory: bool = True


@dataclass
class PatternValidation:
    """Validation result for a kolam pattern."""
    is_authentic: bool
    authenticity_score: float
    authenticity_level: AuthenticityLevel
    violated_rules: List[str]
    cultural_warnings: List[str]
    suggested_modifications: List[str]
    region_match: Optional[KolamRegion]
    type_match: Optional[KolamType]
    validation_timestamp: str


class KolamTraditionalRules:
    """
    Traditional Rule Engine for kolam cultural authenticity validation.
    """

    def __init__(self):
        """Initialize the traditional rule engine."""
        self.cultural_rules = []
        self.region_hierarchies = {}
        self.mathematical_constraints = {}
        self.cultural_experts_db = {}

        # Load traditional rules
        self._load_traditional_rules()

        # Initialize cultural significance scoring
        self.cultural_weights = {
            'symmetry': 0.25,
            'proportions': 0.20,
            'motifs': 0.20,
            'regional_style': 0.15,
            'ritual_significance': 0.10,
            'material_compatibility': 0.10
        }

    def _load_traditional_rules(self):
        """Load comprehensive traditional kolam rules."""
        # Tamil Nadu Pulli Kolam Rules
        self.cultural_rules.extend([
            CulturalRule(
                rule_id="TN_PK_001",
                name="Dot Grid Foundation",
                description="Pulli kolam must start with a foundation of dots in specific mathematical ratios",
                region=KolamRegion.TAMIL_NADU,
                kolam_type=KolamType.PULLI_KOLAM,
                mathematical_properties={
                    'dot_pattern': 'geometric_progression',
                    'grid_symmetry': 'rotational_4_fold',
                    'dot_ratios': [1, 3, 5, 7, 9],
                    'line_continuity': 'unbroken_curves'
                },
                cultural_significance="Represents mathematical precision and devotion",
                authenticity_weight=0.9,
                is_mandatory=True
            ),

            CulturalRule(
                rule_id="TN_PK_002",
                name="Sacred Geometry Compliance",
                description="Patterns must incorporate sacred numbers (3, 5, 7, 9) and geometric forms",
                region=KolamRegion.TAMIL_NADU,
                kolam_type=KolamType.PULLI_KOLAM,
                mathematical_properties={
                    'sacred_numbers': [3, 5, 7, 9],
                    'geometric_forms': ['circle', 'square', 'lotus', 'conch'],
                    'fractal_dimension': 'between_1.2_1.8'
                },
                cultural_significance="Connects to cosmic harmony and divine proportions",
                authenticity_weight=0.85,
                is_mandatory=True
            ),

            # Kerala Nelli Kolam Rules
            CulturalRule(
                rule_id="KL_NK_001",
                name="Continuous Line Principle",
                description="Nelli kolam must be drawable with a single continuous line without lifting the finger",
                region=KolamRegion.KERALA,
                kolam_type=KolamType.NELLI_KOLAM,
                mathematical_properties={
                    'line_continuity': 'single_stroke',
                    'crossing_rule': 'no_self_intersection',
                    'flow_harmony': 'smooth_transitions'
                },
                cultural_significance="Symbolizes life's continuous journey and unity",
                authenticity_weight=0.95,
                is_mandatory=True
            ),

            # Regional Variation Rules
            CulturalRule(
                rule_id="REG_001",
                name="Regional Motif Authenticity",
                description="Motifs must be appropriate to the region and cultural context",
                region=KolamRegion.TAMIL_NADU,
                kolam_type=KolamType.RANGOLI,
                mathematical_properties={
                    'motif_categories': ['floral', 'geometric', 'animal', 'divine'],
                    'regional_restrictions': 'no_inappropriate_symbols'
                },
                cultural_significance="Maintains cultural identity and traditional symbolism",
                authenticity_weight=0.8,
                is_mandatory=True
            )
        ])

        # Mathematical constraints for different regions
        self.mathematical_constraints = {
            KolamRegion.TAMIL_NADU: {
                'symmetry_orders': [4, 8, 12],
                'preferred_ratios': [1, 1.414, 1.732],  # sqrt(2), sqrt(3)
                'dot_counts': [5, 7, 9, 13, 15, 25],
                'line_thickness_ratio': 0.1
            },
            KolamRegion.KERALA: {
                'flow_continuity': 'mandatory',
                'intersection_rules': 'no_crossing',
                'curve_smoothness': 'high',
                'line_weight': 'uniform'
            }
        }

    def validate_pattern_authenticity(self, pattern_features: Dict,
                                      region_context: Optional[KolamRegion] = None) -> PatternValidation:
        """
        Validate pattern authenticity against traditional rules.

        Args:
            pattern_features: Extracted pattern features and properties
            region_context: Cultural region context for validation

        Returns:
            Comprehensive validation result
        """
        logger.info(
            "Validating pattern authenticity against traditional rules...")

        violated_rules = []
        cultural_warnings = []
        suggested_modifications = []
        authenticity_score = 1.0

        # Check each cultural rule
        for rule in self.cultural_rules:
            rule_compliance = self._check_rule_compliance(
                rule, pattern_features)

            if not rule_compliance['compliant']:
                violated_rules.append(rule.rule_id)

                if rule.is_mandatory:
                    # Reduce authenticity score for mandatory violations
                    penalty = rule.authenticity_weight * 0.5
                    authenticity_score -= penalty

                    # Add specific suggestions for mandatory rules
                    suggested_modifications.extend(
                        self._generate_rule_suggestions(rule, pattern_features)
                    )
                else:
                    # Minor penalty for optional rule violations
                    authenticity_score -= rule.authenticity_weight * 0.2

            # Check for cultural warnings even if compliant
            warnings = self._check_cultural_warnings(rule, pattern_features)
            cultural_warnings.extend(warnings)

        # Determine authenticity level
        if authenticity_score >= 0.9:
            authenticity_level = AuthenticityLevel.HIGHLY_AUTHENTIC
        elif authenticity_score >= 0.7:
            authenticity_level = AuthenticityLevel.CULTURALLY_INSPIRED
        elif authenticity_score >= 0.5:
            authenticity_level = AuthenticityLevel.MODERN_INTERPRETATION
        else:
            authenticity_level = AuthenticityLevel.CULTURALLY_INAPPROPRIATE

        # Determine region and type matches
        region_match = self._determine_region_match(pattern_features)
        type_match = self._determine_type_match(pattern_features)

        validation_result = PatternValidation(
            is_authentic=authenticity_score >= 0.7,
            authenticity_score=max(0.0, authenticity_score),
            authenticity_level=authenticity_level,
            violated_rules=violated_rules,
            cultural_warnings=cultural_warnings,
            suggested_modifications=suggested_modifications,
            region_match=region_match,
            type_match=type_match,
            validation_timestamp=datetime.now().isoformat()
        )

        logger.info(
            f"Pattern validation completed. Score: {authenticity_score:.3f}")
        return validation_result

    def _check_rule_compliance(self, rule: CulturalRule, features: Dict) -> Dict:
        """Check compliance with a specific cultural rule."""
        compliant = True
        details = []

        # Check mathematical properties
        for prop, expected_value in rule.mathematical_properties.items():
            if prop in features:
                actual_value = features[prop]

                if not self._compare_mathematical_property(prop, actual_value, expected_value):
                    compliant = False
                    details.append(
                        f"Property '{prop}' mismatch: expected {expected_value}, got {actual_value}"
                    )
            else:
                # Missing required property
                if rule.is_mandatory:
                    compliant = False
                    details.append(f"Missing required property: {prop}")

        return {
            'compliant': compliant,
            'details': details
        }

    def _compare_mathematical_property(self, prop: str, actual: any, expected: any) -> bool:
        """Compare mathematical properties with tolerance for cultural variation."""
        if prop == 'dot_pattern':
            return actual == expected
        elif prop == 'grid_symmetry':
            return actual == expected
        elif prop == 'sacred_numbers':
            if isinstance(expected, list) and isinstance(actual, list):
                return any(num in actual for num in expected)
            return False
        elif prop == 'geometric_forms':
            if isinstance(expected, list) and isinstance(actual, list):
                return any(form in actual for form in expected)
            return False
        elif prop == 'line_continuity':
            return actual == expected
        elif prop == 'fractal_dimension':
            if isinstance(expected, str) and 'between_' in expected:
                try:
                    # Parse "between_X_Y" format
                    parts = expected.replace('between_', '').split('_')
                    if len(parts) >= 2:
                        min_val = float(parts[0])
                        max_val = float(parts[1])
                        if isinstance(actual, (int, float)):
                            return min_val <= actual <= max_val
                except (ValueError, IndexError):
                    logger.warning(
                        f"Invalid fractal_dimension format: {expected}")
                    return False
            elif isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
                return abs(actual - expected) < 0.1
            return False

        # Default to compliant for unknown properties
        return True

    def _check_cultural_warnings(self, rule: CulturalRule, features: Dict) -> List[str]:
        """Check for cultural warnings even if rule is technically compliant."""
        warnings = []

        # Check for regional appropriateness
        if rule.region == KolamRegion.TAMIL_NADU:
            if 'modern_motifs' in features and features['modern_motifs']:
                warnings.append(
                    "Modern motifs detected in traditional Tamil context")

        # Check for ritual appropriateness
        if 'ritual_context' in features:
            ritual_context = features['ritual_context']
            if ritual_context == 'wedding' and rule.kolam_type == KolamType.ALPANA:
                warnings.append(
                    "Alpana patterns may not be appropriate for wedding contexts")

        return warnings

    def _generate_rule_suggestions(self, rule: CulturalRule, features: Dict) -> List[str]:
        """Generate suggestions for rule compliance."""
        suggestions = []

        if 'dot_pattern' in rule.mathematical_properties:
            suggestions.append(
                f"Adjust dot pattern to follow {rule.mathematical_properties['dot_pattern']}"
            )

        if 'sacred_numbers' in rule.mathematical_properties:
            sacred_nums = rule.mathematical_properties['sacred_numbers']
            suggestions.append(
                f"Incorporate sacred numbers {sacred_nums} in the design")

        if 'geometric_forms' in rule.mathematical_properties:
            forms = rule.mathematical_properties['geometric_forms']
            suggestions.append(
                f"Include traditional geometric forms: {', '.join(forms)}")

        return suggestions

    def _determine_region_match(self, features: Dict) -> Optional[KolamRegion]:
        """Determine the most likely cultural region for the pattern."""
        region_scores = {}

        for region in KolamRegion:
            score = 0.0

            # Check against regional constraints
            if region in self.mathematical_constraints:
                constraints = self.mathematical_constraints[region]

                # Symmetry matching
                if 'symmetry_order' in features and 'symmetry_orders' in constraints:
                    if features['symmetry_order'] in constraints['symmetry_orders']:
                        score += 0.3

                # Dot count matching
                if 'dot_count' in features and 'dot_counts' in constraints:
                    if features['dot_count'] in constraints['dot_counts']:
                        score += 0.25

            region_scores[region] = score

        # Return region with highest score
        if region_scores:
            best_region = max(region_scores, key=region_scores.get)
            return best_region if region_scores[best_region] > 0.2 else None

        return None

    def _determine_type_match(self, features: Dict) -> Optional[KolamType]:
        """Determine the most likely kolam type."""
        type_scores = {}

        for kolam_type in KolamType:
            score = 0.0

            # Simple heuristic-based scoring
            if kolam_type == KolamType.PULLI_KOLAM and features.get('dot_based', False):
                score += 0.4
            elif kolam_type == KolamType.NELLI_KOLAM and features.get('line_based', False):
                score += 0.4
            elif kolam_type == KolamType.RANGOLI and features.get('colorful', False):
                score += 0.3

            type_scores[kolam_type] = score

        if type_scores:
            best_type = max(type_scores, key=type_scores.get)
            return best_type if type_scores[best_type] > 0.2 else None

        return None

    def generate_cultural_report(self, validation: PatternValidation,
                                 pattern_features: Dict) -> Dict:
        """
        Generate comprehensive cultural authenticity report.

        Args:
            validation: Pattern validation results
            pattern_features: Original pattern features

        Returns:
            Detailed cultural report
        """
        report = {
            'cultural_authenticity': {
                'is_authentic': validation.is_authentic,
                'authenticity_score': validation.authenticity_score,
                'authenticity_level': validation.authenticity_level.value,
                'cultural_region': validation.region_match.value if validation.region_match else None,
                'kolam_type': validation.type_match.value if validation.type_match else None
            },
            'validation_details': {
                'violated_rules': validation.violated_rules,
                'cultural_warnings': validation.cultural_warnings,
                'suggested_modifications': validation.suggested_modifications
            },
            'mathematical_properties': pattern_features,
            'cultural_significance': self._assess_cultural_significance(pattern_features),
            'traditional_compliance': self._check_traditional_compliance(pattern_features),
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'validation_engine_version': '2.0',
                'cultural_rules_version': '1.0'
            }
        }

        return report

    def _assess_cultural_significance(self, features: Dict) -> Dict:
        """Assess cultural significance of pattern elements."""
        significance = {
            'ritual_importance': 'medium',
            'regional_relevance': 'high',
            'traditional_value': 'high',
            'educational_potential': 'medium'
        }

        # Assess based on features
        if 'sacred_numbers' in features:
            if isinstance(features['sacred_numbers'], list) and len(features['sacred_numbers']) >= 2:
                significance['ritual_importance'] = 'high'

        if features.get('traditional_motifs', False):
            significance['traditional_value'] = 'very_high'

        return significance

    def _check_traditional_compliance(self, features: Dict) -> Dict:
        """Check compliance with traditional standards."""
        compliance = {
            'material_compatibility': True,
            'ritual_appropriateness': True,
            'regional_compatibility': True,
            'skill_level_appropriate': True
        }

        # Add specific compliance checks based on features
        if 'complexity_score' in features:
            complexity = features['complexity_score']
            if complexity > 0.9:
                compliance['skill_level_appropriate'] = 'Requires expert level'
            elif complexity > 0.6:
                compliance['skill_level_appropriate'] = 'Intermediate level'
            else:
                compliance['skill_level_appropriate'] = 'Beginner friendly'

        return compliance


class CulturalDomainExpert:
    """
    Interface for cultural domain experts to contribute to rule validation.
    """

    def __init__(self):
        """Initialize cultural domain expert system."""
        self.expert_rules = {}
        self.cultural_guidelines = {}
        self.regional_variations = {}

    def add_expert_rule(self, expert_id: str, rule: CulturalRule):
        """Add rule from cultural domain expert."""
        if expert_id not in self.expert_rules:
            self.expert_rules[expert_id] = []
        self.expert_rules[expert_id].append(rule)
        logger.info(f"Added expert rule from {expert_id}: {rule.name}")

    def validate_with_expert_insight(self, pattern_features: Dict,
                                     expert_context: Dict) -> Dict:
        """
        Validate pattern with expert cultural insight.

        Args:
            pattern_features: Pattern features to validate
            expert_context: Context from cultural experts

        Returns:
            Expert validation results
        """
        expert_validation = {
            'expert_approved': True,
            'expert_comments': [],
            'cultural_nuances': [],
            'suggested_refinements': []
        }

        # Apply expert rules
        for expert_id, rules in self.expert_rules.items():
            for rule in rules:
                if rule.region.value in expert_context.get('relevant_regions', []):
                    compliance = self._check_expert_rule(
                        rule, pattern_features)
                    if not compliance['approved']:
                        expert_validation['expert_approved'] = False
                        expert_validation['expert_comments'].append(
                            compliance['comment'])
                    if 'refinements' in compliance:
                        expert_validation['suggested_refinements'].extend(
                            compliance['refinements'])

        return expert_validation

    def _check_expert_rule(self, rule: CulturalRule, features: Dict) -> Dict:
        """Check compliance with expert rule."""
        # Expert-specific validation logic
        approved = True
        comment = f"Expert rule '{rule.name}' satisfied"
        refinements = []

        # Add expert-specific checks here
        if 'expert_mathematical_check' in rule.mathematical_properties:
            # Custom expert validation logic
            approved = self._perform_expert_mathematical_check(
                features, rule.mathematical_properties['expert_mathematical_check']
            )
            if not approved:
                comment = f"Expert rule '{rule.name}' not satisfied"
                refinements.append(f"Review {rule.name} requirements")

        return {
            'approved': approved,
            'comment': comment,
            'refinements': refinements
        }

    def _perform_expert_mathematical_check(self, features: Dict, check_config: any) -> bool:
        """Perform expert-level mathematical validation."""
        # Implement expert mathematical checks
        # This is a placeholder - actual implementation would depend on specific expert rules
        if isinstance(check_config, dict):
            for key, value in check_config.items():
                if key not in features:
                    return False
                # Add specific validation logic based on check_config
        return True


def main():
    """Demonstrate the traditional rule engine."""
    # Initialize rule engine
    rule_engine = KolamTraditionalRules()

    # Sample pattern features for testing
    sample_features = {
        'dot_pattern': 'geometric_progression',
        'grid_symmetry': 'rotational_4_fold',
        'sacred_numbers': [3, 5, 7],
        'geometric_forms': ['lotus', 'square'],
        'line_continuity': 'unbroken_curves',
        'fractal_dimension': 1.5,
        'symmetry_order': 4,
        'dot_count': 7,
        'dot_based': True,
        'complexity_score': 0.7
    }

    # Validate pattern
    validation = rule_engine.validate_pattern_authenticity(sample_features)

    # Generate cultural report
    report = rule_engine.generate_cultural_report(validation, sample_features)

    print("TRADITIONAL KOLAM RULE ENGINE DEMONSTRATION")
    print("="*60)
    print(f"Pattern Authenticity: {validation.is_authentic}")
    print(f"Authenticity Score: {validation.authenticity_score:.3f}")
    print(f"Authenticity Level: {validation.authenticity_level.value}")
    print(
        f"Region Match: {validation.region_match.value if validation.region_match else 'None'}")
    print(
        f"Type Match: {validation.type_match.value if validation.type_match else 'None'}")

    if validation.violated_rules:
        print(f"\nViolated Rules: {', '.join(validation.violated_rules)}")

    if validation.cultural_warnings:
        print("\nCultural Warnings:")
        for warning in validation.cultural_warnings:
            print(f"  - {warning}")

    if validation.suggested_modifications:
        print("\nSuggested Modifications:")
        for suggestion in validation.suggested_modifications:
            print(f"  - {suggestion}")

    print(f"\nCultural Report Generated: {len(report)} sections")
    print("\nCultural Significance Assessment:")
    for key, value in report['cultural_significance'].items():
        print(f"  - {key}: {value}")

    print("\nTraditional rule engine is ready for cultural authenticity validation!")


if __name__ == "__main__":
    main()
