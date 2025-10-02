"""
Kolam Comprehensive Documentation Generator

This module generates exhaustive construction guides, material lists, scaling options,
and assembly diagrams for real-world fabrication or digital rendering of kolam patterns.
"""

import json
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
import logging
from datetime import datetime
import os
from jinja2 import Template, Environment, FileSystemLoader
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ConstructionStep:
    """Individual construction step."""
    step_number: int
    title: str
    description: str
    tools_required: List[str]
    materials_needed: List[str]
    time_estimate: str
    difficulty_rating: int  # 1-5 scale
    quality_checks: List[str]
    common_mistakes: List[str]
    tips_and_tricks: List[str]


@dataclass
class MaterialRequirement:
    """Material requirement specification."""
    material_name: str
    quantity: str
    unit: str
    cost_per_unit: float
    source_suggestions: List[str]
    alternatives: List[str]
    storage_notes: str


@dataclass
class DocumentationPackage:
    """Complete documentation package."""
    pattern_id: str
    pattern_name: str
    construction_guide: Dict
    material_requirements: Dict
    scaling_options: Dict
    cultural_context: Dict
    technical_specifications: Dict
    quality_standards: Dict
    troubleshooting_guide: Dict
    generation_metadata: Dict


class KolamDocumentationGenerator:
    """
    Comprehensive documentation generator for kolam patterns.
    """

    def __init__(self):
        """Initialize the documentation generator."""
        self.templates_dir = "backend/documentation_templates"
        self.output_dir = "backend/generated_documentation"

        # Ensure directories exist
        os.makedirs(self.templates_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize Jinja2 environment
        self.jinja_env = Environment(
            loader=FileSystemLoader(self.templates_dir))

    def generate_complete_documentation(self, pattern: np.ndarray,
                                      pattern_features: Dict,
                                      cultural_validation: Dict,
                                      output_formats: List[str] = None) -> DocumentationPackage:
        """
        Generate complete documentation package for kolam pattern.

        Args:
            pattern: Kolam pattern image
            pattern_features: Extracted pattern features
            cultural_validation: Cultural authenticity validation results
            output_formats: List of documentation formats to generate

        Returns:
            Complete documentation package
        """
        logger.info("Generating comprehensive documentation package...")

        if output_formats is None:
            output_formats = ['json', 'markdown', 'html', 'pdf']

        # Generate core documentation components
        construction_guide = self._generate_construction_guide(
            pattern, pattern_features)
        material_requirements = self._generate_material_requirements(
            pattern_features)
        scaling_options = self._generate_scaling_options()
        cultural_context = self._generate_cultural_context(cultural_validation)
        technical_specs = self._generate_technical_specifications(
            pattern, pattern_features)
        quality_standards = self._generate_quality_standards()
        troubleshooting = self._generate_troubleshooting_guide(
            pattern_features)

        # Create documentation package
        package = DocumentationPackage(
            pattern_id=f"kolam_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            pattern_name=pattern_features.get(
                'pattern_type', 'Traditional Kolam'),
            construction_guide=construction_guide,
            material_requirements=material_requirements,
            scaling_options=scaling_options,
            cultural_context=cultural_context,
            technical_specifications=technical_specs,
            quality_standards=quality_standards,
            troubleshooting_guide=troubleshooting,
            generation_metadata={
                'generated_at': datetime.now().isoformat(),
                'generator_version': '3.0',
                'pattern_complexity': pattern_features.get('pattern_complexity_score', 0.5),
                'cultural_authenticity': cultural_validation.get('authenticity_score', 0.0)
            }
        )

        # Generate output files in specified formats
        generated_files = self._generate_output_files(package, output_formats)

        logger.info(
            f"Documentation package generated with {len(generated_files)} formats")
        return package

    def _generate_construction_guide(self, pattern: np.ndarray, features: Dict) -> Dict:
        """Generate detailed step-by-step construction guide."""
        complexity = features.get('pattern_complexity_score', 0.5)
        symmetry_count = len(features.get('dominant_symmetries', []))

        # Determine construction approach based on complexity
        if complexity < 0.3:
            approach = "basic_grid_method"
        elif complexity < 0.7:
            approach = "intermediate_dot_method"
        else:
            approach = "advanced_mathematical_method"

        return {
            'construction_approach': approach,
            'estimated_completion_time': self._estimate_completion_time(complexity),
            'skill_level_required': self._determine_skill_level(complexity, symmetry_count),
            'steps': self._generate_detailed_steps(pattern, features, approach),
            'prerequisites': [
                'Clean, flat surface',
                'Good lighting conditions',
                'Basic measuring tools',
                'Quality rice flour or colored powder'
            ],
            'safety_considerations': [
                'Work in well-ventilated area',
                'Use food-grade materials only',
                'Take breaks to avoid fatigue',
                'Document progress with photos'
            ]
        }

    def _generate_detailed_steps(self, pattern: np.ndarray, features: Dict, approach: str) -> List[Dict]:
        """Generate detailed construction steps."""
        steps = []

        # Step 1: Surface Preparation
        steps.append({
            'step_number': 1,
            'title': 'Surface Preparation',
            'description': 'Prepare the surface for kolam creation',
            'tools_required': ['broom', 'water', 'cleaning_cloth'],
            'materials_needed': ['none'],
            'time_estimate': '5 minutes',
            'difficulty_rating': 1,
            'quality_checks': ['Surface is clean and dry', 'Area is well-lit'],
            'common_mistakes': ['Working on uneven surface', 'Insufficient cleaning'],
            'tips_and_tricks': ['Work early morning for best results', 'Ensure surface is completely dry']
        })

        # Step 2: Grid Construction
        steps.append({
            'step_number': 2,
            'title': 'Grid Construction',
            'description': 'Create the foundation grid for accurate pattern placement',
            'tools_required': ['measuring_tape', 'chalk', 'straight_edge', 'compass'],
            'materials_needed': ['chalk_powder'],
            'time_estimate': '10 minutes',
            'difficulty_rating': 2,
            'quality_checks': ['Grid lines are straight', 'Measurements are accurate', 'Center point is clearly marked'],
            'common_mistakes': ['Uneven grid spacing', 'Incorrect center point'],
            'tips_and_tricks': ['Use string for long straight lines', 'Double-check all measurements']
        })

        # Step 3-6: Pattern Drawing (varies by complexity)
        base_steps = 3
        complexity = features.get('pattern_complexity_score', 0.5)
        if complexity > 0.7:
            base_steps = 6
        elif complexity > 0.4:
            base_steps = 4

        for i in range(base_steps):
            steps.append({
                'step_number': 3 + i,
                'title': f'Pattern Drawing - Phase {i + 1}',
                'description': f'Draw section {i + 1} of the pattern',
                'tools_required': ['fine_sieve', 'straight_edge'],
                'materials_needed': ['rice_flour', 'colored_powder'],
                'time_estimate': f'{15 + i*10} minutes',
                'difficulty_rating': min(5, 2 + i),
                'quality_checks': ['Lines are continuous', 'Proportions are maintained', 'Symmetry is preserved'],
                'common_mistakes': ['Inconsistent line thickness', 'Broken flour flow'],
                'tips_and_tricks': ['Maintain consistent hand pressure', 'Work systematically from center outward']
            })

        # Final Step: Finishing
        steps.append({
            'step_number': len(steps) + 1,
            'title': 'Finishing and Documentation',
            'description': 'Complete the pattern and document the results',
            'tools_required': ['camera', 'notebook'],
            'materials_needed': ['none'],
            'time_estimate': '5 minutes',
            'difficulty_rating': 1,
            'quality_checks': ['Pattern is complete and symmetrical', 'All lines are connected', 'Cultural authenticity is maintained'],
            'common_mistakes': ['Rushing the final touches', 'Not documenting the process'],
            'tips_and_tricks': ['Take photos from multiple angles', 'Note any modifications made']
        })

        return steps

    def _generate_material_requirements(self, features: Dict) -> Dict:
        """Generate detailed material requirements."""
        complexity = features.get('pattern_complexity_score', 0.5)
        pattern_size = features.get('estimated_size', 'medium')

        # Base material calculations
        base_flour_grams = self._calculate_base_flour_requirement(
            pattern_size, complexity)

        return {
            'primary_materials': [
                asdict(MaterialRequirement(
                    material_name='Rice Flour (fine quality)',
                    quantity=f"{base_flour_grams} grams",
                    unit='grams',
                    cost_per_unit=0.5,
                    source_suggestions=['Local grocery store',
                        'Traditional markets', 'Online suppliers'],
                    alternatives=[
                        'All-purpose flour (for practice)', 'Colored rice flour'],
                    storage_notes='Store in airtight container in cool, dry place'
                )),
                asdict(MaterialRequirement(
                    material_name='Colored Powder (optional)',
                    quantity=f"{base_flour_grams * 0.1} grams",
                    unit='grams',
                    cost_per_unit=2.0,
                    source_suggestions=['Art supply stores',
                        'Traditional craft suppliers'],
                    alternatives=['Natural food coloring', 'Turmeric powder'],
                    storage_notes='Keep colors separate to avoid mixing'
                ))
            ],
            'tools_and_equipment': [
                {
                    'tool_name': 'Fine Mesh Sieve',
                    'purpose': 'Even flour distribution',
                    'cost_estimate': '₹200-500',
                    'source': 'Kitchenware stores'
                },
                {
                    'tool_name': 'Measuring Tape',
                    'purpose': 'Accurate measurements',
                    'cost_estimate': '₹50-200',
                    'source': 'Hardware stores'
                },
                {
                    'tool_name': 'Straight Edge (1-2 meter)',
                    'purpose': 'Drawing straight lines',
                    'cost_estimate': '₹100-300',
                    'source': 'Art supply stores'
                }
            ],
            'total_estimated_cost': self._calculate_total_cost(base_flour_grams, complexity),
            'material_preparation_instructions': [
                'Sift rice flour to remove lumps',
                'Mix colored powder thoroughly if using',
                'Prepare tools and clean work area',
                'Test flour flow on small area first'
            ]
        }

    def _generate_scaling_options(self) -> Dict:
        """Generate scaling options for different applications."""
        return {
            'micro_decorative': {
                'scale_factor': 0.25,
                'dimensions': '15cm x 15cm',
                'use_case': 'Photo frames, small decorations',
                'time_estimate': '10 minutes',
                'material_estimate': '25 grams flour',
                'difficulty': 'Very Easy'
            },
            'small_portable': {
                'scale_factor': 0.5,
                'dimensions': '30cm x 30cm',
                'use_case': 'Table decorations, small events',
                'time_estimate': '20 minutes',
                'material_estimate': '50 grams flour',
                'difficulty': 'Easy'
            },
            'medium_residential': {
                'scale_factor': 1.0,
                'dimensions': '1m x 1m',
                'use_case': 'Home entrance, daily rituals',
                'time_estimate': '45 minutes',
                'material_estimate': '200 grams flour',
                'difficulty': 'Intermediate'
            },
            'large_ceremonial': {
                'scale_factor': 2.0,
                'dimensions': '2m x 2m',
                'use_case': 'Festivals, special ceremonies',
                'time_estimate': '2 hours',
                'material_estimate': '800 grams flour',
                'difficulty': 'Advanced'
            },
            'extra_large_community': {
                'scale_factor': 3.0,
                'dimensions': '3m x 3m',
                'use_case': 'Community events, temples',
                'time_estimate': '4+ hours',
                'material_estimate': '2 kg flour',
                'difficulty': 'Expert'
            }
        }

    def _generate_cultural_context(self, cultural_validation: Dict) -> Dict:
        """Generate cultural context and significance information."""
        authenticity_score = cultural_validation.get('authenticity_score', 0.0)

        return {
            'cultural_significance': {
                'primary_meaning': 'Welcome and prosperity symbol',
                'ritual_importance': 'Daily devotion and meditation',
                'social_function': 'Community bonding and cultural continuity',
                'spiritual_symbolism': 'Connection between art and mathematics'
            },
            'regional_variations': {
                'tamil_nadu': 'Focus on mathematical precision and sacred numbers',
                'kerala': 'Emphasis on continuous line drawing',
                'andhra_pradesh': 'Incorporation of local floral motifs',
                'karnataka': 'Integration of temple architectural elements'
            },
            'historical_context': {
                'origins': 'Ancient tradition dating back 5000+ years',
                'evolution': 'From simple dot patterns to complex artistic expressions',
                'cultural_preservation': 'Maintained through oral traditions and family teachings',
                'modern_adaptation': 'Digital tools enhancing traditional techniques'
            },
            'authenticity_assessment': {
                'validation_score': authenticity_score,
                'cultural_compliance': 'high' if authenticity_score > 0.8 else 'moderate',
                'traditional_accuracy': 'verified' if authenticity_score > 0.9 else 'approximate',
                'expert_review_status': 'pending'  # Would be updated by cultural experts
            }
        }

    def _generate_technical_specifications(self, pattern: np.ndarray, features: Dict) -> Dict:
        """Generate technical specifications for the pattern."""
        height, width = pattern.shape

        return {
            'pattern_dimensions': {
                'pixel_dimensions': f"{width} x {height}",
                'recommended_physical_size': f"{width * 10}mm x {height * 10}mm",
                'aspect_ratio': width / height,
                'total_area': f"{width * height * 100} mm²"
            },
            'pattern_metrics': {
                'line_count': self._count_pattern_lines(pattern),
                'intersection_points': self._count_intersection_points(pattern),
                'curve_complexity': features.get('curvature_measure', 0.5),
                'symmetry_order': len(features.get('dominant_symmetries', []))
            },
            'mathematical_properties': {
                'fractal_dimension': features.get('mathematical_properties', {}).get('fractal_dimension', 1.0),
                'lacunarity': features.get('mathematical_properties', {}).get('lacunarity', 1.0),
                'correlation_dimension': features.get('mathematical_properties', {}).get('correlation_dimension', 1.0),
                'complexity_score': features.get('pattern_complexity_score', 0.5)
            },
            'construction_parameters': {
                'grid_spacing': 20,  # mm
                'line_thickness': '2-3mm',
                'dot_size': '5mm diameter',
                'flour_density': '1.5 kg/m²'
            }
        }

    def _generate_quality_standards(self) -> Dict:
        """Generate quality standards and best practices."""
        return {
            'accuracy_standards': {
                'measurement_tolerance': '±2mm',
                'angle_tolerance': '±2 degrees',
                'line_thickness_consistency': '±0.5mm',
                'symmetry_deviation': '±1mm'
            },
            'material_standards': {
                'flour_quality': 'Fine ground, no lumps',
                'color_fastness': 'Fade-resistant for outdoor use',
                'moisture_content': '<5% for longevity',
                'purity': 'Food-grade materials only'
            },
            'construction_standards': {
                'line_continuity': 'No breaks or gaps',
                'edge_smoothness': 'Clean, defined edges',
                'proportion_accuracy': 'Golden ratio compliance',
                'cultural_authenticity': 'Traditional method adherence'
            },
            'documentation_standards': {
                'progress_photos': 'Required at each major step',
                'measurement_records': 'All dimensions documented',
                'material_tracking': 'Batch numbers and sources recorded',
                'completion_certificate': 'Self-assessment checklist completed'
            }
        }

    def _generate_troubleshooting_guide(self, features: Dict) -> Dict:
        """Generate troubleshooting guide for common issues."""
        complexity = features.get('pattern_complexity_score', 0.5)

        return {
            'common_problems': [
                {
                    'problem': 'Uneven line thickness',
                    'cause': 'Inconsistent hand pressure or flour flow',
                    'solution': 'Use steady, even pressure; practice on scrap surface first',
                    'prevention': 'Regular sieve cleaning and consistent flour quality'
                },
                {
                    'problem': 'Pattern asymmetry',
                    'cause': 'Incorrect center point or measurement errors',
                    'solution': 'Mark center clearly; use measuring tools for all points',
                    'prevention': 'Double-check all measurements before starting pattern'
                },
                {
                    'problem': 'Flour clumping',
                    'cause': 'Moisture in flour or humid conditions',
                    'solution': 'Sift flour thoroughly; work in dry conditions',
                    'prevention': 'Store flour properly; check humidity levels'
                }
            ],
            'complexity_specific_issues': [
                {
                    'complexity_range': 'low',
                    'common_issues': ['Oversized elements', 'Poor proportion control'],
                    'solutions': ['Follow grid strictly', 'Use reference measurements']
                },
                {
                    'complexity_range': 'high',
                    'common_issues': ['Line intersection errors', 'Curve continuity problems'],
                    'solutions': ['Work in sections', 'Use temporary guide lines']
                }
            ],
            'environmental_considerations': [
                'High humidity causes flour to stick',
                'Direct sunlight fades colors quickly',
                'Wind affects fine detail work',
                'Temperature affects flour flow properties'
            ]
        }

    def _estimate_completion_time(self, complexity: float) -> str:
        """Estimate total completion time."""
        base_time = 30  # minutes
        complexity_multiplier = 1 + (complexity * 2)

        total_minutes = base_time * complexity_multiplier

        if total_minutes < 60:
            return f"{int(total_minutes)} minutes"
        else:
            hours = int(total_minutes // 60)
            minutes = int(total_minutes % 60)
            return f"{hours}h {minutes}m"

    def _determine_skill_level(self, complexity: float, symmetry_count: int) -> str:
        """Determine required skill level."""
        if complexity < 0.3 and symmetry_count <= 2:
            return "Beginner"
        elif complexity < 0.6 and symmetry_count <= 4:
            return "Intermediate"
        elif complexity < 0.8:
            return "Advanced"
        else:
            return "Expert"

    def _calculate_base_flour_requirement(self, pattern_size: str, complexity: float) -> int:
        """Calculate base flour requirement in grams."""
        size_multipliers = {
            'small': 1.0,
            'medium': 2.0,
            'large': 4.0,
            'extra_large': 8.0
        }

        base_grams = 100
        size_multiplier = size_multipliers.get(pattern_size, 2.0)
        complexity_multiplier = 1 + (complexity * 1.5)

        return int(base_grams * size_multiplier * complexity_multiplier)

    def _calculate_total_cost(self, flour_grams: int, complexity: float) -> str:
        """Calculate total estimated cost."""
        flour_cost = (flour_grams / 1000) * 50  # ₹50 per kg
        tools_cost = 500  # Average tools cost

    def _calculate_total_cost(self, flour_grams: int, complexity: float) -> str:
        """Calculate total estimated cost."""
        flour_cost = (flour_grams / 1000) * 50  # ₹50 per kg
        tools_cost = 500  # Average tools cost
        complexity_markup = complexity * 0.3

        total_cost = (flour_cost + tools_cost) * (1 + complexity_markup)
        return f"₹{total_cost:.0f}" return f"₹{total_cost:.0f}"        edges = cv2.Canny(pattern.astype(np.uint8), 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30,
                                minLineLength=10, maxLineGap=5)
        return len(lines) if lines is not None else 0

    def _count_intersection_points(self, pattern: np.ndarray) -> int:
        """Count intersection points in pattern."""
        # Dilate edges to find intersection areas
        edges = cv2.Canny(pattern.astype(np.uint8), 50, 150)
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)

        # Count connected components (intersection regions)
        num_components, _ = cv2.connectedComponents(dilated)
        return num_components

    def _generate_output_files(self, package: DocumentationPackage, formats: List[str]) -> Dict[str, str]:
        """Generate documentation in specified formats."""
        generated_files = {}

        for format_type in formats:
            try:
                if format_type == 'json':
                    file_path = self._generate_json_documentation(package)
                elif format_type == 'markdown':
                    file_path = self._generate_markdown_documentation(package)
                elif format_type == 'html':
                    file_path = self._generate_html_documentation(package)
                elif format_type == 'pdf':
                    file_path = self._generate_pdf_documentation(package)
                else:
                    continue

                generated_files[format_type] = file_path
                logger.info(
                    f"Generated {format_type} documentation: {file_path}")

            except Exception as e:
                logger.error(
                    f"Error generating {format_type} documentation: {e}")
                continue

        return generated_files

    def _generate_json_documentation(self, package: DocumentationPackage) -> str:
        """Generate JSON format documentation."""
        filename = f"{package.pattern_id}_documentation.json"
        file_path = os.path.join(self.output_dir, filename)

        # Convert package to dictionary
        package_dict = {
            'pattern_id': package.pattern_id,
            'pattern_name': package.pattern_name,
            'construction_guide': package.construction_guide,
            'material_requirements': package.material_requirements,
            'scaling_options': package.scaling_options,
            'cultural_context': package.cultural_context,
            'technical_specifications': package.technical_specifications,
            'quality_standards': package.quality_standards,
            'troubleshooting_guide': package.troubleshooting_guide,
            'generation_metadata': package.generation_metadata
        }

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(package_dict, f, indent=2, ensure_ascii=False)

        return file_path

    def _generate_markdown_documentation(self, package: DocumentationPackage) -> str:
        """Generate Markdown format documentation."""
        filename = f"{package.pattern_id}_documentation.md"
        file_path = os.path.join(self.output_dir, filename)

        # Create comprehensive markdown documentation
        markdown_content = f"""# Kolam Pattern Construction Guide

## Pattern Information
- **Pattern ID**: {package.pattern_id}
- **Pattern Name**: {package.pattern_name}
- **Generated**: {package.generation_metadata['generated_at']}
- **Complexity Score**: {package.generation_metadata['pattern_complexity']:.3f}
- **Cultural Authenticity**: {package.generation_metadata['cultural_authenticity']:.3f}

## Construction Overview
- **Estimated Time**: {package.construction_guide['estimated_completion_time']}
- **Skill Level**: {package.construction_guide['skill_level_required']}
- **Approach**: {package.construction_guide['construction_approach']}

## Step-by-Step Instructions

"""

        # Add construction steps
        for step in package.construction_guide['steps']:
            markdown_content += f"""### Step {step['step_number']}: {step['title']}

**Description**: {step['description']}

**Tools Required**: {', '.join(step['tools_required'])}
**Materials Needed**: {', '.join(step['materials_needed'])}
**Time Estimate**: {step['time_estimate']}
**Difficulty**: {'⭐' * step['difficulty_rating']} ({step['difficulty_rating']}/5)

**Quality Checks**:
"""
            for check in step['quality_checks']:
                markdown_content += f"- {check}\n"

            markdown_content += "\n**Common Mistakes**:\n"
            for mistake in step['common_mistakes']:
                markdown_content += f"- {mistake}\n"

            markdown_content += "\n**Tips & Tricks**:\n"
            for tip in step['tips_and_tricks']:
                markdown_content += f"- {tip}\n"

            markdown_content += "\n---\n\n"

        # Add material requirements
        markdown_content += "## Material Requirements\n\n"
        for material in package.material_requirements['primary_materials']:
            markdown_content += f"### {material['material_name']}\n"
            markdown_content += f"- **Quantity**: {material['quantity']}\n"
            markdown_content += f"- **Cost per Unit**: ₹{material['cost_per_unit']}\n"
            markdown_content += f"- **Source Suggestions**: {', '.join(material['source_suggestions'])}\n"
            markdown_content += f"- **Storage Notes**: {material['storage_notes']}\n\n"

        # Add cultural context
        markdown_content += "## Cultural Context\n\n"
        cultural = package.cultural_context
        markdown_content += f"**Primary Meaning**: {cultural['cultural_significance']['primary_meaning']}\n\n"
        markdown_content += "**Historical Context**: " + \
            cultural['historical_context']['origins'] + "\n\n"

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)

        return file_path

    def _generate_html_documentation(self, package: DocumentationPackage) -> str:
        """Generate HTML format documentation."""
        filename = f"{package.pattern_id}_documentation.html"
        file_path = os.path.join(self.output_dir, filename)

        # Create HTML template
        html_template = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Kolam Pattern Guide - {{ pattern_name }}</title>
            <style>
                body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; line-height: 1.6; }
                .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; }
                .section { margin: 30px 0; padding: 20px; border-left: 4px solid #667eea; background: #f9f9f9; }
                .step { background: white; margin: 15px 0; padding: 15px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
                .step-number { display: inline-block; background: #667eea; color: white; width: 30px; height: 30px; border-radius: 50%; text-align: center; line-height: 30px; margin-right: 15px; }
                table { width: 100%; border-collapse: collapse; margin: 15px 0; }
                th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
                th { background-color: #f2f2f2; }
                .authenticity-badge { padding: 5px 10px; border-radius: 15px; color: white; font-weight: bold; }
                .high-authenticity { background-color: #4CAF50; }
                .medium-authenticity { background-color: #FF9800; }
                .low-authenticity { background-color: #f44336; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Kolam Pattern Construction Guide</h1>
                <h2>{{ pattern_name }}</h2>
                <p><strong>Pattern ID:</strong> {{ pattern_id }}</p>
                <p><strong>Generated:</strong> {{ generation_metadata.generated_at }}</p>
                <p><strong>Complexity:</strong> {{ "%.2f"|format(generation_metadata.pattern_complexity) }}</p>
                <p><strong>Cultural Authenticity:</strong>
                    <span class="authenticity-badge high-authenticity">
                        {{ "%.1f"|format(
                            generation_metadata.cultural_authenticity) }}/1.0
                    </span>
                </p>
            </div>

            <div class="section">
                <h2>Construction Overview</h2>
                <p><strong>Estimated Time:</strong> {{ construction_guide.estimated_completion_time }}</p>
                <p><strong>Skill Level:</strong> {{ construction_guide.skill_level_required }}</p>
                <p><strong>Approach:</strong> {{ construction_guide.construction_approach }}</p>
            </div>

            <div class="section">
                <h2>Step-by-Step Instructions</h2>
                {% for step in construction_guide.steps %}
                <div class="step">
                    <span class="step-number">{{ step.step_number }}</span>
                    <h3>{{ step.title }}</h3>
                    <p><strong>Description:</strong> {{ step.description }}</p>
                    <p><strong>Tools:</strong> {{ step.tools_required|join(', ') }}</p>
                    <p><strong>Materials:</strong> {{ step.materials_needed|join(', ') }}</p>
                    <p><strong>Time:</strong> {{ step.time_estimate }}</p>
                    <p><strong>Difficulty:</strong> {{ '⭐' * step.difficulty_rating }}</p>
                </div>
                {% endfor %}
            </div>

            <div class="section">
                <h2>Material Requirements</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Material</th>
                            <th>Quantity</th>
                            <th>Estimated Cost</th>
                            <th>Source</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for material in material_requirements.primary_materials %}
                        <tr>
                            <td>{{ material.material_name }}</td>
                            <td>{{ material.quantity }}</td>
{% for material in material_requirements.primary_materials %}
<tr>
    <td>{{ material.material_name }}</td>
    <td>{{ material.quantity }}</td>
    <td>₹{{ (material.cost_per_unit * material.quantity|replace(' grams', '')|int) | round(1) }}</td>
    <td>{{ material.source_suggestions[0] }}</td>
</tr>
{% endfor %}            <div class="section">
                <h2>Cultural Context</h2>
                <p><strong>Primary Meaning:</strong> {{ cultural_context.cultural_significance.primary_meaning }}</p>
                <p><strong>Ritual Importance:</strong> {{ cultural_context.cultural_significance.ritual_importance }}</p>
                <p><strong>Historical Origins:</strong> {{ cultural_context.historical_context.origins }}</p>
            </div>
        </body>
        </html>
        """

        # Render template
        template = Template(html_template)
        html_content = template.render(
            pattern_id=package.pattern_id,
            pattern_name=package.pattern_name,
            generation_metadata=package.generation_metadata,
            construction_guide=package.construction_guide,
            material_requirements=package.material_requirements,
            cultural_context=package.cultural_context
        )

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return file_path

    def _generate_pdf_documentation(self, package: DocumentationPackage) -> str:
        """Generate PDF format documentation (simplified - would use reportlab in practice)."""
        filename = f"{package.pattern_id}_documentation.pdf"
        file_path = os.path.join(self.output_dir, filename)

        # For now, create a text-based PDF placeholder
        # In practice, would use a PDF library like reportlab or weasyprint
        pdf_content = f"""
        KOLAM PATTERN DOCUMENTATION
        ==========================

        Pattern: {package.pattern_name}
        ID: {package.pattern_id}
        Generated: {package.generation_metadata['generated_at']}

        CONSTRUCTION GUIDE:
        {package.construction_guide['estimated_completion_time']}
        Skill Level: {package.construction_guide['skill_level_required']}

        MATERIAL REQUIREMENTS:
        See detailed material list in JSON format.

        CULTURAL CONTEXT:
        {package.cultural_context['cultural_significance']['primary_meaning']}

        TECHNICAL SPECIFICATIONS:
        Dimensions: {
            package.technical_specifications['pattern_dimensions']['recommended_physical_size']}

        For complete documentation, please refer to the HTML and JSON formats.
        """

        with open(file_path.replace('.pdf', '.txt'), 'w', encoding='utf-8') as f:
            f.write(pdf_content)

        logger.info(
            f"PDF documentation placeholder created: {file_path.replace('.pdf', '.txt')}")
        return file_path.replace('.pdf', '.txt')  # Return actual file created


def main():
    """Demonstrate the documentation generator."""
    # Initialize generator
    generator = KolamDocumentationGenerator()

    # Create test pattern
    test_pattern = np.zeros((100, 100), dtype=np.uint8)
    center = 50
    cv2.circle(test_pattern, (center, center), 30, 255, 2)

    # Mock features and validation
    mock_features = {
        'pattern_type': 'Traditional Pulli Kolam',
        'dominant_symmetries': ['rotational_4_fold'],
        'pattern_complexity_score': 0.6,
        'estimated_size': 'medium'
    }

    mock_validation = {
        'authenticity_score': 0.85,
        'is_authentic': True
    }

    print("KOLAM COMPREHENSIVE DOCUMENTATION GENERATOR")
    print("="*60)

    try:
        # Generate complete documentation package
        documentation = generator.generate_complete_documentation(
            test_pattern, mock_features, mock_validation,
            output_formats=['json', 'markdown', 'html']
        )

        print("✅ Documentation package generated successfully!")
        print(f"📄 Pattern ID: {documentation.pattern_id}")
        print(f"📋 Pattern Name: {documentation.pattern_name}")
        print(
            f"⏱️  Estimated Construction Time: {documentation.construction_guide['estimated_completion_time']}")
        print(
            f"👥 Skill Level: {documentation.construction_guide['skill_level_required']}")
        print(
            f"📊 Total Steps: {len(documentation.construction_guide['steps'])}")
        print(
            f"🛠️  Material Items: {len(documentation.material_requirements['primary_materials'])}")
        print(f"📏 Scaling Options: {len(documentation.scaling_options)}")
        print(
            f"📚 Cultural Authenticity Score: {documentation.generation_metadata['cultural_authenticity']:.3f}")

        print("
📁 GENERATED FORMATS:"        print("   • JSON - Complete structured data"        print("   • Markdown - Human-readable guide"        print("   • HTML - Interactive web documentation"
        print(f"\n💾 All files saved to: {generator.output_dir}")

    except Exception as e:
        print("\n📁 GENERATED FORMATS:")
        print("   • JSON - Complete structured data")
        print("   • Markdown - Human-readable guide")
        print("   • HTML - Interactive web documentation")
        print(f"\n💾 All files saved to: {generator.output_dir}")    print("COMPREHENSIVE DOCUMENTATION FEATURES:")
    print("✅ Step-by-step construction guides")
    print("✅ Detailed material requirements")
    print("✅ Multiple scaling options")
    print("✅ Cultural context and significance")
    print("✅ Technical specifications")
    print("✅ Quality standards and best practices")
    print("✅ Troubleshooting guides")
    print("✅ Multiple output formats (JSON, Markdown, HTML)")
    print("✅ Cost estimates and time calculations")
    print("✅ Skill level assessments")
    print("✅ Safety considerations")

    print("\nDocumentation generator is ready for comprehensive kolam pattern guides!")


if __name__ == "__main__":
    main()