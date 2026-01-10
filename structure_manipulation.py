"""
Module for manipulating clinical note structure.
Creates different structural variants to test RQ1.
"""

import re
import random
from typing import Dict, List


class NoteStructureManipulator:
    """
    Transform clinical notes into different structural formats
    to test the effect of structure on LLM performance.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)

        # Common section headers in clinical notes
        self.section_patterns = [
            r'Chief Complaint:',
            r'History of Present Illness:',
            r'Past Medical History:',
            r'Social History:',
            r'Family History:',
            r'Physical Exam:',
            r'Pertinent Results:',
            r'Brief Hospital Course:',
            r'Discharge Medications:',
            r'Discharge Disposition:',
            r'Discharge Diagnosis:',
            r'Discharge Condition:',
            r'Discharge Instructions:',
            r'Major Surgical or Invasive Procedure:',
            r'Medications on Admission:',
        ]

    def extract_clinical_content(self, text: str) -> str:
        """
        Extract clinical content, skipping administrative header.
        """
        # Find the start of clinical content (usually Chief Complaint)
        match = re.search(r'Chief Complaint:', text, re.IGNORECASE)
        if match:
            return text[match.start():]

        # Fallback: skip first 500 chars (admin info)
        return text[500:]

    def get_original(self, text: str) -> str:
        """
        Return original structured format.
        """
        return self.extract_clinical_content(text)

    def remove_headers(self, text: str) -> str:
        """
        Remove section headers but keep content and structure.
        """
        clinical_text = self.extract_clinical_content(text)

        # Remove section headers
        for pattern in self.section_patterns:
            clinical_text = re.sub(pattern, '', clinical_text, flags=re.IGNORECASE)

        return clinical_text.strip()

    def remove_formatting(self, text: str) -> str:
        """
        Remove all formatting: headers, bullets, excess whitespace.
        Convert to continuous text.
        """
        clinical_text = self.extract_clinical_content(text)

        # Remove section headers
        for pattern in self.section_patterns:
            clinical_text = re.sub(pattern, '', clinical_text, flags=re.IGNORECASE)

        # Remove bullet points
        clinical_text = re.sub(r'[\*\-•]\s+', '', clinical_text)

        # Replace multiple newlines with single space
        clinical_text = re.sub(r'\n+', ' ', clinical_text)

        # Replace multiple spaces with single space
        clinical_text = re.sub(r'\s+', ' ', clinical_text)

        return clinical_text.strip()

    def shuffle_sections(self, text: str) -> str:
        """
        Shuffle the order of sections to disrupt logical flow.
        Tests if models rely on typical clinical note structure.
        """
        clinical_text = self.extract_clinical_content(text)

        # Find all sections
        sections = []
        current_pos = 0

        # Create pattern to match any section header
        combined_pattern = '|'.join(self.section_patterns)

        for match in re.finditer(combined_pattern, clinical_text, re.IGNORECASE):
            if current_pos > 0:
                # Save previous section
                sections.append(clinical_text[current_pos:match.start()])
            current_pos = match.start()

        # Add last section
        if current_pos > 0:
            sections.append(clinical_text[current_pos:])

        # Shuffle sections (if we found multiple)
        if len(sections) > 1:
            random.shuffle(sections)
            return '\n\n'.join(sections).strip()

        # If no sections found, return original
        return clinical_text

    def apply_variant(self, text: str, variant: str) -> str:
        """
        Apply specified structure variant to text.

        Args:
            text: Original clinical note
            variant: One of ['original', 'no_headers', 'no_formatting', 'shuffled']

        Returns:
            Transformed text
        """
        variant_map = {
            'original': self.get_original,
            'no_headers': self.remove_headers,
            'no_formatting': self.remove_formatting,
            'shuffled': self.shuffle_sections,
        }

        if variant not in variant_map:
            raise ValueError(f"Unknown variant: {variant}. Choose from {list(variant_map.keys())}")

        return variant_map[variant](text)

    def create_all_variants(self, text: str) -> Dict[str, str]:
        """
        Create all structure variants for a single note.

        Returns:
            Dictionary mapping variant name to transformed text
        """
        return {
            variant: self.apply_variant(text, variant)
            for variant in ['original', 'no_headers', 'no_formatting', 'shuffled']
        }


def demo_structure_variants():
    """
    Demonstrate the different structure variants.
    """
    sample_note = """
Chief Complaint:
L sided weakness/numbness

History of Present Illness:
The pt is a right-handed woman with a history of HTN, HL, migraines.
She reports that she was in the shower when she noticed difficulty
raising her left arm.

Physical Exam:
Vitals: 97.7 78 118/78 14 95% RA
General: Awake, pleasant and cooperative, NAD.
Neurologic: Alert, oriented x 3.
    """

    manipulator = NoteStructureManipulator()

    print("="*80)
    print("ORIGINAL:")
    print("="*80)
    print(manipulator.get_original(sample_note))

    print("\n" + "="*80)
    print("NO HEADERS:")
    print("="*80)
    print(manipulator.remove_headers(sample_note))

    print("\n" + "="*80)
    print("NO FORMATTING:")
    print("="*80)
    print(manipulator.remove_formatting(sample_note))

    print("\n" + "="*80)
    print("SHUFFLED:")
    print("="*80)
    print(manipulator.shuffle_sections(sample_note))


if __name__ == '__main__':
    demo_structure_variants()
