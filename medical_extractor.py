import re
import json
from typing import Dict, List, Any, Tuple
import spacy
from collections import defaultdict

class MedicalInformationExtractor:
    def __init__(self):
        """Initialize the medical information extractor with dynamic capabilities."""
        try:
            # Try to load the base English model
            self.nlp = spacy.load("en_core_web_sm")
            self.use_spacy = True
        except OSError:
            print("spaCy model not found. Using rule-based extraction.")
            self.nlp = None
            self.use_spacy = False
        
        # Medical context keywords for dynamic extraction
        self.medical_context = {
            'symptoms': [
                'pain', 'ache', 'discomfort', 'burning', 'sharp', 'dull', 'cramping',
                'shortness of breath', 'difficulty breathing', 'dyspnea', 'wheezing',
                'cough', 'fever', 'chills', 'fatigue', 'weakness', 'exhaustion',
                'nausea', 'vomiting', 'diarrhea', 'constipation', 'bloating',
                'dizziness', 'lightheadedness', 'fainting', 'syncope',
                'headache', 'migraine', 'confusion', 'memory loss',
                'rash', 'itching', 'swelling', 'edema', 'redness',
                'numbness', 'tingling', 'weakness', 'paralysis'
            ],
            'vital_signs': [
                'blood pressure', 'bp', 'hypertension', 'hypotension',
                'heart rate', 'pulse', 'tachycardia', 'bradycardia',
                'temperature', 'fever', 'hypothermia',
                'respiratory rate', 'breathing rate', 'tachypnea', 'bradypnea',
                'oxygen saturation', 'spo2', 'oxygen level'
            ],
            'lab_tests': [
                'blood test', 'lab test', 'laboratory', 'cbc', 'cmp',
                'troponin', 'ck', 'ck-mb', 'cardiac enzymes',
                'glucose', 'blood sugar', 'hba1c',
                'creatinine', 'bun', 'kidney function',
                'electrolytes', 'sodium', 'potassium', 'chloride',
                'liver function', 'ast', 'alt', 'bilirubin',
                'lipid panel', 'cholesterol', 'ldl', 'hdl', 'triglycerides',
                'thyroid', 'tsh', 't3', 't4',
                'inflammation', 'crp', 'esr',
                'urinalysis', 'urine test'
            ],
            'medications': [
                'medication', 'medicine', 'drug', 'dose', 'dosage', 'mg', 'mcg',
                'tablet', 'capsule', 'injection', 'iv', 'po', 'prn',
                'antibiotic', 'antihypertensive', 'antidiabetic', 'anticoagulant',
                'pain medication', 'analgesic', 'nsaid', 'opioid'
            ],
            'procedures': [
                'ecg', 'ekg', 'electrocardiogram', 'x-ray', 'ct scan', 'mri',
                'ultrasound', 'sonogram', 'echocardiogram', 'echo',
                'stress test', 'holter', 'catheterization', 'angiogram',
                'endoscopy', 'colonoscopy', 'biopsy', 'lumbar puncture'
            ]
        }
        
        # Common medical abbreviations
        self.medical_abbreviations = {
            'c/o': 'complains of',
            'h/o': 'history of',
            'p/w': 'presents with',
            's/s': 'signs and symptoms',
            'r/o': 'rule out',
            'w/i': 'within',
            'w/o': 'without',
            'b/l': 'bilateral',
            'u/l': 'unilateral',
            'qhs': 'at bedtime',
            'qid': 'four times daily',
            'tid': 'three times daily',
            'bid': 'twice daily',
            'qd': 'once daily',
            'prn': 'as needed',
            'stat': 'immediately'
        }
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess the medical text for better extraction."""
        # Convert to lowercase for consistency
        text = text.lower()
        
        # Expand common medical abbreviations
        for abbr, expansion in self.medical_abbreviations.items():
            text = re.sub(r'\b' + re.escape(abbr) + r'\b', expansion, text)
        
        # Normalize medical terms
        text = re.sub(r'\b(st\s+elevation|st-e)\b', 'st elevation', text)
        text = re.sub(r'\b(st\s+depression|st-d)\b', 'st depression', text)
        text = re.sub(r'\b(t\s+wave|t-wave)\b', 't wave', text)
        text = re.sub(r'\b(q\s+wave|q-wave)\b', 'q wave', text)
        
        return text
    
    def extract_entities_spacy(self, text: str) -> Dict[str, List[str]]:
        """Extract medical entities using spaCy NER."""
        if not self.use_spacy:
            return {}
        
        doc = self.nlp(text)
        entities = defaultdict(list)
        
        for ent in doc.ents:
            # Categorize entities based on context
            entity_text = ent.text.lower()
            
            # Check if entity is in medical context
            for category, keywords in self.medical_context.items():
                if any(keyword in entity_text for keyword in keywords):
                    entities[category].append(ent.text)
                    break
            else:
                # Default categorization based on entity type
                if ent.label_ in ['PERSON', 'NORP', 'ORG']:
                    continue  # Skip non-medical entities
                elif ent.label_ in ['CARDINAL', 'QUANTITY']:
                    if 'year' in entity_text or 'age' in entity_text:
                        entities['demographics'].append(ent.text)
                    elif any(unit in entity_text for unit in ['mg', 'mcg', 'g', 'ml', 'l']):
                        entities['lab_results'].append(ent.text)
                    elif any(unit in entity_text for unit in ['bpm', 'mmhg', '%', '°c', '°f']):
                        entities['vital_signs'].append(ent.text)
                elif ent.label_ in ['DISEASE', 'DISORDER', 'SYMPTOM']:
                    entities['symptoms'].append(ent.text)
                elif ent.label_ in ['MEDICATION', 'DRUG']:
                    entities['medications'].append(ent.text)
                elif ent.label_ in ['PROCEDURE', 'TEST']:
                    entities['procedures'].append(ent.text)
        
        return dict(entities)
    
    def extract_entities_rule_based(self, text: str) -> Dict[str, List[str]]:
        """Extract medical entities using rule-based patterns."""
        entities = defaultdict(list)
        
        # Extract age
        age_patterns = [
            r'(\d+)-year-old',
            r'age\s+(\d+)',
            r'(\d+)\s+years?\s+old',
            r'aged\s+(\d+)'
        ]
        for pattern in age_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                entities['demographics'].append(f"{match} years old")
        
        # Extract gender
        gender_patterns = [
            (r'\b(male|man|gentleman)\b', 'male'),
            (r'\b(female|woman|lady)\b', 'female')
        ]
        for pattern, gender in gender_patterns:
            if re.search(pattern, text):
                entities['demographics'].append(gender)
        
        # Extract vital signs with values
        vital_patterns = {
            'blood_pressure': r'bp\s+(\d+)/(\d+)|blood pressure\s+(\d+)/(\d+)|(\d+)/(\d+)\s+mmhg',
            'heart_rate': r'heart rate\s+(\d+)|pulse\s+(\d+)|hr\s+(\d+)',
            'temperature': r'temperature\s+(\d+\.?\d*)|temp\s+(\d+\.?\d*)',
            'respiratory_rate': r'respiratory rate\s+(\d+)|rr\s+(\d+)',
            'oxygen_saturation': r'oxygen saturation\s+(\d+)%|spo2\s+(\d+)%'
        }
        
        for vital, pattern in vital_patterns.items():
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, tuple):
                    match = [m for m in match if m][0]  # Get first non-empty group
                entities['vital_signs'].append(f"{vital}: {match}")
        
        # Extract lab values
        lab_patterns = {
            'troponin': r'troponin\s*:?\s*(\d+\.?\d*)',
            'glucose': r'glucose\s*:?\s*(\d+)',
            'creatinine': r'creatinine\s*:?\s*(\d+\.?\d*)',
            'hemoglobin': r'hemoglobin\s*:?\s*(\d+\.?\d*)',
            'white_blood_cell': r'wbc\s*:?\s*(\d+,?\d*)'
        }
        
        for lab, pattern in lab_patterns.items():
            matches = re.findall(pattern, text)
            for match in matches:
                entities['lab_results'].append(f"{lab}: {match}")
        
        # Extract symptoms by context
        for symptom in self.medical_context['symptoms']:
            if symptom in text:
                # Extract the context around the symptom
                pattern = rf'.{{0,50}}{re.escape(symptom)}.{{0,50}}'
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    entities['symptoms'].append(match.strip())
        
        return dict(entities)
    
    def extract_medical_history(self, text: str) -> List[str]:
        """Extract medical history dynamically."""
        history_keywords = [
            'history of', 'past medical history', 'pmh', 'pmhx',
            'diagnosed with', 'known case of', 'suffers from',
            'chronic', 'long-standing', 'history of'
        ]
        
        history_items = []
        text_lower = text.lower()
        
        for keyword in history_keywords:
            # Find sentences containing history keywords
            pattern = rf'[^.]*{re.escape(keyword)}[^.]*\.'
            matches = re.findall(pattern, text_lower)
            for match in matches:
                # Clean and add to history
                cleaned = re.sub(r'\b(the|a|an|with|has|have|had)\b', '', match).strip()
                if len(cleaned) > 10:  # Filter out very short matches
                    history_items.append(cleaned)
        
        return history_items
    
    def extract_medications_dynamic(self, text: str) -> List[str]:
        """Extract medications dynamically."""
        medication_indicators = [
            'takes', 'on', 'prescribed', 'medication', 'medicine', 'drug',
            'dose', 'dosage', 'mg', 'tablet', 'capsule', 'injection'
        ]
        
        medications = []
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            if any(indicator in sentence.lower() for indicator in medication_indicators):
                # Extract potential medication names (capitalized words)
                words = re.findall(r'\b[A-Z][a-z]+\b', sentence)
                for word in words:
                    if len(word) > 3:  # Filter out short words
                        medications.append(word)
        
        return list(set(medications))
    
    def extract_timeline_dynamic(self, text: str) -> Dict[str, str]:
        """Extract timeline information dynamically."""
        timeline = {}
        
        # Time expressions
        time_patterns = {
            'onset': r'(sudden|gradual|acute|chronic|insidious)\s+onset',
            'duration': r'(\d+)\s+(hours?|days?|weeks?|months?|years?)\s+ago',
            'frequency': r'(daily|weekly|monthly|occasionally|frequently|rarely)',
            'progression': r'(worsening|improving|stable|progressive)'
        }
        
        for aspect, pattern in time_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                timeline[aspect] = match.group(0)
        
        return timeline
    
    def extract_all_information(self, text: str) -> Dict[str, Any]:
        """Extract all meaningful medical information dynamically."""
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Extract entities using available methods
        if self.use_spacy:
            spacy_entities = self.extract_entities_spacy(processed_text)
        else:
            spacy_entities = {}
        
        rule_entities = self.extract_entities_rule_based(processed_text)
        
        # Merge entities from both methods
        merged_entities = defaultdict(list)
        for entities in [spacy_entities, rule_entities]:
            for category, items in entities.items():
                merged_entities[category].extend(items)
        
        # Remove duplicates while preserving order
        for category in merged_entities:
            seen = set()
            merged_entities[category] = [x for x in merged_entities[category] if not (x in seen or seen.add(x))]
        
        # Extract dynamic information
        extracted_info = {
            'demographics': merged_entities.get('demographics', []),
            'symptoms': merged_entities.get('symptoms', []),
            'vital_signs': merged_entities.get('vital_signs', []),
            'lab_results': merged_entities.get('lab_results', []),
            'medications': self.extract_medications_dynamic(text),
            'medical_history': self.extract_medical_history(text),
            'procedures': merged_entities.get('procedures', []),
            'timeline': self.extract_timeline_dynamic(text)
        }
        
        return extracted_info
    
    def create_enhanced_query(self, extracted_info: Dict[str, Any]) -> str:
        """Create an enhanced query for the RAG system based on extracted information."""
        query_parts = []
        
        # Add demographics
        if extracted_info['demographics']:
            demo_text = ' '.join(extracted_info['demographics'][:2])
            query_parts.append(f"Patient: {demo_text}")
        
        # Add chief symptoms (limit to most relevant)
        if extracted_info['symptoms']:
            symptoms_text = ' '.join(extracted_info['symptoms'][:3])
            query_parts.append(f"Symptoms: {symptoms_text}")
        
        # Add vital signs abnormalities
        if extracted_info['vital_signs']:
            vitals_text = ' '.join(extracted_info['vital_signs'][:3])
            query_parts.append(f"Vital signs: {vitals_text}")
        
        # Add key lab results
        if extracted_info['lab_results']:
            labs_text = ' '.join(extracted_info['lab_results'][:3])
            query_parts.append(f"Lab results: {labs_text}")
        
        # Add relevant medical history
        if extracted_info['medical_history']:
            history_text = ' '.join(extracted_info['medical_history'][:2])
            query_parts.append(f"History: {history_text}")
        
        # Add timeline information
        if extracted_info['timeline']:
            timeline_items = [f"{k}: {v}" for k, v in extracted_info['timeline'].items()]
            query_parts.append(f"Timeline: {', '.join(timeline_items)}")
        
        return '. '.join(query_parts)
    
    def extract_key_findings(self, text: str) -> List[str]:
        """Extract key clinical findings dynamically."""
        # Look for sentences with clinical significance
        key_patterns = [
            r'\b(elevated|increased|high|raised)\b.*\b(troponin|glucose|creatinine|white blood cell)\b',
            r'\b(decreased|low|reduced)\b.*\b(hemoglobin|oxygen|blood pressure)\b',
            r'\b(positive|negative)\b.*\b(test|result)\b',
            r'\b(abnormal|normal)\b.*\b(ecg|ekg|x-ray|ct|mri)\b',
            r'\b(st\s+elevation|st\s+depression|t\s+wave\s+inversion)\b',
            r'\b(consolidation|effusion|infarct|edema)\b'
        ]
        
        key_findings = []
        for pattern in key_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            key_findings.extend(matches)
        
        return list(set(key_findings))