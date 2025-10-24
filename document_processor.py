import os
import io
import tempfile
import base64
from typing import Dict, List, Any, Union, Tuple
import fitz  # PyMuPDF
import pdfplumber
import pytesseract
from PIL import Image
import easyocr
import re
import json
import numpy as np  # Added this import

class DocumentProcessor:
    def __init__(self):
        """Initialize the document processor with OCR capabilities."""
        # Initialize EasyOCR reader (more reliable than pytesseract for medical text)
        try:
            self.reader = easyocr.Reader(['en'])
            self.use_easyocr = True
            print("EasyOCR initialized successfully")
        except Exception as e:
            print(f"EasyOCR initialization failed: {e}")
            self.use_easyocr = False
        
        # Common medical terms to improve OCR accuracy
        self.medical_terms = {
            'symptoms': ['pain', 'chest pain', 'shortness of breath', 'dyspnea', 'fever', 'cough', 'nausea', 'vomiting', 'fatigue', 'weakness'],
            'diagnoses': ['hypertension', 'diabetes', 'pneumonia', 'myocardial infarction', 'stroke', 'copd', 'asthma'],
            'medications': ['aspirin', 'insulin', 'metformin', 'lisinopril', 'atorvastatin', 'metoprolol'],
            'procedures': ['ecg', 'ekg', 'ct scan', 'mri', 'x-ray', 'ultrasound', 'echocardiogram'],
            'vitals': ['bp', 'blood pressure', 'hr', 'heart rate', 'rr', 'respiratory rate', 'temp', 'temperature']
        }
    
    def extract_text_from_pdf(self, pdf_file: Union[bytes, str]) -> str:
        """Extract text from PDF file using multiple methods."""
        text_content = ""
        
        try:
            # Method 1: Try pdfplumber first (better for structured PDFs)
            if isinstance(pdf_file, bytes):
                with pdfplumber.open(io.BytesIO(pdf_file)) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text.strip():
                            text_content += page_text + "\n"
            else:
                with pdfplumber.open(pdf_file) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text.strip():
                            text_content += page_text + "\n"
            
            # If we got substantial text, return it
            if len(text_content.strip()) > 100:
                return text_content
        except Exception as e:
            print(f"pdfplumber extraction failed: {e}")
        
        try:
            # Method 2: Try PyMuPDF
            if isinstance(pdf_file, bytes):
                doc = fitz.open(stream=io.BytesIO(pdf_file), filetype="pdf")
            else:
                doc = fitz.open(pdf_file)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                if page_text.strip():
                    text_content += page_text + "\n"
            
            doc.close()
            
            # If we got substantial text, return it
            if len(text_content.strip()) > 100:
                return text_content
        except Exception as e:
            print(f"PyMuPDF extraction failed: {e}")
        
        # Method 3: If text extraction failed, try OCR
        try:
            return self.extract_text_from_pdf_ocr(pdf_file)
        except Exception as e:
            print(f"PDF OCR extraction failed: {e}")
            return ""
    
    def extract_text_from_pdf_ocr(self, pdf_file: Union[bytes, str]) -> str:
        """Extract text from PDF using OCR."""
        text_content = ""
        
        try:
            # Convert PDF to images
            if isinstance(pdf_file, bytes):
                doc = fitz.open(stream=io.BytesIO(pdf_file), filetype="pdf")
            else:
                doc = fitz.open(pdf_file)
            
            for page_num in range(min(len(doc), 10)):  # Limit to first 10 pages
                page = doc.load_page(page_num)
                # Convert page to image
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                
                # Extract text from image
                page_text = self.extract_text_from_image(img)
                if page_text.strip():
                    text_content += page_text + "\n"
            
            doc.close()
        except Exception as e:
            print(f"PDF OCR failed: {e}")
        
        return text_content
    
    def extract_text_from_image(self, image_file: Union[bytes, Image.Image, str]) -> str:
        """Extract text from image using OCR."""
        try:
            # Convert to PIL Image if needed
            if isinstance(image_file, bytes):
                image = Image.open(io.BytesIO(image_file))
            elif isinstance(image_file, str):
                image = Image.open(image_file)
            else:
                image = image_file
            
            # Preprocess image for better OCR
            image = self.preprocess_image(image)
            
            # Use EasyOCR if available
            if self.use_easyocr:
                try:
                    # Convert PIL Image to numpy array for EasyOCR
                    image_array = np.array(image)
                    results = self.reader.readtext(image_array)
                    text = " ".join([result[1] for result in results])
                    return self.post_process_ocr_text(text)
                except Exception as e:
                    print(f"EasyOCR failed: {e}")
            
            # Fallback to pytesseract
            try:
                text = pytesseract.image_to_string(image)
                return self.post_process_ocr_text(text)
            except Exception as e:
                print(f"Tesseract OCR failed: {e}")
                return ""
        except Exception as e:
            print(f"Image processing failed: {e}")
            return ""
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image for better OCR results."""
        # Convert to grayscale
        if image.mode != 'L':
            image = image.convert('L')
        
        # Increase contrast
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)
        
        # Resize if too large
        max_size = 2000
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.LANCZOS)
        
        return image
    
    def post_process_ocr_text(self, text: str) -> str:
        """Post-process OCR text to fix common errors."""
        # Fix common OCR errors in medical terms
        corrections = {
            'rn': 'm',  # Common OCR error
            'cl': 'd',   # Common OCR error
            '0': 'o',   # Number zero to letter o in context
            '1': 'l',   # Number one to letter l in context
            '5': 's',   # Number five to letter s in context
            '8': 'b',   # Number eight to letter b in context
        }
        
        # Apply corrections in medical context
        for wrong, right in corrections.items():
            text = re.sub(r'\b' + wrong + r'\b', right, text)
        
        # Fix spacing issues
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common medical term OCR errors
        medical_corrections = {
            'hypertemsion': 'hypertension',
            'diabites': 'diabetes',
            'pneumona': 'pneumonia',
            'myocardial': 'myocardial',
            'cardiac': 'cardiac',
            'respiratoy': 'respiratory',
            'cardiovascuar': 'cardiovascular'
        }
        
        for wrong, right in medical_corrections.items():
            text = re.sub(r'\b' + wrong + r'\b', right, text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def extract_structured_data(self, text: str) -> Dict[str, Any]:
        """Extract structured medical data from OCR text."""
        structured_data = {
            'demographics': {},
            'vitals': {},
            'symptoms': [],
            'diagnoses': [],
            'medications': [],
            'procedures': [],
            'lab_results': {}
        }
        
        # Extract age
        age_patterns = [
            r'(\d+)[- ]?year[- ]?old',
            r'age[:\s]+(\d+)',
            r'(\d+)\s+years?\s+old',
            r'aged\s+(\d+)'
        ]
        
        for pattern in age_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                structured_data['demographics']['age'] = int(match.group(1))
                break
        
        # Extract gender
        if re.search(r'\b(male|man|gentleman)\b', text, re.IGNORECASE):
            structured_data['demographics']['gender'] = 'male'
        elif re.search(r'\b(female|woman|lady)\b', text, re.IGNORECASE):
            structured_data['demographics']['gender'] = 'female'
        
        # Extract vitals
        vital_patterns = {
            'blood_pressure': r'bp[:\s]*(\d+)/(\d+)|blood pressure[:\s]*(\d+)/(\d+)',
            'heart_rate': r'hr[:\s]*(\d+)|heart rate[:\s]*(\d+)|pulse[:\s]*(\d+)',
            'temperature': r'temp[:\s]*(\d+\.?\d*)|temperature[:\s]*(\d+\.?\d*)',
            'respiratory_rate': r'rr[:\s]*(\d+)|respiratory rate[:\s]*(\d+)',
            'oxygen_saturation': r'spo2[:\s]*(\d+)%|oxygen saturation[:\s]*(\d+)%'
        }
        
        for vital, pattern in vital_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    if vital == 'blood_pressure':
                        structured_data['vitals'][vital] = {
                            'systolic': int(match.group(1)),
                            'diastolic': int(match.group(2))
                        }
                    else:
                        value = match.group(1) or match.group(2) or match.group(3) or match.group(4)
                        structured_data['vitals'][vital] = float(value) if '.' in value else int(value)
                except (ValueError, IndexError):
                    pass
        
        # Extract symptoms (using medical terms dictionary)
        for symptom in self.medical_terms['symptoms']:
            if re.search(r'\b' + re.escape(symptom) + r'\b', text, re.IGNORECASE):
                structured_data['symptoms'].append(symptom)
        
        # Extract diagnoses
        for diagnosis in self.medical_terms['diagnoses']:
            if re.search(r'\b' + re.escape(diagnosis) + r'\b', text, re.IGNORECASE):
                structured_data['diagnoses'].append(diagnosis)
        
        # Extract medications
        for medication in self.medical_terms['medications']:
            if re.search(r'\b' + re.escape(medication) + r'\b', text, re.IGNORECASE):
                structured_data['medications'].append(medication)
        
        # Extract procedures
        for procedure in self.medical_terms['procedures']:
            if re.search(r'\b' + re.escape(procedure) + r'\b', text, re.IGNORECASE):
                structured_data['procedures'].append(procedure)
        
        # Extract lab results
        lab_patterns = {
            'troponin': r'troponin[:\s]*(\d+\.?\d*)',
            'glucose': r'glucose[:\s]*(\d+)',
            'creatinine': r'creatinine[:\s]*(\d+\.?\d*)',
            'hemoglobin': r'hemoglobin[:\s]*(\d+\.?\d*)',
            'wbc': r'wbc[:\s]*(\d+,?\d*)'
        }
        
        for lab, pattern in lab_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    value = match.group(1)
                    value = value.replace(',', '')
                    structured_data['lab_results'][lab] = float(value) if '.' in value else int(value)
                except (ValueError, IndexError):
                    pass
        
        return structured_data
    
    def process_document(self, file: Union[bytes, str], file_type: str) -> Dict[str, Any]:
        """Process document (PDF or image) and extract text and structured data."""
        result = {
            'success': False,
            'text': '',
            'structured_data': {},
            'file_type': file_type,
            'error': None
        }
        
        try:
            if file_type.lower() == 'pdf':
                result['text'] = self.extract_text_from_pdf(file)
            elif file_type.lower() in ['jpg', 'jpeg', 'png', 'tiff', 'bmp']:
                result['text'] = self.extract_text_from_image(file)
            else:
                result['error'] = f"Unsupported file type: {file_type}"
                return result
            
            if result['text'].strip():
                result['structured_data'] = self.extract_structured_data(result['text'])
                result['success'] = True
            else:
                result['error'] = "No text could be extracted from the document"
        
        except Exception as e:
            result['error'] = str(e)
        
        return result