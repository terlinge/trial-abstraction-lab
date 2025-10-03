# Environment Setup Instructions

## Prerequisites
- Miniforge/Mambaforge conda distribution
- Git for version control
- Node.js 16+ for frontend development

## Backend Setup (Python)

### 1. Create Conda Environment
```bash
conda create -n trial-abst python=3.11 -c conda-forge
conda activate trial-abst
```

### 2. Install Binary Dependencies
```bash
# PDF processing and OCR
conda install -c conda-forge poppler tesseract pillow pdf2image pytesseract

# Table extraction 
conda install -c conda-forge openjdk jpype1 tabula-py camelot-py[cv]

# Web framework and database
conda install -c conda-forge uvicorn fastapi sqlalchemy psycopg2-binary

# PDF libraries
conda install -c conda-forge pymupdf pdfplumber
```

### 3. Install Python NLP
```bash
# Fix pydantic compatibility for spaCy
pip install --upgrade --force-reinstall "pydantic>=2.0"

# Install spaCy and English model
python -m spacy download en_core_web_sm
```

### 4. Verify Installation
```bash
python -c "
import fitz, spacy, pdf2image, pytesseract, camelot, tabula
print('PDF:', fitz.__version__)
print('spaCy:', spacy.__version__)
nlp = spacy.load('en_core_web_sm')
print('Model loaded successfully')
"
```

## Frontend Setup (Node.js)

### 1. Install Dependencies
```bash
cd frontend
npm install
```

### 2. Start Development Server
```bash
npm run dev -- --host 127.0.0.1 --port 5173
```

## Backend Server

### 1. Start FastAPI Server
```bash
conda activate trial-abst
cd backend
python -m uvicorn main:app --host 127.0.0.1 --port 8001 --reload
```

### 2. Test Extraction
```bash
python -u run_extract.py
```

## Environment Variables
Create `.env` file in backend directory:
```bash
USE_LLM=false
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o-mini
GROBID_URL=http://127.0.0.1:8070
```

## Common Issues

### spaCy Import Error
```
ImportError: cannot import name 'GetCoreSchemaHandler' from 'pydantic'
```
**Solution**: `pip install --force-reinstall "pydantic>=2.0"`

### pdf2image Error
```
PDFInfoNotInstalledError: Unable to get page count. Is poppler installed and in PATH?
```
**Solution**: `conda install -c conda-forge poppler`

### Vite IPv6 Binding Issue
```
ERR_ADDRESS_UNREACHABLE at http://[::1]:5173
```
**Solution**: Use `--host 127.0.0.1` flag when starting Vite

### JPype Missing (tabula fallback)
```
Failed to import jpype dependencies. Fallback to subprocess.
```
**Solution**: `conda install -c conda-forge jpype1 openjdk`