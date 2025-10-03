# Trial Abstraction Webapp - Development Log

## Recent Improvements (October 2025)

### OCR-First Extraction Pipeline
- **High-DPI OCR**: Added `_ocr_pdf_text_highdpi()` with 600 DPI and configurable PSM for better small text extraction
- **OCR Preprocessing**: Deskew, denoise, and contrast enhancement using OpenCV and Pillow
- **Force OCR Option**: `force_ocr=True` parameter for testing and scanned documents
- **Diagnostics**: Runtime checks for pdf2image, pytesseract, poppler, tesseract availability

### Author Detection Improvements
- **spaCy NER Integration**: Fixed pydantic v1→v2 compatibility to enable PERSON entity recognition
- **Multi-layer Fallback**: Layout analysis → nearby-title heuristics → spaCy NER → high-DPI OCR retry
- **Quality Filtering**: Vowel ratio checks, gibberish detection, masthead filtering
- **Author Candidates**: Collect multiple candidate names for UI reviewer selection

### Title/Author Heuristics
- **Title Scoring**: Generic scoring function that penalizes boilerplate, footers, and method sentences
- **Masthead Detection**: Filter out journal mastheads (NEJM, Lancet, etc.) from author candidates
- **Name Cleaning**: Handle kerned OCR text, repair "T Homas" → "Thomas", strip leading "by"
- **Layout Analysis**: Use PyMuPDF spans and blocks for structured title/author extraction

### UI Reviewer Enhancements
- **Author Candidate Widget**: Click-to-add buttons for candidate names in custom author input
- **Quality Indicators**: Show `_flags` for OCR/GROBID/LLM usage and table extraction status
- **Table Metadata**: Display `_tables` information when available

### Environment Setup
- **Conda Environment**: `trial-abst` with Python 3.11 and conda-forge packages
- **Binary Dependencies**: poppler (pdftoppm/pdfinfo), tesseract, OpenJDK for tabula/JPype
- **Package Versions**: pydantic 2.11.9, spacy 3.8.7, thinc 8.3.6, en_core_web_sm 3.8.0

### Current Test Results (DASH Study PDF)
```
Title: A Clinical Trial of the Effects of Dietary Patterns
Authors: ['Thomas J Moore']
Flags: {'grobid': False, 'llm': False, 'ocr': True, 'camelot_tables': False}
```

### Next Steps
- [ ] LLM enrichment integration (USE_LLM=true with OpenAI API)
- [ ] Table extraction mapping to arms/outcomes
- [ ] GROBID integration for TEI parsing
- [ ] Database persistence and reviewer workflow
- [ ] Additional PDF test cases and validation

## File Structure
```
backend/
├── extractors.py          # Main extraction pipeline
├── run_extract.py         # Test runner script  
├── main.py               # FastAPI server
├── models.py             # Data models
├── routes_persist.py     # DB persistence routes
├── settings.py           # Configuration
└── data/
    ├── uploads/          # PDF files
    ├── drafts/           # JSON drafts
    └── cache/            # GROBID cache

frontend/
├── src/
│   ├── App.tsx          # Main UI component
│   └── main.tsx         # Entry point
└── package.json         # Dependencies
```