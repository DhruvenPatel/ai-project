# Mini Document Processing System

This is a simplified AI-powered document processor implementing OCR (Tesseract) + NER (spaCy) + CSV export.

## Setup
1. Install system dependencies (Tesseract OCR, poppler).
2. Create virtual environment & install requirements:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Usage
```bash
python main.py input.pdf --out output.csv
```

Works with PDFs and images.
