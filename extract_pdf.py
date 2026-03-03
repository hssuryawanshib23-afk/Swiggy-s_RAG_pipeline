import os
import json
import io
import argparse
from tqdm import tqdm

import pdfplumber
import fitz  # PyMuPDF
import pytesseract
from PIL import Image

# Set Tesseract path for Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def clean_text(text):
    if not text:
        return ""
    # Basic cleaning
    return text.strip()

def extract_pdf(pdf_path, output_path, start_page=1, end_page=None):
    if not os.path.exists(os.path.dirname(output_path)) and os.path.dirname(output_path) != "":
        os.makedirs(os.path.dirname(output_path))

    extracted_pages = []
    
    print(f"Opening {pdf_path}...")
    doc_fitz = fitz.open(pdf_path)
    total_pages = len(doc_fitz)
    
    if end_page is None or end_page > total_pages:
        end_page = total_pages
        
    print(f"Processing pages {start_page} to {end_page} out of {total_pages} total pages.")

    with pdfplumber.open(pdf_path) as pdf:
        for i in tqdm(range(start_page - 1, end_page), desc="Extracting"):
            page_num = i + 1
            page = pdf.pages[i]
            
            # 1. Try selectable text first
            text = clean_text(page.extract_text())
            content_type = "text"
            
            # 2. Check if scanned (less than 50 chars of selectable text)
            if len(text) < 50:
                content_type = "ocr_text"
                
                # Render page to image with PyMuPDF
                fitz_page = doc_fitz[i]
                pix = fitz_page.get_pixmap(dpi=300) # High DPI for OCR
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                
                # Run OCR
                text = clean_text(pytesseract.image_to_string(img))
                
            # 3. Extract tables
            # Using text based vertical strategy for borderless tables
            table_settings = {
                "vertical_strategy": "text", 
                "horizontal_strategy": "lines"
            }
            tables = page.extract_tables(table_settings)
            
            # Format tables slightly better
            markdown_tables = []
            if tables:
                for table in tables:
                    cleaned_table = []
                    for row in table:
                        cleaned_row = [str(cell).replace('\n', ' ').strip() if cell is not None else "" for cell in row]
                        cleaned_table.append(" | ".join(cleaned_row))
                    markdown_tables.append("\n".join(cleaned_table))

            page_data = {
                "page": page_num,
                "type": content_type,
                "text": text,
                "tables": markdown_tables
            }
            extracted_pages.append(page_data)
            
    doc_fitz.close()
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(extracted_pages, f, ensure_ascii=False, indent=2)
        
    print(f"\nExtraction complete! Saved {len(extracted_pages)} pages to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract text and tables from PDF (hybrid pdfplumber + OCR)")
    parser.add_argument("pdf_path", help="Path to input PDF")
    parser.add_argument("output_path", help="Path to output JSON")
    parser.add_argument("--start", type=int, default=1, help="Start page")
    parser.add_argument("--end", type=int, default=None, help="End page")
    
    args = parser.parse_args()
    extract_pdf(args.pdf_path, args.output_path, args.start, args.end)
