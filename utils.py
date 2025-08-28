from pdf2image import convert_from_path
from PIL import Image
import os

def pdf_to_pil_images(pdf_path, dpi=300, first_page=None, last_page=None):
    pages = convert_from_path(pdf_path, dpi=dpi, first_page=first_page, last_page=last_page)
    return pages

def load_image_as_pil(path):
    return Image.open(path).convert("RGB")

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
