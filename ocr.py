import cv2
import numpy as np
from PIL import Image
import pytesseract
from tqdm import tqdm

def preprocess_image_for_ocr(pil_image):
    img = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 15, 9)
    kernel = np.ones((1,1), np.uint8)
    opened = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
    return opened

def ocr_image_to_data(pil_image, lang='eng', config='--oem 3 --psm 1'):
    img_proc = preprocess_image_for_ocr(pil_image)
    data = pytesseract.image_to_data(img_proc, lang=lang, config=config, output_type=pytesseract.Output.DICT)
    words = []
    n = len(data['text'])
    page_text_pieces = []
    for i in range(n):
        txt = data['text'][i].strip()
        conf = int(data['conf'][i]) if data['conf'][i].strip() != '' else -1
        if txt != '':
            words.append({
                'text': txt,
                'conf': conf,
                'left': int(data['left'][i]),
                'top': int(data['top'][i]),
                'width': int(data['width'][i]),
                'height': int(data['height'][i])
            })
            page_text_pieces.append(txt)
    page_text = " ".join(page_text_pieces)
    return words, page_text

def ocr_image_pages(pil_images, lang='eng'):
    results = []
    for img in tqdm(pil_images, desc="OCR pages"):
        w, t = ocr_image_to_data(img, lang=lang)
        results.append({'words': w, 'text': t})
    return results
