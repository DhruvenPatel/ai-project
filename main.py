import argparse
from pathlib import Path
from utils import pdf_to_pil_images, load_image_as_pil
from ocr import ocr_image_pages
from ner import extract_entities_from_text, map_entities_to_wordboxes
import pandas as pd

def process_file(input_path, out_csv, lang='eng'):
    p = Path(input_path)
    if p.suffix.lower() == '.pdf':
        pages = pdf_to_pil_images(str(p))
    else:
        pages = [load_image_as_pil(str(p))]

    ocr_results = ocr_image_pages(pages, lang=lang)
    rows = []
    for page_idx, page_res in enumerate(ocr_results, start=1):
        words = page_res['words']
        text = page_res['text']
        ents = extract_entities_from_text(text)
        mapped = map_entities_to_wordboxes(ents, words, text)
        for m in mapped:
            rows.append({
                'document': p.name,
                'page': page_idx,
                'entity_text': m['text'],
                'label': m['label'],
                'left': m['left'],
                'top': m['top'],
                'width': m['width'],
                'height': m['height'],
                'confidence': m['conf']
            })
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {len(df)} entities to {out_csv}")

def main():
    parser = argparse.ArgumentParser(description="Simple OCR + NER document processor")
    parser.add_argument("input", help="PDF or image input file")
    parser.add_argument("--out", "-o", help="CSV output path", default="output_entities.csv")
    parser.add_argument("--lang", "-l", help="Tesseract language (default 'eng')", default="eng")
    args = parser.parse_args()

    process_file(args.input, args.out, lang=args.lang)

if __name__ == "__main__":
    main()
