import spacy

nlp = spacy.load("en_core_web_sm")

def extract_entities_from_text(text):
    doc = nlp(text)
    ents = []
    for ent in doc.ents:
        ents.append({
            'text': ent.text,
            'label': ent.label_,
            'start': ent.start_char,
            'end': ent.end_char
        })
    return ents

def map_entities_to_wordboxes(entities, word_boxes, full_text):
    words = [w['text'] for w in word_boxes]
    offsets = []
    pos = 0
    reconstructed = ""
    for i, w in enumerate(words):
        if reconstructed == "":
            reconstructed = w
            offsets.append((pos, pos + len(w)))
            pos += len(w)
        else:
            reconstructed += " " + w
            pos += 1
            offsets.append((pos, pos + len(w)))
            pos += len(w)

    mapped = []
    for ent in entities:
        ent_text = ent['text'].strip()
        found_idx = reconstructed.find(ent_text)
        if found_idx == -1:
            found_idx = reconstructed.lower().find(ent_text.lower())

        if found_idx == -1:
            mapped.append({**ent, 'left': None, 'top': None, 'width': None, 'height': None, 'conf': None})
            continue

        start_word = None
        end_word = None
        for i,(s,e) in enumerate(offsets):
            if s <= found_idx < e:
                start_word = i
                break
        end_char = found_idx + len(ent_text) - 1
        for j,(s,e) in enumerate(offsets):
            if s <= end_char < e:
                end_word = j
                break
        if start_word is None: start_word = 0
        if end_word is None: end_word = min(len(words)-1, start_word)

        boxes = word_boxes[start_word:end_word+1]
        if not boxes:
            mapped.append({**ent, 'left': None, 'top': None, 'width': None, 'height': None, 'conf': None})
            continue

        lefts = [b['left'] for b in boxes]
        tops  = [b['top'] for b in boxes]
        rights = [b['left'] + b['width'] for b in boxes]
        bottoms= [b['top'] + b['height'] for b in boxes]
        left = min(lefts)
        top = min(tops)
        right = max(rights)
        bottom = max(bottoms)
        width = right - left
        height = bottom - top
        confs = [b['conf'] for b in boxes if b.get('conf', -1) >= 0]
        avg_conf = sum(confs)/len(confs) if confs else None

        mapped.append({
            'text': ent['text'],
            'label': ent['label'],
            'left': left,
            'top': top,
            'width': width,
            'height': height,
            'conf': avg_conf
        })
    return mapped
