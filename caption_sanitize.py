import re

_JUNK_MARKER = re.compile(
    r'(\bcaption\s*:|\balternative\b|\btext[\- ]to[\- ]image\b|\binput\s*:|\boutput\s*:|-?\s*\bassistant\b|\bthe caption\.|\bnote\s*:)',
    re.IGNORECASE,
)
_WATERMARK = re.compile(
    r'(\bwatermark\b|\blogo\b|\bsignature\b|\bbranding\b|brand\s+name|copyright|\b[\w-]+\.com\b|website\s+name|text\s+overlay)',
    re.IGNORECASE,
)
_ABSENCE = re.compile(
    r'(\bnot\s+visible\b|\bnot\s+shown\b|\bnot\s+depicted\b|\bnot\s+in\s+view\b|cannot\s+be\s+seen|can.?t\s+be\s+seen|is\s+not\s+visible|are\s+not\s+visible|isn.?t\s+visible|aren.?t\s+visible|no\s+genitals\b|genitals\s+are\s+not\b|no\s+visible\s+genital|are\s+absent\b|is\s+absent\b|none\s+(?:are|is)\s+visible|no\s+other\s+(?:people|person|objects|one)\b|nothing\s+else\s+is\s+(?:visible|present))',
    re.IGNORECASE,
)
_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')


def sanitize_caption(text):
    s = (text or "").strip()
    marker = _JUNK_MARKER.search(s)
    if marker:
        s = s[:marker.start()].strip()
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', s) if p.strip()]
    if len(paragraphs) > 1:
        head = ' '.join(paragraphs[0].lower().split()[:6])
        kept = [paragraphs[0]]
        for para in paragraphs[1:]:
            if ' '.join(para.lower().split()[:6]) == head:
                break
            kept.append(para)
        s = ' '.join(kept)
    s = re.sub(r'\s*\n+\s*', ' ', s)
    s = re.sub(r'\s{2,}', ' ', s).strip()
    sentences = [
        sent for sent in _SENT_SPLIT.split(s)
        if sent and not _WATERMARK.search(sent) and not _ABSENCE.search(sent)
    ]
    if len(sentences) > 1 and not re.search(r'[.!?]["\')\]]?$', sentences[-1].strip()):
        sentences = sentences[:-1]
    seen = set()
    deduped = []
    for sent in sentences:
        norm = ' '.join(re.sub(r'[^a-z0-9 ]', '', sent.lower()).split())
        if len(norm.split()) >= 5 and norm in seen:
            break
        seen.add(norm)
        deduped.append(sent)
    s = ' '.join(deduped)
    s = re.sub(r'\s{2,}', ' ', s).strip()
    return s
