import re

_JUNK_MARKER = re.compile(
    r'(\bcaption\s*:|\balternative\b|\btext[\- ]to[\- ]image\b|\binput\s*:|\boutput\s*:|-?\s*\bassistant\b|\bthe caption\.|\bnote\s*:)',
    re.IGNORECASE,
)


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
    return s
