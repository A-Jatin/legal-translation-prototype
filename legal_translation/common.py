"""Shared file contracts; ML imports stay out of preparation and saved scoring."""
import hashlib
import json
import unicodedata
from pathlib import Path


def normalize(text):
    text = unicodedata.normalize('NFC', text.removeprefix('\ufeff').replace('\r\n', '\n').replace('\r', '\n'))
    return '\n'.join(' '.join(line.split()) for line in text.split('\n')).strip()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in rows), encoding='utf-8')


def unique(rows):
    indexed = {r['pair_id']: r for r in rows}
    if len(indexed) != len(rows):
        raise ValueError('Duplicate pair IDs')
    return indexed
