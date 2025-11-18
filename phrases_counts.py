#!/usr/bin/env python3
"""
word_counts.py

Parse meeting transcripts (.vtt, .srt, .txt), strip timestamps and speaker
labels, count words and N-grams, and write frequency CSVs.

Usage:
  python3 word_counts.py --input transcripts --out word_counts.csv --ngrams 2 3
"""

import argparse
import csv
import re
from collections import Counter
from pathlib import Path
import sys
from typing import Optional, Set, Dict, List
from itertools import islice

# --- Timestamp patterns ---
RANGE_VTT = re.compile(r"\b\d{2}:\d{2}:\d{2}[.,]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[.,]\d{3}\b")
STAMP_HMS_MS = re.compile(r"\b\d{2}:\d{2}:\d{2}[.,]\d{3}\b")
STAMP_HMS = re.compile(r"\b\d{2}:\d{2}:\d{2}\b")
STAMP_MS = re.compile(r"(?<!\d)\b\d{1,2}:\d{2}\b(?!:\d)")
STAMP_BRACKETED = re.compile(r"\[\s*\d{1,2}:\d{2}(?::\d{2})?\s*\]")

LINE_NUMBER_SRT = re.compile(r"^\s*\d+\s*$")
WEBVTT_HEADER = re.compile(r"^\s*WEBVTT\b", re.IGNORECASE)
VTT_CUE_TAG = re.compile(r"</?c(?:\.[^>]+)?>", re.IGNORECASE)

SPEAKER_PREFIX = re.compile(r"^\s*[^:]{1,80}:\s+")
HTML_TAG = re.compile(r"<[^>]+>")
URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

NON_ALNUM_APOST = re.compile(r"[^a-zA-Z0-9'\-]+")
APOST_EDGES = re.compile(r"(^'+|'+$)")

DEFAULT_STOPWORDS = {
    "a","an","and","the","or","but","if","then","than","that","this","those","these","there","here","when","where",
    "why","how","what","which","who","whom","whose","with","without","for","from","to","in","into","on","onto",
    "of","at","by","be","is","am","are","was","were","been","being","as","do","does","did","done","doing","have",
    "has","had","having","i","me","my","myself","we","our","ours","ourselves","you","your","yours","yourself",
    "yourselves","he","him","his","himself","she","her","hers","herself","it","its","itself","they","them","their",
    "theirs","themselves","all","any","both","each","few","more","most","other","some","such","no","nor","not",
    "only","own","same","so","too","very","can","could","should","would","may","might","will","just","also",
    "um","uh","erm","hmm","like","kinda","sorta","yeah","right","okay","ok","alright"
}

def guess_fmt(p: Path) -> str:
    ext = p.suffix.lower()
    if ext == ".vtt": return "vtt"
    if ext == ".srt": return "srt"
    return "txt"

def strip_timestamps_everywhere(s: str) -> str:
    s = RANGE_VTT.sub(" ", s)
    s = STAMP_BRACKETED.sub(" ", s)
    s = STAMP_HMS_MS.sub(" ", s)
    s = STAMP_HMS.sub(" ", s)
    s = STAMP_MS.sub(" ", s)
    return s

def clean_line(line: str, fmt: str) -> str:
    s = line.rstrip("\r\n")
    if not s:
        return ""
    if fmt == "vtt":
        if WEBVTT_HEADER.match(s):
            return ""
        s = VTT_CUE_TAG.sub(" ", s)
    if fmt == "srt" and LINE_NUMBER_SRT.match(s):
        return ""

    s = URL.sub(" ", s)
    s = HTML_TAG.sub(" ", s)
    s = strip_timestamps_everywhere(s)
    s = SPEAKER_PREFIX.sub("", s)

    if (s.startswith("(") and s.endswith(")")) or (s.startswith("[") and s.endswith("]")):
        return ""
    return s.strip()

def tokenize(text: str, keep_numbers: bool) -> List[str]:
    text = NON_ALNUM_APOST.sub(" ", text)
    out = []
    for raw in text.split():
        tok = APOST_EDGES.sub("", raw).replace("’", "'")
        tok = tok.replace("'", "")
        if not keep_numbers and tok.isdigit():
            continue
        if tok:
            out.append(tok)
    return out

def load_stopwords(path: Optional[str]) -> Set[str]:
    sw = set(DEFAULT_STOPWORDS)
    if path:
        p = Path(path)
        if p.is_file():
            sw |= {ln.strip().lower() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()}
    return sw

def iter_files(root: Path):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".vtt", ".srt", ".txt"}:
            yield p

# --- N-GRAM GENERATOR ---
def make_ngrams(tokens: List[str], n: int):
    if len(tokens) < n:
        return []
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

# --- MAIN PARSER ---
def parse_dir(root: Path, args):
    word_counts = Counter()
    ngram_counts = {n: Counter() for n in args.ngrams}

    stop = load_stopwords(args.stopwords)

    for fp in iter_files(root):
        if args.verbose:
            print(f"Processing: {fp}", file=sys.stderr)

        fmt = guess_fmt(fp)
        try:
            text = fp.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            text = fp.read_text(encoding="latin-1", errors="ignore")

        lines = (clean_line(ln, fmt) for ln in text.splitlines())
        cleaned = " ".join(l for l in lines if l)

        if args.lower:
            cleaned = cleaned.lower()

        tokens = tokenize(cleaned, keep_numbers=args.keep_numbers)
        tokens = [t for t in tokens if len(t) >= args.minlen and t not in stop]

        word_counts.update(tokens)

        # Count N-grams
        for n in args.ngrams:
            ngrams = make_ngrams(tokens, n)
            ngram_counts[n].update(ngrams)

    return word_counts, ngram_counts

# --- CSV WRITER ---
def write_csv(counts: Counter, out_path: Path, top: Optional[int]):
    rows = counts.most_common(top) if top else counts.most_common()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["phrase", "count"])
        w.writerows(rows)

# --- MAIN ---
def main():
    ap = argparse.ArgumentParser(description="Aggregate transcript word & N-gram counts.")
    ap.add_argument("--input", required=True, help="Directory with transcripts")
    ap.add_argument("--out", default="word_counts.csv", help="Output CSV (base name)")
    ap.add_argument("--minlen", type=int, default=2, help="Minimum token length")
    ap.add_argument("--top", type=int, default=0, help="Top N phrases (0 = all)")
    ap.add_argument("--stopwords", default=None, help="Optional stopwords file")
    ap.add_argument("--lower", dest="lower", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--keep-numbers", action="store_true")
    ap.add_argument("--verbose", action="store_true")

    # NEW FLAG: list of N-gram sizes
    ap.add_argument("--ngrams", nargs="+", type=int, default=[],
                    help="N-gram sizes to count, e.g., --ngrams 2 3")

    args = ap.parse_args()

    root = Path(args.input)
    if not root.exists():
        print(f"Input path not found: {root}", file=sys.stderr)
        sys.exit(1)

    word_counts, ngram_counts = parse_dir(root, args)

    # Write unigram CSV
    write_csv(word_counts, Path(args.out), top=(args.top or None))

    # Write each N-gram CSV
    for n, counter in ngram_counts.items():
        out_path = Path(args.out).with_name(
            f"{Path(args.out).stem}_ngram{n}.csv"
        )
        write_csv(counter, out_path, top=(args.top or None))

    if args.verbose:
        print(f"Wrote unigrams → {args.out}", file=sys.stderr)
        for n in args.ngrams:
            print(f"Wrote {n}-grams → {Path(args.out).stem}_ngram{n}.csv", file=sys.stderr)

if __name__ == "__main__":
    main()
