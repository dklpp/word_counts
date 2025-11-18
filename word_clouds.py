#!/usr/bin/env python3
"""
word_counts.py

Parse meeting transcripts (.vtt, .srt, .txt), strip timestamps and speaker
labels, count words, and write a word-frequency CSV.

Usage:
  python3 word_count.py /transcripts --out word_counts.csv
"""

import argparse
import csv
import re
from collections import Counter
from pathlib import Path
import sys
from typing import Optional, Set

# --- Timestamp patterns (remove anywhere they appear) ---
# e.g., "00:01:23.456 --> 00:01:25.789", "00:01:23,456 --> 00:01:25,789"
RANGE_VTT = re.compile(r"\b\d{2}:\d{2}:\d{2}[.,]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[.,]\d{3}\b")
# e.g., "00:01:23.456" or "00:01:23,456"
STAMP_HMS_MS = re.compile(r"\b\d{2}:\d{2}:\d{2}[.,]\d{3}\b")
# e.g., "00:01:23" (no ms)
STAMP_HMS = re.compile(r"\b\d{2}:\d{2}:\d{2}\b")
# e.g., "01:23" (mm:ss) often shows up in plain text exports
STAMP_MS = re.compile(r"(?<!\d)\b\d{1,2}:\d{2}\b(?!:\d)")
# e.g., bracketed timestamps like "[00:01:23]" or "[01:23]"
STAMP_BRACKETED = re.compile(r"\[\s*\d{1,2}:\d{2}(?::\d{2})?\s*\]")

# SRT line numbers
LINE_NUMBER_SRT = re.compile(r"^\s*\d+\s*$")

# VTT artifacts
WEBVTT_HEADER = re.compile(r"^\s*WEBVTT\b", re.IGNORECASE)
VTT_CUE_TAG = re.compile(r"</?c(?:\.[^>]+)?>", re.IGNORECASE)

# Speaker labels at the start of a line: "Alice:", "Bob (Host):", "Team A - John:"
SPEAKER_PREFIX = re.compile(r"^\s*[^:]{1,80}:\s+")

# Generic markup/URLs
HTML_TAG = re.compile(r"<[^>]+>")
URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

# Token rules
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
    # common fillers
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
    # Drop headers / SRT numbering / VTT cue tags
    if fmt == "vtt":
        if WEBVTT_HEADER.match(s):
            return ""
        s = VTT_CUE_TAG.sub(" ", s)
    if fmt == "srt" and LINE_NUMBER_SRT.match(s):
        return ""

    # Generic cleanup
    s = URL.sub(" ", s)
    s = HTML_TAG.sub(" ", s)
    s = strip_timestamps_everywhere(s)
    s = SPEAKER_PREFIX.sub("", s)  # remove leading "Name: "
    # Remove parenthetical or bracketed non-speech cues like "(music)" "[laughter]"
    if (s.startswith("(") and s.endswith(")")) or (s.startswith("[") and s.endswith("]")):
        return ""
    return s.strip()

def tokenize(text: str, keep_numbers: bool) -> list[str]:
    text = NON_ALNUM_APOST.sub(" ", text)
    out = []
    for raw in text.split():
        tok = APOST_EDGES.sub("", raw).replace("’", "'")
        tok = tok.replace("'", "")  # collapse don't -> dont
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

def parse_dir(root: Path, args) -> Counter:
    counts = Counter()
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
        for t in tokens:
            if len(t) < args.minlen or t in stop:
                continue
            counts[t] += 1
    return counts

def write_csv(counts: Counter, out_path: Path, top: Optional[int]):
    rows = counts.most_common(top) if top else counts.most_common()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["word", "count"])
        w.writerows(rows)

def main():
    ap = argparse.ArgumentParser(description="Aggregate transcript word counts; ignore timestamps and speaker labels.")
    ap.add_argument("--input", help="Directory with .vtt/.srt/.txt transcripts")
    ap.add_argument("--out", default="word_counts.csv", help="Output CSV path")
    ap.add_argument("--minlen", type=int, default=2, help="Minimum token length")
    ap.add_argument("--top", type=int, default=0, help="Keep only top N words (0 = all)")
    ap.add_argument("--stopwords", default=None, help="Optional newline-delimited stopwords file")
    ap.add_argument("--lower", dest="lower", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--keep-numbers", action="store_true", help="Keep numeric tokens (default: drop)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    root = Path(args.input)
    if not root.exists():
        print(f"Input path not found: {root}", file=sys.stderr)
        sys.exit(1)

    counts = parse_dir(root, args)
    if not counts:
        print("No tokens found. Check your input path and formats.", file=sys.stderr)

    write_csv(counts, Path(args.out), top=(args.top or None))
    if args.verbose:
        total = sum(counts.values())
        print(f"Wrote {args.out} | {total} tokens | {len(counts)} unique", file=sys.stderr)

if __name__ == "__main__":
    main()