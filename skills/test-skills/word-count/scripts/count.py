#!/usr/bin/env python3
"""Count words in stdin. Exit non-zero on empty input."""
import sys

text = sys.stdin.read().strip()
if not text:
    print("error: empty input", file=sys.stderr)
    sys.exit(1)
print(len(text.split()))
