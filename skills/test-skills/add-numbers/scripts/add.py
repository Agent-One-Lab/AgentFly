#!/usr/bin/env python3
"""Read two integers from stdin separated by whitespace; print their sum."""
import sys

a, b = sys.stdin.read().split()
print(int(a) + int(b))
