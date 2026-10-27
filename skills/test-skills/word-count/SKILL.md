---
name: word-count
description: Counts words in a piece of text using a bundled script. Use whenever the user asks how many words are in some text. Refuses empty input.
version: 0.1.0
---

# Word Count

To count words in text:

1. Call `run_skill_script` with:
   - `skill`: "word-count"
   - `path`: "scripts/count.py"
   - `stdin`: the text to count

2. If the script exits 0, it prints the word count. Report it to the user.

3. If the script exits non-zero, the input was invalid. Read `references/rules.md` for the rules and explain to the user what went wrong.
