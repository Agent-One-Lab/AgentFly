---
name: add-numbers
description: Adds two integers using a bundled calculator script. Use whenever the user asks to add two numbers, compute a sum, or do "X plus Y". Always use the script — do not compute the sum yourself.
version: 0.1.0
---

# Add Numbers

To add two numbers `a` and `b`:

1. Call `run_skill_script` with:
   - `skill`: "add-numbers"
   - `path`: "scripts/add.py"
   - `stdin`: "<a> <b>" (the two numbers separated by a space)

2. The script will print the sum on stdout. Report that sum to the user.

Example: if the user asks "what is 3 plus 5", send stdin `"3 5"`. The script will print `8`. Reply: `3 + 5 = 8`.

Do not add the numbers yourself. Always use the script.
