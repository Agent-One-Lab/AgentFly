---
name: country-capital
description: Looks up the capital city of a country from a bundled reference table. Use whenever the user asks "what is the capital of <country>". Always consult the reference; do not answer from memory.
version: 0.1.0
---

# Country Capital Lookup

To answer "what is the capital of <country>":

1. Call `read_skill_file` with:
   - `skill`: "country-capital"
   - `path`: "references/capitals.md"

2. The file contains a table of country → capital pairs. Find the country in the table.

3. Reply with: `The capital of <country> is <capital>.`

If the country is not in the table, reply: `I don't have that country in my reference.` Do not guess.
