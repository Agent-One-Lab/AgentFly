# Test Skills

Four minimal skills for bringing up skill infrastructure on small models. Each one tests **one mechanism** of the three-tool surface so failures can be localized.

## Run order

Install and test in this order. Each skill adds exactly one new mechanism on top of the previous.

| # | Skill | Tests | If it fails, debug... |
|---|---|---|---|
| 1 | `hello-skill` | `load_skill` round-trip | Description listing, tool definition, body delivery |
| 2 | `add-numbers` | `run_skill_script` end-to-end | stdin plumbing, stdout return, model trusting the script |
| 3 | `country-capital` | `read_skill_file` end-to-end | Path resolution, content delivery, file-vs-memory behavior |
| 4 | `word-count` | All three tools + exit-code branching | Manifest navigation, error-path handling |

If skill N fails but skill N-1 works, the bug is in the new mechanism that skill N introduces.

## What each skill catches

**`hello-skill`** — Zero scripts, zero references. Just SKILL.md. If this doesn't work, something is wrong before any tool surface complexity is involved.

**`add-numbers`** — The SKILL.md emphasizes "do not add the numbers yourself" precisely so the verifier can check whether `run_skill_script` was actually called. A model that computes the sum mentally but doesn't call the script will fail the `must_invoke_tool` check.

**`country-capital`** — The Canberra/Sydney trap. Small models love to say "Sydney" from memory. If the model says Canberra, you know the file was actually read.

**`word-count`** — Tests exit-code branching: empty input exits non-zero, and SKILL.md says to consult `references/rules.md` on error. The verifier checks that all three tools (`load_skill`, `run_skill_script`, `read_skill_file`) were invoked on the empty-input path.

## Each skill's tests folder

Each skill has:

- `tests/trigger_eval.json` — 3 should-fire and 3 shouldn't-fire prompts. If the should-fire prompts don't trigger the skill, the description needs more keywords. If the shouldn't-fire prompts incorrectly trigger it, the description is too broad.
- `tests/e2e/task_*.json` — End-to-end tasks with deterministic verifiers. Each one has a prompt, an expected skill, and a rubric checking output content + tool invocations.

## Verifier rubric format

```json
{
  "must_contain_any": ["string1", "string2"],
  "must_not_contain": ["bad1", "bad2"],
  "must_invoke_tool": ["load_skill", "run_skill_script"]
}
```

- `must_contain_any`: response must contain at least one of these strings.
- `must_not_contain`: response must not contain any of these.
- `must_invoke_tool`: trajectory must include calls to all listed tools.

## Folder layout

```
test-skills/
├── hello-skill/
│   ├── SKILL.md
│   └── tests/
│       ├── trigger_eval.json
│       └── e2e/task_001_greeting.json
├── add-numbers/
│   ├── SKILL.md
│   ├── scripts/add.py
│   └── tests/
│       ├── trigger_eval.json
│       └── e2e/task_001_simple_add.json
├── country-capital/
│   ├── SKILL.md
│   ├── references/capitals.md
│   └── tests/
│       ├── trigger_eval.json
│       └── e2e/
│           ├── task_001_canberra_trap.json
│           └── task_002_missing_country.json
└── word-count/
    ├── SKILL.md
    ├── scripts/count.py
    ├── references/rules.md
    └── tests/
        ├── trigger_eval.json
        └── e2e/
            ├── task_001_normal.json
            └── task_002_empty.json
```
