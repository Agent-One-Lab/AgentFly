# Skill Infrastructure Spec (v0.1)

This document captures the design decisions made so far for the skill infrastructure that supports inference and RL training. It covers two things:

1. **How skills are formulated** — folder structure, frontmatter, subfolders, conventions.
2. **How skills are discovered and loaded** — system prompt content, tool surface for activation and reading.

Open questions and deferred items are flagged inline as **[TODO]**.

---

## 1. Skill format

### 1.1 A skill is a folder

Each skill is a directory. The directory name doubles as the skill name for the purposes of the file system; the canonical name is declared in frontmatter. The structure:

```
skill-name/
├── SKILL.md          # required
├── scripts/          # optional: bundled executables
├── references/       # optional: docs read on demand
├── assets/           # optional: templates the LLM emits as output
└── tests/            # optional: trigger eval, unit tests, e2e tasks
```

The three subfolders `scripts/`, `references/`, `assets/` correspond to three different roles for content in a skill, and the distinction is load-bearing:

| Subfolder | Role | What the LLM does with it | Token cost |
|---|---|---|---|
| `scripts/` | Deterministic operation | Executes via tool; only output enters context | output only |
| `references/` | Extended guidance | Reads into context on demand | reading cost |
| `assets/` | Output template / raw bytes | Uses as a template or emits verbatim | varies |

`tests/` is not a Claude Skills convention but is part of this spec because static skills need programmatic verifiers for RL training and for regression-proof skill edits.

Skill authors may use other subfolders if they wish, but the four above are the only ones the runtime treats specially.

### 1.2 SKILL.md

The canonical entry point. Structure: YAML frontmatter + free-form markdown body.

The body is a **playbook**: natural-language guidance for the LLM on when to use the skill and what steps to take. Keep it under ~500 lines; push depth into `references/`. The empirical finding from SkillsBench is that focused 2–3-module skills outperform comprehensive ones.

### 1.3 Frontmatter fields

Eight fields, three required, five optional. Keep this minimal; we can grow later.

| Field | Required | Type | Purpose |
|---|---|---|---|
| `name` | yes | string, `[a-z0-9-]{1,64}` | Skill identifier. Used as the handle for invocation. No underscores, no reserved words. |
| `description` | yes | string, ≤1024 chars | The load-bearing trigger string. Third-person, "what + when" pattern. This is what the model sees when deciding whether to load. |
| `version` | yes | semver string | Static-skill cache invalidation. Any content change is a new version. |
| `allowed-tools` | no | list of strings | Whitelist of external tools (with MCP qualification, e.g. `Postgres:explain_query`). The runtime enforces this whitelist while the skill is active. |
| `paths` | no | list of globs | Path-based auto-activation. **[TODO]**: not in v1; revisit later. |
| `disable-model-invocation` | no | bool, default false | If true, the model cannot auto-invoke; only explicit `/skill-name` works. For destructive skills. |
| `model` | no | string | Per-skill model override. Useful for RL training (cheaper models for deterministic skills). |
| `compatibility` | no | object | Environment requirements (Python version, packages, network policy). |

Fields **not** included in this spec, with reasoning:

- `when_to_use`: undocumented in Anthropic's official spec; redundant with `description`. Fold the trigger into the description's "when" clause.
- `context: fork` / `agent`: subagent UI features; not in scope until we have a subagent abstraction.
- `effort`, `shell`, `hooks`, `argument-hint`: Claude Code UI features; defer until requested.
- `license`: skill-level licensing can live in a `LICENSE.txt` file rather than frontmatter.

### 1.4 The three roles for code

Code can appear in a skill in three different ways. The SKILL.md should disambiguate which is which.

1. **Bundled scripts** in `scripts/`. The model invokes them via the runtime; only output enters context. No schema needed; they're skill-internal.
2. **Code shown as documentation** inline in SKILL.md or in `references/*.md`. The model reads it as guidance, not as something to execute. Convention: precede with "Reference:" or put inside a `### Example` heading.
3. **Tools referenced via `allowed-tools`**. Defined elsewhere (as MCP servers). The skill *uses* them but does not *define* them. This preserves the trust boundary at the MCP layer.

**Skills do not define new tools.** The trust model depends on this: tools come from MCP servers (audited at the server boundary); skills are pure context + bundled deterministic scripts. Collapsing the distinction is what produces the malicious-skill attack surface documented in recent literature.

### 1.5 Subfolder conventions in detail

**`scripts/`** — Executable code (Python and Bash dominant). Scripts should:

- Read primary input from stdin when reasonable. This is the single biggest reliability win for small models, since the alternative is the model writing input to a temp file and getting the shell escaping wrong.
- Accept a file path argument as fallback.
- Exit non-zero on failure, with a one-line reason on stderr.
- Print structured output to stdout when the agent will parse it.

**`references/`** — Markdown docs loaded into context only when the SKILL.md body explicitly points to them. Use this for:

- Detailed criteria that don't fit in SKILL.md (e.g. "how to read a Postgres EXPLAIN plan").
- Data the policy needs to consult (e.g. lists of protected tables, code-style rules).
- Domain knowledge that's only relevant for some branches of the workflow.

**`assets/`** — Templates and raw bytes. Things the LLM uses as a *template for output* rather than as guidance. Examples: a report template with `{{placeholder}}` slots, a logo PNG, a docx skeleton. The LLM doesn't "read" these in the guidance sense; it copies, fills, or includes them.

**`tests/`** — Three sub-conventions:

- `tests/trigger_eval.json` — prompts that should/shouldn't activate the skill. Used by description-optimization loops.
- `tests/unit/` — pytest on bundled scripts.
- `tests/e2e/` — SkillsBench-style tasks with deterministic verifiers. Each task has a prompt, an expected skill invocation, and a rubric that the verifier checks against the agent's response.

---

## 2. Discovery and loading

### 2.1 System prompt content

At session start, the runtime builds an `<available_skills>` section in the system prompt containing **names + descriptions** for every installed skill.

The runtime is responsible for budget management. The current budget guidance, following Claude Code's mechanics:

- Aim for total skill listing under ~15K characters.
- Per-skill description capped at 1024 chars by the frontmatter validator.
- Names are always preserved; descriptions are truncated first under budget pressure.

**Scaling regime:** This design is well-suited to libraries up to ~30-50 skills. Past that, the system prompt gets crowded and skill recall degrades. The escape hatch is `search_skills` (see §2.4), which we add only when the problem appears.

**[TODO]** Path-based auto-activation. Deferred for v1. The mechanic — file-glob matching that injects a skill body into a file-read tool result — is a second channel parallel to description routing, useful for codebase-convention skills tied to file types. Revisit when description routing breaks down.

### 2.2 Static skills only

For v1, skills are static. The skill folder is read-only at runtime. Edits require a new `version` and re-installation. This avoids the catastrophic-interference and credit-assignment problems that dynamic skill banks (Voyager, SAGE, D2Skill) introduce.

Two consequences worth keeping in mind:

- The `version` field becomes meaningful. RL rollouts cache by `name@version`; eval results are tagged with version; downstream models trained against skill v0.3.1 should know they were trained against v0.3.1.
- Hooks for SKILL0-style training-time dynamism (skills present during training, withdrawn at inference) remain available, because the *runtime* can decide which skills are loaded — they don't change, but the active set can.

### 2.3 Tool surface — small models

For small models that can't reliably use raw bash, the surface is three tools.

#### `load_skill`

```json
{
  "name": "load_skill",
  "description": "Load a skill into context. Returns the skill's instructions (SKILL.md body) and a manifest of files available in the skill. After loading, use read_skill_file to read reference docs and run_skill_script to execute bundled scripts.",
  "input_schema": {
    "type": "object",
    "properties": {
      "name": {
        "type": "string",
        "description": "The skill name from the available_skills listing. Must match exactly."
      }
    },
    "required": ["name"]
  }
}
```

Tool result:

```json
{
  "name": "sql-query-review",
  "version": "0.3.1",
  "body": "# SQL Query Review\n\n...",
  "files": [
    {"path": "scripts/lint.py", "kind": "script", "description": "Lint a SQL file"},
    {"path": "scripts/safety_check.py", "kind": "script", "description": "Check for unsafe patterns"},
    {"path": "references/plan-review.md", "kind": "reference", "size": 2341},
    {"path": "references/protected-tables.md", "kind": "reference", "size": 891},
    {"path": "assets/review-report.md.template", "kind": "asset", "size": 412}
  ]
}
```

Notes:

- The `files` manifest is always included. Small models don't reliably guess paths; the manifest gives them an enumerated list.
- `kind` tells the model which downstream tool to use: `script` → `run_skill_script`, `reference` → `read_skill_file`, `asset` → typically used as output (read only if needed).
- Per-script `description` is extracted from a docstring or header comment at install time. This makes script selection much more reliable.
- No `base_path` is exposed. The model never needs absolute paths; downstream tools are scoped to the skill.

#### `read_skill_file`

```json
{
  "name": "read_skill_file",
  "description": "Read a file from a loaded skill. Use for reference documentation listed in the skill's file manifest. The skill must be loaded first via load_skill.",
  "input_schema": {
    "type": "object",
    "properties": {
      "skill": {"type": "string", "description": "The skill name (must already be loaded)."},
      "path": {"type": "string", "description": "Relative path within the skill, e.g. 'references/plan-review.md'. Must match an entry in the file manifest."}
    },
    "required": ["skill", "path"]
  }
}
```

Tool result:

```json
{
  "skill": "sql-query-review",
  "path": "references/plan-review.md",
  "content": "...",
  "truncated": false
}
```

Runtime constraints:

- `path` is resolved relative to the skill's base directory. The resolved path must be inside the skill. Reject `..`, absolute paths, and symlinks pointing outside.
- The path must appear in the manifest returned by `load_skill`. Hidden files and unsurfaced files are not readable.
- Cap content at 64KB; set `truncated: true` if exceeded. Pagination is **[TODO]**.
- If the skill isn't loaded, return a specific error pointing at `load_skill`.

#### `run_skill_script`

```json
{
  "name": "run_skill_script",
  "description": "Execute a bundled script from a loaded skill. Args are passed as a list (no shell). Returns stdout, stderr, and exit code. The skill must be loaded first.",
  "input_schema": {
    "type": "object",
    "properties": {
      "skill": {"type": "string"},
      "path": {"type": "string", "description": "Path to the script within the skill, e.g. 'scripts/safety_check.py'. Must match a script entry in the manifest."},
      "args": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Arguments passed to the script as argv. Each item is one argument; do not concatenate or shell-quote.",
        "default": []
      },
      "stdin": {
        "type": "string",
        "description": "Optional input piped to the script's stdin. Use this to pass content (like a SQL query) without writing to a file first.",
        "default": ""
      },
      "timeout_seconds": {
        "type": "integer",
        "default": 30,
        "minimum": 1,
        "maximum": 300
      }
    },
    "required": ["skill", "path"]
  }
}
```

Tool result:

```json
{
  "skill": "sql-query-review",
  "path": "scripts/safety_check.py",
  "exit_code": 1,
  "stdout": "",
  "stderr": "unsafe: write to protected table: ORDERS\n",
  "timed_out": false,
  "truncated": false
}
```

Design rationale:

- **Args as list, not string.** Eliminates the shell-quoting class of failure. The model produces `["foo bar"]` and gets exactly one argv element.
- **Stdin as field.** The model passes content directly rather than writing temp files. This is the biggest small-model reliability win.
- **Structured exit code.** Models reliably read structured fields and unreliably parse stderr for success indicators.
- **No env vars, no working directory, no shell.** All the bash failure modes are absent from the surface.

### 2.4 Tool surface — large models (deferred)

When a model can be trusted with raw bash, the surface collapses to one tool (`load_skill`) plus the agent's existing `view` and `bash`. The migration is mechanical: SKILL.md and bundled scripts don't change; only the *invocation tools* differ. This decoupling lets us run small and large models against the same skill library.

**[TODO]** Decide when and how a session uses the small vs. large surface — probably a runtime config per agent.

### 2.5 Future tools (deferred)

Three plausible extensions, each tied to a specific scaling problem. Add only when the problem appears.

- **`search_skills(query)`** — when the library exceeds ~50 skills and the `<available_skills>` listing no longer fits. Returns ranked descriptions; system prompt switches to names-only.
- **`unload_skill(name)`** — when long trajectories show context pressure from stale loaded skills. Lower priority; runtime-side compaction probably handles this better.
- **`list_skills(category)`** — only if we add categorization to frontmatter. Defer.

### 2.6 Activation injection mechanic

`load_skill` returns content as a normal tool result. The body sits in the conversation history, in the trajectory. This is important for RL: skill activation is a policy decision the model made, and it should be visible in the rollout for credit assignment.

We do **not** inject loaded skills into the system prompt or into a privileged side channel. Everything is in-line tool results, in the trajectory, trainable.

---

## 3. Sandbox and trust

**[TODO]** Full sandbox spec. Current direction:

- Per-skill execution runs in an isolation boundary stronger than the agent process. Microvm (Firecracker) or userspace kernel (gVisor) preferred over containers, given the malicious-skill threat model.
- Agent process (holding credentials, model access, training state) separated from execution sandbox via narrow RPC.
- Network proxy with per-skill allowlists. Default deny; skills declare required destinations in `compatibility.network`.
- Read tools (`read_skill_file`) are also sandboxed via path allowlists; not just script execution.
- Install-time lint: reject invisible Unicode, HTML comments hiding instructions, high-priority instruction overrides. Patterns from the recent malicious-skill literature.

To be expanded in a separate sandbox spec document.

---

## 4. RL training affordances

These are properties of the design above that matter specifically for RL training:

- **Static + versioned** → rollouts are reproducible. Cache key is `(skill_name, version, prompt_seed)`.
- **Explicit invocation via `load_skill`** → skill choice is a tool call, visible in the trajectory, differentiable for credit assignment.
- **Per-skill `model` override** → SKILL0-style curricula can route different skills to different models without changing the skill content.
- **Deterministic verifiers in `tests/e2e/`** → per-skill reward signals separate from end-task reward.
- **Active-skill set is runtime-controlled** → training curricula can withdraw skills progressively (SKILL0) or present subsets (D2Skill) without touching the skill library.

**[TODO]** Skill-conditioned advantage computation, paired skill/no-skill rollouts for utility estimation, on-policy helpfulness curriculum. These belong in a separate RL training spec.

---

## 5. Open questions / TODO summary

A consolidated list of deferred decisions:

1. **Path-based auto-activation** (§2.1). Not in v1. The frontmatter field `paths` is reserved; the runtime ignores it. Revisit when description routing strains.
2. **`search_skills` / `unload_skill` / `list_skills`** (§2.5). Add only when the corresponding scaling problem appears.
3. **Small vs. large model tool surface selection** (§2.4). Likely runtime config; needs a switch.
4. **Sandbox spec** (§3). Separate document. Microvm vs. gVisor tradeoff for RL rollout throughput is the key open question.
5. **Install-time lint policy** (§3). Specific patterns to reject; threat-model coverage.
6. **RL training mechanics** (§4). Separate document.
7. **`view_range` / pagination in `read_skill_file`** (§2.3). Add if reference files routinely exceed 64KB.
8. **Skill categorization / tags** (§2.5). Only if `list_skills` becomes worth adding.

---

## Appendix: example skill

See the `sql-query-review` skill in the example output. It exercises every feature in this spec: all eight frontmatter fields, all four subfolders, all three roles for code, both bundled scripts and MCP tool references via `allowed-tools`, trigger eval, unit tests, and an e2e task with a deterministic verifier.
