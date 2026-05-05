"""
command_filter.py — Detect and block dangerous or resource-intensive commands.

Usage:
    from command_filter import CommandFilter

    f = CommandFilter()
    ok, reason = f.check("pip install torch")
    # ok=False, reason="Blocked: package installation (pip install) is not allowed"

    # Or use as a shell wrapper:
    #   python command_filter.py "pip install torch"
    #   Exit code 0 = allowed, 1 = blocked (reason printed to stderr)
"""

import re
import sys
import shlex
from dataclasses import dataclass, field


@dataclass
class BlockRule:
    """A single rule that matches and blocks a command pattern."""
    name: str
    pattern: re.Pattern
    reason: str


class CommandFilter:
    """
    Filter that checks shell commands against a set of block rules.

    You can customize by:
      - Adding/removing rules after construction
      - Subclassing and overriding build_rules()
      - Using allow_list for specific exceptions
    """

    def __init__(
        self,
        extra_rules: list[BlockRule] | None = None,
        allow_list: list[str] | None = None,
    ):
        """
        Args:
            extra_rules: Additional BlockRules to add on top of defaults.
            allow_list: List of regex patterns for commands that should always
                        be allowed, even if they match a block rule.
                        E.g., ["pip install numpy"] to allow numpy specifically.
        """
        self.rules: list[BlockRule] = self._build_default_rules()
        if extra_rules:
            self.rules.extend(extra_rules)

        self.allow_patterns: list[re.Pattern] = []
        if allow_list:
            for pattern in allow_list:
                self.allow_patterns.append(
                    re.compile(pattern, re.IGNORECASE)
                )

    def check(self, command: str) -> tuple[bool, str]:
        """
        Check if a command is allowed.

        Args:
            command: The shell command string to check.

        Returns:
            (allowed: bool, reason: str)
            If allowed, reason is empty.
            If blocked, reason explains why.
        """
        command_stripped = command.strip()

        # Check allow list first
        for pattern in self.allow_patterns:
            if pattern.search(command_stripped):
                return True, ""

        # Check each block rule
        for rule in self.rules:
            if rule.pattern.search(command_stripped):
                return False, f"Blocked: {rule.reason}"

        return True, ""

    def filter_command(self, command: str) -> str:
        """
        If the command is allowed, return it unchanged.
        If blocked, return an echo command that prints the reason.
        Useful as a drop-in replacement in a shell pipeline.
        """
        allowed, reason = self.check(command)
        if allowed:
            return command
        else:
            safe_reason = reason.replace("'", "'\\''")
            return f"echo '{safe_reason}' >&2; exit 1"

    def add_rule(self, name: str, pattern: str, reason: str):
        """Add a custom block rule."""
        self.rules.append(BlockRule(
            name=name,
            pattern=re.compile(pattern, re.IGNORECASE),
            reason=reason,
        ))

    def remove_rule(self, name: str):
        """Remove a rule by name."""
        self.rules = [r for r in self.rules if r.name != name]

    # ─── Default rules ──────────────────────────────────────────────────

    @staticmethod
    def _build_default_rules() -> list[BlockRule]:
        return [
            # ── Package installation ────────────────────────────────
            BlockRule(
                name="pip_install",
                pattern=re.compile(
                    r"\bpip[23]?\s+install\b", re.IGNORECASE
                ),
                reason="Package installation (pip install) is not allowed",
            ),
            BlockRule(
                name="pip_download",
                pattern=re.compile(
                    r"\bpip[23]?\s+download\b", re.IGNORECASE
                ),
                reason="Package download (pip download) is not allowed",
            ),
            BlockRule(
                name="conda_install",
                pattern=re.compile(
                    r"\bconda\s+install\b", re.IGNORECASE
                ),
                reason="Package installation (conda install) is not allowed",
            ),
            BlockRule(
                name="apt_install",
                pattern=re.compile(
                    r"\bapt(?:-get)?\s+install\b", re.IGNORECASE
                ),
                reason="Package installation (apt install) is not allowed",
            ),
            BlockRule(
                name="apt_update",
                pattern=re.compile(
                    r"\bapt(?:-get)?\s+update\b", re.IGNORECASE
                ),
                reason="Package update (apt update) is not allowed",
            ),
            BlockRule(
                name="apt_upgrade",
                pattern=re.compile(
                    r"\bapt(?:-get)?\s+upgrade\b", re.IGNORECASE
                ),
                reason="Package upgrade (apt upgrade) is not allowed",
            ),
            BlockRule(
                name="yum_install",
                pattern=re.compile(
                    r"\byum\s+install\b", re.IGNORECASE
                ),
                reason="Package installation (yum install) is not allowed",
            ),
            BlockRule(
                name="dnf_install",
                pattern=re.compile(
                    r"\bdnf\s+install\b", re.IGNORECASE
                ),
                reason="Package installation (dnf install) is not allowed",
            ),
            BlockRule(
                name="npm_install",
                pattern=re.compile(
                    r"\bnpm\s+install\b", re.IGNORECASE
                ),
                reason="Package installation (npm install) is not allowed",
            ),
            BlockRule(
                name="yarn_add",
                pattern=re.compile(
                    r"\byarn\s+add\b", re.IGNORECASE
                ),
                reason="Package installation (yarn add) is not allowed",
            ),

            # ── Git operations (large downloads) ────────────────────
            BlockRule(
                name="git_clone",
                pattern=re.compile(
                    r"\bgit\s+clone\b", re.IGNORECASE
                ),
                reason="Git clone is not allowed (may download large repos)",
            ),
            BlockRule(
                name="git_lfs_pull",
                pattern=re.compile(
                    r"\bgit\s+lfs\s+pull\b", re.IGNORECASE
                ),
                reason="Git LFS pull is not allowed (may download large files)",
            ),

            # ── Large downloads ─────────────────────────────────────
            BlockRule(
                name="wget",
                pattern=re.compile(
                    r"\bwget\s+", re.IGNORECASE
                ),
                reason="Downloading files (wget) is not allowed",
            ),
            BlockRule(
                name="curl_download",
                pattern=re.compile(
                    r"\bcurl\s+.*(-o|-O|--output|--remote-name)\b",
                    re.IGNORECASE,
                ),
                reason="Downloading files (curl -o) is not allowed",
            ),

            # ── Dangerous system commands ───────────────────────────
            BlockRule(
                name="rm_rf_root",
                pattern=re.compile(
                    r"\brm\s+.*-[rR].*\s+/\s*$|"
                    r"\brm\s+.*-[rR].*\s+/[a-z]+\s*$",
                    re.IGNORECASE,
                ),
                reason="Recursive deletion of system directories is not allowed",
            ),
            BlockRule(
                name="fork_bomb",
                pattern=re.compile(
                    r":\(\)\s*\{\s*:\|:\s*&\s*\}\s*;|"
                    r"\.\s*\(\)\s*\{\s*\.\s*\|\s*\.\s*&",
                ),
                reason="Fork bomb detected and blocked",
            ),
            BlockRule(
                name="dd_device",
                pattern=re.compile(
                    r"\bdd\s+.*of=/dev/", re.IGNORECASE
                ),
                reason="Writing to device files (dd of=/dev/) is not allowed",
            ),
            BlockRule(
                name="mkfs",
                pattern=re.compile(
                    r"\bmkfs\b", re.IGNORECASE
                ),
                reason="Filesystem creation (mkfs) is not allowed",
            ),

            # ── Resource-intensive operations ───────────────────────
            BlockRule(
                name="stress_test",
                pattern=re.compile(
                    r"\b(stress|stress-ng|memtester)\b", re.IGNORECASE
                ),
                reason="Stress testing tools are not allowed",
            ),
            BlockRule(
                name="crypto_mining",
                pattern=re.compile(
                    r"\b(xmrig|minerd|cgminer|bfgminer|cpuminer)\b",
                    re.IGNORECASE,
                ),
                reason="Crypto mining tools are not allowed",
            ),

            # ── Python memory-heavy patterns ────────────────────────
            BlockRule(
                name="python_huggingface_download",
                pattern=re.compile(
                    r"from_pretrained\(|snapshot_download\(|hf_hub_download\(",
                    re.IGNORECASE,
                ),
                reason="Downloading models (from_pretrained/snapshot_download) is not allowed",
            ),

            # ── Docker/container escape ─────────────────────────────
            BlockRule(
                name="docker_run",
                pattern=re.compile(
                    r"\bdocker\s+(run|pull)\b", re.IGNORECASE
                ),
                reason="Running nested containers (docker) is not allowed",
            ),
            BlockRule(
                name="nohup_background",
                pattern=re.compile(
                    r"\bnohup\s+.*&\s*$", re.IGNORECASE
                ),
                reason="Background persistent processes (nohup) are not allowed",
            ),
        ]


# ─── CLI usage ──────────────────────────────────────────────────────────────

def main():
    """
    CLI: python command_filter.py "<command>"
    Exit 0 if allowed, exit 1 if blocked.
    """
    if len(sys.argv) < 2:
        print("Usage: python command_filter.py '<command>'", file=sys.stderr)
        sys.exit(2)

    command = " ".join(sys.argv[1:])
    f = CommandFilter()
    allowed, reason = f.check(command)

    if allowed:
        print(f"ALLOWED: {command}")
        sys.exit(0)
    else:
        print(f"{reason}: {command}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
