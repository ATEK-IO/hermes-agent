"""Refuse a recursive tree walk over a whole machine (``grep -r`` / ``find``).

``approvals.deny`` cannot express this rule. fnmatch sees one flat string, so a
rule that names a root either misses every other root (``grep *-*[rr]* /home/hr/ *``
does not match ``... /home/filipe/ ...``) or, widened to ``<root>*``, also
matches the bounded form ``grep -rn x /home/hr/agent-roots/scripts | head``
because the pipe supplies the trailing space. The rule is therefore enforced on
the parsed command instead of on the text:

* every absolute path argument is a tree root;
* a recursive ``grep`` (or ``find``, which is always recursive) is refused when a
  root is ``/``, ``$HOME``/``~`` or a home directory itself, or when two or more
  arguments are directory roots (several roots in one call).

Why this exists: ATE-18741 measured a ``grep -rl`` over five roots that ran
5.2 h at load 37 and passed every deny rule, because each rule named one literal
root. Bounded single-root forms stay allowed —
``grep -rn x /home/hr/agent-roots/scripts | head`` and
``find /home/hr/agent-roots -name '*.md'`` are the documented way to search.

The walk is matched on the same deobfuscated command variants the deny matcher
uses (``tools.approval_detection._deny_command_variants``), so ``sudo``, ``env``,
``bash -c '...'`` and an executable given by path do not hide it; a quoted
payload (``ssh host '...'``, ``eval``) and an unquoted ssh command are scanned
one level deeper.
"""

import os
import posixpath
import re

from tools.approval_detection import (
    _deny_command_variants,
    _iter_top_level_shell_segments,
    _shell_segment_tokens,
)
from tools.shell_heredoc import strip_inert_heredoc_bodies

#: Searchers whose positional arguments are tree roots, by recursion behaviour.
#: ``rg``/``ag``/``ack`` walk a tree with no flag, so leaving them out would let
#: the same walk through under another binary name.
_GREP_FAMILY = frozenset({"grep", "egrep", "fgrep", "rgrep", "zgrep"})
_RECURSIVE_BY_DEFAULT = frozenset({"rg", "ag", "ack", "ack-grep"})
_FIND = "find"

#: Directory names whose direct children are human home directories.
_HOME_PARENTS = frozenset({"home", "Users"})

#: ``ssh`` options that consume the next token, so its value is not read as the host.
_SSH_OPTIONS_WITH_ARG = frozenset({
    "-b", "-c", "-D", "-e", "-F", "-i", "-J", "-L", "-l", "-m", "-O", "-o",
    "-p", "-P", "-Q", "-R", "-S", "-W", "-w",
})
_SSH_NAMES = frozenset({"ssh", "mosh", "rsh"})

#: grep short options that consume the next token (the subset that shifts operands).
_GREP_SHORT_OPTIONS_WITH_ARG = frozenset({"A", "B", "C", "D", "d", "e", "f", "m"})
#: grep long options that consume the next token.
_GREP_OPTIONS_WITH_ARG = frozenset({
    "--after-context", "--before-context", "--binary-files", "--context",
    "--devices", "--directories", "--exclude", "--exclude-dir", "--exclude-from",
    "--file", "--include", "--label", "--max-count", "--regexp",
})

#: How deep a nested payload (ssh inside bash -c inside ...) is still scanned.
_MAX_NESTING = 4

#: Cheap prefilter for a nested payload: only a token that names a searcher can
#: start a whole-machine walk.
_NESTED_SCAN_RE = re.compile(
    r"(?:^|[\s;&|(`])(?:grep|egrep|fgrep|rgrep|zgrep|find|rg|ag|ack)\b",
    re.IGNORECASE,
)

_HOME_VARIABLE_RE = re.compile(r"^\$\{?home\}?", re.IGNORECASE)


def _searcher_name_re(name: str) -> str:
    """A searcher name that tolerates shell quoting between its letters.

    ``grep``, ``gr""ep`` and ``g\\rep`` are the same word to the shell, and one
    character class between each letter is cheaper than deobfuscating the whole
    command before deciding whether this guard has anything to look at.
    """
    gap = r"[\"'\\]*"
    return gap.join(re.escape(char) for char in name)


#: Cheap precheck: a command that cannot name a searcher cannot be a scan. This
#: runs on every terminal command, so it stays a single regex pass — the full
#: deobfuscating variant walk costs tens of milliseconds.
_MAYBE_SCAN_RE = re.compile(
    r"(?<![a-z])(?:" + "|".join(
        _searcher_name_re(name)
        for name in ("grep", "egrep", "fgrep", "rgrep", "zgrep", "find", "rg", "ag", "ack")
    ) + r")(?![a-z])",
    re.IGNORECASE,
)


def recursive_scan_reason(command: str) -> str | None:
    """Return a one-line description of the whole-machine walk in ``command``.

    ``None`` means no recursive search over a whole tree was found. The returned
    string names the executable and the roots, and is used only in the block
    message.
    """
    if not isinstance(command, str) or not command.strip():
        return None
    if not _MAYBE_SCAN_RE.search(command):
        return None
    # A quoted-delimiter heredoc body is inert data: a comment or a document that
    # quotes the scan must stay writable (the same masking the background guard uses).
    return _scan(strip_inert_heredoc_bodies(command), 0)


def recursive_scan_block_result(detail: str) -> dict:
    """Build the unconditional block result for a whole-machine scan."""
    return {"approved": False, "hardline": True, "recursive_scan": True, "message": (
        f"BLOCKED: recursive scan of a whole machine tree ({detail}). A recursive grep/find "
        "rooted at /, at a home directory itself, or at two or more roots at once walks every "
        "file on the host — the scan this guard exists for ran 5.2 h at load 37 "
        "(ATE-18741). Scope the search to one bounded folder instead: "
        "grep -rn PATTERN /path/to/subdir | head, or "
        "find /path/to/subdir -name PATTERN. If the wide scan is really wanted, a person runs "
        "it in a terminal outside the agent. Do NOT retry the same roots one at a time to get "
        "around this guard.")}


def _scan(command: str, depth: int) -> str | None:
    if depth > _MAX_NESTING:
        return None
    for variant in _deny_command_variants(command):
        if not variant:
            continue
        detail = _scan_variant(variant)
        if detail:
            return detail
    for payload in _nested_commands(command):
        detail = _scan(payload, depth + 1)
        if detail:
            return detail
    return None


def _scan_variant(variant: str) -> str | None:
    """Analyse every command segment of one deny-matcher variant."""
    for segment in _iter_top_level_shell_segments(variant):
        tokens = _shell_segment_tokens(segment, 0)
        if not tokens:
            continue
        detail = _scan_tokens(tokens)
        if detail:
            return detail
    return None


def _scan_tokens(tokens: list[str]) -> str | None:
    name = _executable_name(tokens[0])
    if name == _FIND:
        return _judge(_FIND, _find_roots(tokens[1:]))
    if name in _RECURSIVE_BY_DEFAULT:
        recursive, roots = _grep_operands(tokens[1:], default_recursive=True)
        return _judge(name, roots) if recursive else None
    if name in _GREP_FAMILY:
        recursive, roots = _grep_operands(tokens[1:], default_recursive=False)
        return _judge(name, roots) if recursive else None
    return None


def _executable_name(token: str) -> str:
    return os.path.basename(token).lower()


def _grep_operands(args: list[str], *, default_recursive: bool) -> tuple[bool, list[str]]:
    """Return (recursive, positional operands that are not the pattern)."""
    recursive = default_recursive
    operands: list[str] = []
    explicit_pattern = False
    options = True
    i = 0
    while i < len(args):
        token = args[i]
        if options and token == "--":
            options = False
            i += 1
            continue
        if options and token.startswith("--"):
            option, equals, _ = token.partition("=")
            lowered = option.lower()
            if lowered in {"--recursive", "--dereference-recursive"}:
                recursive = True
            elif lowered == "--directories":
                value = args[i + 1] if not equals and i + 1 < len(args) else ""
                recursive = recursive or value.lower().startswith("recurse")
                if not equals:
                    i += 1
            elif lowered in {"--regexp", "--file"}:
                explicit_pattern = True
                if not equals:
                    i += 1
            elif lowered in _GREP_OPTIONS_WITH_ARG and not equals:
                i += 1
            i += 1
            continue
        if options and token.startswith("-") and token != "-":
            chars = token[1:]
            for offset, char in enumerate(chars):
                if char in "rR":
                    recursive = True
                elif char in {"e", "f"}:
                    explicit_pattern = True
                if char not in _GREP_SHORT_OPTIONS_WITH_ARG:
                    continue
                attached = chars[offset + 1:]
                if char == "d":
                    value = attached or (args[i + 1] if i + 1 < len(args) else "")
                    recursive = recursive or value.lower().startswith("recurse")
                if not attached:
                    # The value is the next token; the rest of this bundle is exhausted.
                    i += 1
                break
            i += 1
            continue
        operands.append(token)
        i += 1
    if operands and not explicit_pattern:
        operands = operands[1:]  # the first positional argument is the pattern
    return recursive, operands


def _find_roots(args: list[str]) -> list[str]:
    """Return find's leading path arguments; the expression starts at the first operator."""
    roots: list[str] = []
    for token in args:
        if token.startswith("-") or token in {"(", ")", "!", ",", "{"}:
            break
        roots.append(token)
    return roots


def _judge(name: str, roots: list[str]) -> str | None:
    if not roots:
        return None
    directories: list[str] = []
    for raw in roots:
        info = _root_info(raw)
        if info is None:
            continue  # a relative path is not a tree root for this rule
        normalized, is_home, is_directory = info
        if is_home:
            shown = roots if len(roots) <= 5 else [*roots[:5], "…"]
            return f"{name} over {_describe_root(normalized)} — {' '.join(shown)}"
        if is_directory:
            directories.append(raw)
    if len(directories) >= 2:
        shown = directories if len(directories) <= 5 else [*directories[:5], "…"]
        return f"{name} over {len(directories)} roots — {' '.join(shown)}"
    return None


def _describe_root(normalized: str) -> str:
    return "the filesystem root /" if normalized == "/" else f"the home directory {normalized}"


def _root_info(raw: str) -> tuple[str, bool, bool] | None:
    """Return (normalized path, is a home directory itself, is directory-like).

    ``None`` for a relative path: only absolute arguments are tree roots here.
    """
    absolute = _absolute_form(raw)
    if absolute is None:
        return None
    normalized = posixpath.normpath(absolute)
    if normalized == "/" or normalized == "/root":
        return normalized, True, True
    parts = [part for part in normalized.split("/") if part]
    is_home = len(parts) <= 2 and bool(parts) and parts[0] in _HOME_PARENTS
    return normalized, is_home, _is_directory(raw, normalized)


def _absolute_form(raw: str) -> str | None:
    """Absolute form of ``raw`` — ``~``, ``$HOME``, ``~user`` expanded — or None."""
    if raw.startswith("/"):
        return raw
    home = os.path.expanduser("~")
    if raw == "~" or raw.startswith("~/"):
        return home + raw[1:]
    if raw.startswith("~"):
        user, _, rest = raw[1:].partition("/")
        if not user:
            return None
        return "/home/" + user + ("/" + rest if rest else "")
    match = _HOME_VARIABLE_RE.match(raw)
    if match:
        return home + raw[match.end():]
    return None


def _is_directory(raw: str, normalized: str) -> bool:
    if raw.endswith("/"):
        return True
    try:
        if os.path.isdir(normalized):
            return True
        if os.path.exists(normalized):
            return False
    except OSError:
        return False
    # A path that does not exist here is usually a remote (ssh) path: a name with
    # no extension is far more often a directory than a file, and counting it as a
    # root refuses a wide walk instead of letting one through.
    name = posixpath.basename(normalized)
    return bool(name) and "." not in name and not any(char in name for char in "*?[")


def _nested_commands(command: str):
    """Yield command strings carried inside another command.

    Two shapes: a quoted payload (``ssh host 'grep -r …'``, ``bash -c '…'``) which
    is one shell word, and an unquoted ssh command (``ssh host grep -r …``) whose
    words the remote shell re-joins.
    """
    for segment in _iter_top_level_shell_segments(command):
        tokens = _shell_segment_tokens(segment, 0)
        if not tokens:
            continue
        for token in tokens:
            if len(token) > 1 and any(char.isspace() for char in token) and _NESTED_SCAN_RE.search(token):
                yield token
        name = _executable_name(tokens[0])
        if name in _SSH_NAMES:
            payload = _ssh_payload(tokens[1:])
            if payload:
                yield payload


def _ssh_payload(args: list[str]) -> str | None:
    index = 0
    while index < len(args) and args[index].startswith("-"):
        index += 2 if args[index] in _SSH_OPTIONS_WITH_ARG else 1
    if index >= len(args):
        return None
    return " ".join(args[index + 1:]) or None
