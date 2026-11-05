"""Tests for the whole-machine recursive-scan floor (ATE-18764).

``approvals.deny`` is fnmatch over one flat command string, so it cannot tell a
second tree root from a pipe. ATE-18741 measured a ``grep -rl`` over five roots
that passed every deny rule; this floor refuses the shape on the parsed command
instead, unconditionally (before yolo / mode=off / cron approve-mode).
"""

import pytest

from tools import approval as mod
from tools import approval_context
from tools import scan_guard

#: The exact command ATE-18741 measured on r410: 5.2 h, load 37.
INCIDENT = (
    'grep -rl "mode=all|mode: \'all\'|mode:\\"all\\"" '
    "/home/filipe/ /home/hakim/ /home/hr/ /home/ilyas/ /opt"
)

#: The documented bounded form: one root, one pipeline. Must stay allowed.
BOUNDED = "grep -rn x /home/hr/agent-roots/scripts | head"
BOUNDED_FIND = "find /home/hr/agent-roots -name '*.md'"


@pytest.fixture
def quiet_config(monkeypatch):
    """No approvals.deny rules: the floor alone must decide."""
    monkeypatch.setattr(
        approval_context, "_get_approval_config", lambda: {"mode": "manual", "deny": []}
    )


@pytest.fixture
def clean_env(monkeypatch):
    """Non-interactive, non-gateway, non-cron, non-yolo baseline."""
    for var in (
        "HERMES_YOLO_MODE",
        "HERMES_GATEWAY_SESSION",
        "HERMES_CRON_SESSION",
        "HERMES_INTERACTIVE",
        "HERMES_EXEC_ASK",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(mod, "_YOLO_MODE_FROZEN", False)


class TestBlocked:
    @pytest.mark.parametrize(
        "command",
        [
            INCIDENT,
            # a single root that is the whole machine or a home directory itself
            "find / -xdev -name '*.pem'",
            "find /",
            "grep -r x /home/filipe",
            "grep -r x /home/filipe/",
            "grep -r x ~",
            "grep -r x ~/",
            "grep -r x $HOME",
            "grep -r x ${HOME}",
            "grep -r x /root",
            "grep -R x /",
            "grep --recursive x /home/hr",
            "grep -d recurse x /",
            # two roots at once
            "grep -rn x /home/hr /opt",
            "grep -rl x /home/hr/agent-roots /home/hakim/agent-roots",
            "grep -rl x /home/hakim/ /home/hr/",
            "find /tmp /var",
            # the executable's identity does not hide it
            "sudo grep -rl x /home/hr",
            "sudo -u root grep -r x /home/hr",
            "/usr/bin/grep -r x /home/hr",
            '"/usr/bin/grep" -r x /home/hr',
            'g""rep -r x /home/hr',
            'gr"ep" -r x /home/hr/ /opt/',
            "env -i grep -r x /home/hr/ /opt/",
            "cd /tmp && grep -r x /home/hr/ /opt/",
            "true; find / -name x",
            # a shell carrier, including the ssh shape whose body the tool call carries
            "bash -c 'grep -r x /home/a /home/b'",
            "sh -lc 'grep -r x /home/hr /opt'",
            "ssh r410 'grep -r x /home/filipe'",
            'ssh r410 "grep -rl mode /home/filipe/ /home/hakim/"',
            "ssh -p 22 r410 grep -r x /home/a /home/b",
            # recursive-by-default searchers are the same walk under another name
            "rg foo /home/a /home/b",
            # an unquoted heredoc body stays visible: it is expanded data that a
            # shell (`bash <<EOF`) can execute
            "cat <<EOF\ngrep -rl pattern /home/hr\nEOF",
        ],
    )
    def test_whole_machine_walk_is_named(self, command):
        assert scan_guard.recursive_scan_reason(command)


class TestAllowed:
    @pytest.mark.parametrize(
        "command",
        [
            BOUNDED,
            BOUNDED_FIND,
            "grep -rn TODO /home/hr/agent-roots/scripts --include=*.md | head",
            "grep -rn x /home/hr/agent-roots/scripts -A 3",
            "grep -r -e 'a b' /home/hr/agent-roots/scripts",
            "find /home/hr/agent-roots -maxdepth 2 -name '*.md'",
            # one root, several files: a file is not a tree root
            "grep -n foo /etc/passwd /etc/hosts",
            "grep -l x /home/hakim/notes.md",
            # not recursive
            "grep -c x /var/log/syslog",
            # relative arguments are not absolute tree roots
            "grep -rn foo tools/",
            "grep -r foo relative/dir",
            "find . -name '*.py'",
            # one bounded absolute root on another volume
            "grep -rn PATTERN /srv/app/data",
            "ls -la /home/hakim",
            "git log --oneline -5",
            # a scan quoted as data, not executed: a comment or document stays writable
            "cat <<'EOF'\ngrep -rl pattern /home/filipe/ /home/hakim/ /opt\nEOF",
        ],
    )
    def test_bounded_forms_still_pass(self, command):
        assert scan_guard.recursive_scan_reason(command) is None


class TestFloorIntegration:
    def test_terminal_guard_refuses_the_incident(self, quiet_config, clean_env):
        result = mod.check_all_command_guards(INCIDENT, "local")
        assert result["approved"] is False
        assert result["recursive_scan"] is True
        assert "BLOCKED" in result["message"]
        assert "ATE-18741" in result["message"]

    def test_the_dangerous_command_entry_point_also_refuses(self, quiet_config, clean_env):
        result = mod.check_dangerous_command("find / -xdev -name '*.pem'", "local")
        assert result["approved"] is False
        assert result["recursive_scan"] is True

    def test_yolo_does_not_bypass_the_floor(self, quiet_config, clean_env, monkeypatch):
        monkeypatch.setattr(mod, "_YOLO_MODE_FROZEN", True)
        result = mod.check_all_command_guards(INCIDENT, "local")
        assert result["approved"] is False
        assert result["recursive_scan"] is True

    def test_mode_off_does_not_bypass_the_floor(self, clean_env, monkeypatch):
        monkeypatch.setattr(
            approval_context, "_get_approval_config", lambda: {"mode": "off", "deny": []}
        )
        result = mod.check_all_command_guards(INCIDENT, "local")
        assert result["approved"] is False
        assert result["recursive_scan"] is True

    def test_bounded_form_is_not_refused_by_this_guard(self, quiet_config, clean_env):
        result = mod.check_all_command_guards(BOUNDED, "local")
        assert result["approved"] is True
        assert "recursive_scan" not in result
