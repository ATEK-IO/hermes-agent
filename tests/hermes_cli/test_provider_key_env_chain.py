"""``providers.<name>.key_env`` accepts an ordered credential chain (ATE-15419).

A provider entry may name a LIST of env vars instead of one: the names are read left to right
and the first one that holds a value wins. That is what lets a single ``litellm`` entry read the
agent's own key during a seat run (``LITELLM_KEY``, injected per run and defined in no ``.env``)
and the shared key in the gateway/cron/CLI shape (``LITELLM_API_KEY``, in ``.env``) with no
per-seat rebinding and no ``.env`` edit.

The order lives on ``ProviderDef.api_key_env_vars``, which
``auth._resolve_api_key_provider_secret`` and ``agent.client_lifecycle`` already walk
first-usable-value-wins. These tests pin BOTH ends, so a reordering, a dropped entry, or a
``str(list)`` regression on either side fails here:

* the config surface (``provider_key_env_names`` / ``resolve_user_provider``): a string is still
  a one-tuple, a list keeps its order, a blank first name falls through, duplicates collapse;
* the consumer (``_resolve_api_key_provider_secret``): with both variables set the FIRST name
  wins, with only the second set the second wins — which is exactly the two process shapes the
  litellm card measured.
"""

from pathlib import Path

import pytest


def _make_pconfig(provider_id, env_vars=None):
    from hermes_cli.auth import ProviderConfig
    return ProviderConfig(
        id=provider_id,
        name=provider_id.title(),
        auth_type="api_key",
        api_key_env_vars=tuple(env_vars or [f"{provider_id.upper()}_API_KEY"]),
    )


@pytest.fixture
def isolated_hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    for key in ("LITELLM_KEY", "LITELLM_API_KEY", "GATEWAY_KEY"):
        monkeypatch.delenv(key, raising=False)
    return home


def _write_env_file(home: Path, **kwargs) -> None:
    lines = [f"{k}={v}" for k, v in kwargs.items()]
    (home / ".env").write_text("\n".join(lines) + "\n")


class TestKeyEnvNames:
    """``key_env`` normalisation: string in, one-tuple out; list in, order out."""

    def test_plain_string_is_a_one_tuple(self):
        from hermes_cli.providers import provider_key_env_names

        assert provider_key_env_names("MY_KEY") == ("MY_KEY",)
        assert provider_key_env_names("  MY_KEY  ") == ("MY_KEY",)

    def test_list_keeps_the_declared_order(self):
        from hermes_cli.providers import provider_key_env_names

        assert provider_key_env_names(["LITELLM_KEY", "LITELLM_API_KEY"]) == (
            "LITELLM_KEY",
            "LITELLM_API_KEY",
        )
        # Reversing the declaration must reverse the chain: order is the contract.
        assert provider_key_env_names(["LITELLM_API_KEY", "LITELLM_KEY"]) == (
            "LITELLM_API_KEY",
            "LITELLM_KEY",
        )

    def test_blank_first_name_falls_through_to_the_second(self):
        from hermes_cli.providers import provider_key_env_names, first_key_env

        assert provider_key_env_names(["", "SHARED_KEY"]) == ("SHARED_KEY",)
        assert provider_key_env_names(["   ", "SHARED_KEY"]) == ("SHARED_KEY",)
        assert first_key_env(["", "SHARED_KEY"]) == "SHARED_KEY"

    def test_empty_and_absent_forms_yield_no_names(self):
        from hermes_cli.providers import provider_key_env_names, first_key_env

        assert provider_key_env_names("") == ()
        assert provider_key_env_names([]) == ()
        assert provider_key_env_names(None) == ()
        assert first_key_env(None) == ""

    def test_duplicates_collapse_keeping_first_position(self):
        from hermes_cli.providers import provider_key_env_names

        assert provider_key_env_names(["A", "B", "A"]) == ("A", "B")

    def test_non_string_entries_are_ignored_not_stringified(self):
        from hermes_cli.providers import provider_key_env_names

        assert provider_key_env_names(["A", 42, None, "B"]) == ("A", "B")


class TestResolverCarriesTheChain:
    """``providers.<name>.key_env`` → ``ProviderDef.api_key_env_vars``, unchanged for strings."""

    def _pdef(self, key_env):
        from hermes_cli.providers import resolve_user_provider

        entry = {"base_url": "https://gateway.example.com/v1", "key_env": key_env}
        return resolve_user_provider("my-gateway", {"my-gateway": entry})

    def test_string_key_env_still_yields_one_entry(self):
        assert self._pdef("MY_GATEWAY_KEY").api_key_env_vars == ("MY_GATEWAY_KEY",)

    def test_list_key_env_yields_the_ordered_chain(self):
        assert self._pdef(["LITELLM_KEY", "LITELLM_API_KEY"]).api_key_env_vars == (
            "LITELLM_KEY",
            "LITELLM_API_KEY",
        )

    def test_no_key_env_yields_no_entries(self):
        assert self._pdef(None).api_key_env_vars == ()

    def test_api_key_env_alias_still_resolves(self):
        from hermes_cli.providers import resolve_user_provider

        pdef = resolve_user_provider(
            "my-gateway",
            {"my-gateway": {"base_url": "https://gateway.example.com/v1", "api_key_env": "X_KEY"}},
        )
        assert pdef.api_key_env_vars == ("X_KEY",)


class TestConsumerReadsTheChainInOrder:
    """The resolver half: first name with a value wins; an empty first name falls through."""

    def test_first_name_wins_when_both_are_set(self, isolated_hermes_home):
        """Seat run: ``.env`` holds the shared key, the run env injects the seat key."""
        _write_env_file(isolated_hermes_home, LITELLM_API_KEY="sk-shared-0123456789")
        import os

        os.environ["LITELLM_KEY"] = "sk-seat-0123456789"
        try:
            from hermes_cli.auth import _resolve_api_key_provider_secret

            key, source = _resolve_api_key_provider_secret(
                "litellm", _make_pconfig("litellm", ["LITELLM_KEY", "LITELLM_API_KEY"])
            )
        finally:
            os.environ.pop("LITELLM_KEY", None)
        assert key == "sk-seat-0123456789"
        assert source == "LITELLM_KEY"

    def test_second_name_wins_when_the_first_is_unset(self, isolated_hermes_home):
        """Gateway/cron shape: no seat key at all, so the chain falls to the shared key."""
        _write_env_file(isolated_hermes_home, LITELLM_API_KEY="sk-shared-0123456789")

        from hermes_cli.auth import _resolve_api_key_provider_secret

        key, source = _resolve_api_key_provider_secret(
            "litellm", _make_pconfig("litellm", ["LITELLM_KEY", "LITELLM_API_KEY"])
        )
        assert key == "sk-shared-0123456789"
        assert source == "LITELLM_API_KEY"

    def test_second_name_wins_when_the_first_is_empty(self, isolated_hermes_home):
        """An exported-but-empty first name is not a credential: keep walking the chain."""
        _write_env_file(isolated_hermes_home, LITELLM_API_KEY="sk-shared-0123456789")
        import os

        os.environ["LITELLM_KEY"] = ""
        try:
            from hermes_cli.auth import _resolve_api_key_provider_secret

            key, source = _resolve_api_key_provider_secret(
                "litellm", _make_pconfig("litellm", ["LITELLM_KEY", "LITELLM_API_KEY"])
            )
        finally:
            os.environ.pop("LITELLM_KEY", None)
        assert key == "sk-shared-0123456789"
        assert source == "LITELLM_API_KEY"

    def test_end_to_end_from_a_config_entry(self, isolated_hermes_home):
        """Config text → chain → resolved secret, with no hand-built ProviderConfig."""
        _write_env_file(isolated_hermes_home, LITELLM_API_KEY="sk-shared-0123456789")

        from hermes_cli.auth import _resolve_api_key_provider_secret
        from hermes_cli.providers import resolve_user_provider

        pdef = resolve_user_provider(
            "litellm",
            {
                "litellm": {
                    "base_url": "http://vm.atek.vpn:4200",
                    "key_env": ["LITELLM_KEY", "LITELLM_API_KEY"],
                }
            },
        )
        key, source = _resolve_api_key_provider_secret(
            "litellm", _make_pconfig("litellm", pdef.api_key_env_vars)
        )
        assert (key, source) == ("sk-shared-0123456789", "LITELLM_API_KEY")


class TestSingleNameReadersUseTheFirstName:
    """One-name surfaces must take the first name — never ``str(['A', 'B'])``."""

    def test_model_level_key_env_takes_the_first_name(self, isolated_hermes_home, monkeypatch):
        """``model.key_env`` is a single pointer: a chain copied there reads as its first name."""
        from hermes_cli import auth

        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda *a, **k: {
                "model": {"provider": "litellm", "key_env": ["LITELLM_KEY", "LITELLM_API_KEY"]}
            },
        )
        assert auth._model_level_key_env("litellm") == "LITELLM_KEY"
        # A provider other than the configured one is still not consulted.
        assert auth._model_level_key_env("openrouter") == ""
