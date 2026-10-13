import re
from pathlib import Path

import pytest

from recomlab.config import ConfigError, SecretStr, load_settings

SRC = Path(__file__).resolve().parents[2] / "src" / "recomlab"


def test_defaults():
    s = load_settings(environ={})
    assert s.env == "development"
    assert s.evaluation.mode == "full"
    assert s.evaluation.k_values == (5, 10)
    assert not s.provider.enabled
    assert s.paths.processed_dir == Path("data") / "processed"


def test_file_then_env_then_overrides(tmp_path):
    cfg = tmp_path / "cfg.toml"
    cfg.write_text('env = "test"\n[evaluation]\nk_values = [3]\nseed = 7\n', encoding="utf-8")
    s = load_settings(config_file=cfg, environ={})
    assert (s.env, s.evaluation.k_values, s.evaluation.seed) == ("test", (3,), 7)
    s = load_settings(config_file=cfg, environ={"RECOMLAB_EVAL_SEED": "9"})
    assert s.evaluation.seed == 9 and s.evaluation.k_values == (3,)
    s = load_settings(
        config_file=cfg,
        environ={"RECOMLAB_EVAL_SEED": "9", "RECOMLAB_EVAL_K_VALUES": "1,2"},
        overrides={"evaluation": {"seed": 11}},
    )
    assert s.evaluation.seed == 11 and s.evaluation.k_values == (1, 2)


@pytest.mark.parametrize(
    "environ",
    [
        {"RECOMLAB_ENV": "staging"},
        {"RECOMLAB_EVAL_MODE": "approx"},
        {"RECOMLAB_EVAL_K_VALUES": "0,5"},
        {"RECOMLAB_EVAL_K_VALUES": "a"},
        {"RECOMLAB_EVAL_CANDIDATE_BUDGET": "-1"},
        {"RECOMLAB_DB_URL": ""},
        {"RECOMLAB_LLM_PROVIDER": "skynet"},
        {"RECOMLAB_LLM_PROVIDER": "openai", "RECOMLAB_LLM_MODEL": "m"},
        {"RECOMLAB_LLM_PROVIDER": "anthropic", "RECOMLAB_LLM_API_KEY": "k"},
    ],
)
def test_invalid_values_rejected(environ):
    with pytest.raises(ConfigError):
        load_settings(environ=environ)


def test_unknown_keys_rejected():
    with pytest.raises(ConfigError):
        load_settings(environ={}, overrides={"evaluation": {"nope": 1}})
    with pytest.raises(ConfigError):
        load_settings(environ={}, overrides={"bogus": 1})


def test_provider_secret_is_masked():
    s = load_settings(
        environ={
            "RECOMLAB_LLM_PROVIDER": "anthropic",
            "RECOMLAB_LLM_MODEL": "some-model",
            "RECOMLAB_LLM_API_KEY": "sk-secret-value",
        }
    )
    assert s.provider.enabled
    assert s.provider.llm_api_key.get_secret_value() == "sk-secret-value"
    assert "sk-secret" not in repr(s)
    assert repr(SecretStr("x")) == "SecretStr('***')"


def test_env_example_contains_only_placeholders():
    text = (SRC.parents[1] / ".env.example").read_text(encoding="utf-8")
    for line in text.splitlines():
        if "KEY" in line or "SECRET" in line or "TOKEN" in line:
            assert line.split("=", 1)[1].strip() == "", line


def test_only_config_module_reads_environment():
    pattern = re.compile(r"os\.environ|os\.getenv|getenv\(")
    offenders = [
        p.relative_to(SRC).as_posix()
        for p in SRC.rglob("*.py")
        if pattern.search(p.read_text(encoding="utf-8"))
    ]
    assert offenders == ["config/settings.py"]
