# tests/test_config.py
import os
from pathlib import Path

import pytest

from xbrain.config import load_config
from xbrain.llm_client import (
    DEFAULT_NANOGPT_BASE_URL,
    DEFAULT_NANOGPT_MODEL,
    DEFAULT_NANOGPT_VISION_MODEL,
)


def _write_repo(root: Path, handle: str = "vgonpa") -> None:
    (root / "config.toml").write_text(
        "[paths]\n"
        'vault = "/tmp/vault"\n'
        'output_subdir = "learnings/x-knowledge"\n'
        'data_dir = "data"\n'
        "[x]\n"
        f'handle = "{handle}"\n',
        encoding="utf-8",
    )


def test_load_config_resolves_paths(tmp_path: Path):
    _write_repo(tmp_path)
    cfg = load_config(tmp_path)
    assert cfg.x_handle == "vgonpa"
    assert cfg.output_dir == Path("/tmp/vault/learnings/x-knowledge")
    assert cfg.items_path == tmp_path / "data" / "items.json"


def test_load_config_defaults_transcribe_command_to_parakeet(tmp_path: Path):
    """No [transcribe] section → the external transcriber defaults to
    `parakeet-mlx`, model unset (the transcriber's own default)."""
    _write_repo(tmp_path)
    cfg = load_config(tmp_path)
    assert cfg.transcribe_command == "parakeet-mlx"
    assert cfg.transcribe_model is None


def test_load_config_round_trips_transcribe_command_and_model(tmp_path: Path):
    (tmp_path / "config.toml").write_text(
        "[paths]\n"
        'vault = "/tmp/vault"\n'
        'output_subdir = "learnings/x-knowledge"\n'
        'data_dir = "data"\n'
        "[x]\n"
        'handle = "vgonpa"\n'
        "[transcribe]\n"
        'command = "my-asr --quiet"\n'
        'model = "parakeet-tdt-0.6b-v2"\n',
        encoding="utf-8",
    )
    cfg = load_config(tmp_path)
    assert cfg.transcribe_command == "my-asr --quiet"
    assert cfg.transcribe_model == "parakeet-tdt-0.6b-v2"


def test_load_config_defaults_vision_command_to_unset(tmp_path: Path):
    """No [vision] section → the external vision command is unset (`""`) and the
    model is None. `digest-video --frames` errors clearly until it is configured —
    there is NO bundled default vision model (#44 PR4)."""
    _write_repo(tmp_path)
    cfg = load_config(tmp_path)
    assert cfg.vision_command == ""
    assert cfg.vision_model is None


def test_load_config_round_trips_vision_command_and_model(tmp_path: Path):
    (tmp_path / "config.toml").write_text(
        "[paths]\n"
        'vault = "/tmp/vault"\n'
        'output_subdir = "learnings/x-knowledge"\n'
        'data_dir = "data"\n'
        "[x]\n"
        'handle = "vgonpa"\n'
        "[vision]\n"
        'command = "vlm-describe --fast"\n'
        'model = "qwen2-vl-7b"\n',
        encoding="utf-8",
    )
    cfg = load_config(tmp_path)
    assert cfg.vision_command == "vlm-describe --fast"
    assert cfg.vision_model == "qwen2-vl-7b"


def test_load_config_defaults_output_language_to_english(tmp_path: Path):
    """No [output] section → English default."""
    _write_repo(tmp_path)
    cfg = load_config(tmp_path)
    assert cfg.output_language == "English"


def test_load_config_round_trips_spanish_language(tmp_path: Path):
    (tmp_path / "config.toml").write_text(
        "[paths]\n"
        'vault = "/tmp/vault"\n'
        'output_subdir = "learnings/x-knowledge"\n'
        'data_dir = "data"\n'
        "[x]\n"
        'handle = "vgonpa"\n'
        "[output]\n"
        'language = "Spanish"\n',
        encoding="utf-8",
    )
    cfg = load_config(tmp_path)
    assert cfg.output_language == "Spanish"


def test_load_config_rejects_unknown_language(tmp_path: Path):
    (tmp_path / "config.toml").write_text(
        "[paths]\n"
        'vault = "/tmp/vault"\n'
        'output_subdir = "learnings/x-knowledge"\n'
        'data_dir = "data"\n'
        "[x]\n"
        'handle = "vgonpa"\n'
        "[output]\n"
        'language = "Klingon"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Klingon"):
        load_config(tmp_path)


def test_load_config_rejects_empty_handle(tmp_path: Path):
    _write_repo(tmp_path, handle="")
    with pytest.raises(ValueError, match="handle"):
        load_config(tmp_path)


def test_load_config_reads_pipeline_settings(tmp_path: Path):
    (tmp_path / "config.toml").write_text(
        "[paths]\n"
        'vault = "/tmp/vault"\n'
        'output_subdir = "learnings/x-knowledge"\n'
        'data_dir = "data"\n'
        "[x]\n"
        'handle = "vgonpa"\n'
        "[llm]\n"
        'model = "zai-org/glm-5.2"\n'
        'vision_model = "minimax/minimax-m3"\n'
        "[enrich]\n"
        'executor = "api"\n'
        "[vocab]\n"
        "target_count = 25\n",
        encoding="utf-8",
    )
    cfg = load_config(tmp_path)
    assert cfg.llm_provider == "nanogpt"
    assert cfg.llm_model == "zai-org/glm-5.2"
    assert cfg.llm_vision_model == "minimax/minimax-m3"
    assert cfg.enrich_executor == "api"
    assert cfg.enrich_model == "zai-org/glm-5.2"
    assert cfg.describe_model == "minimax/minimax-m3"
    assert cfg.vocab_target_count == 25


def test_load_config_pipeline_settings_have_defaults(tmp_path: Path):
    _write_repo(tmp_path)  # config.toml WITHOUT [enrich]/[vocab]
    cfg = load_config(tmp_path)
    assert cfg.llm_provider == "nanogpt"
    assert cfg.llm_base_url == DEFAULT_NANOGPT_BASE_URL
    assert cfg.llm_model == DEFAULT_NANOGPT_MODEL
    assert cfg.llm_vision_model == DEFAULT_NANOGPT_VISION_MODEL
    assert cfg.enrich_executor == "claude-code"  # subscription is the default
    assert cfg.enrich_model == DEFAULT_NANOGPT_MODEL
    assert cfg.describe_model == DEFAULT_NANOGPT_VISION_MODEL
    assert cfg.vocab_target_count == 30


def test_load_config_reads_llm_section(tmp_path: Path):
    (tmp_path / "config.toml").write_text(
        "[paths]\n"
        'vault = "/tmp/vault"\n'
        'output_subdir = "learnings/x-knowledge"\n'
        'data_dir = "data"\n'
        "[x]\n"
        'handle = "vgonpa"\n'
        "[llm]\n"
        'provider = "anthropic"\n',
        encoding="utf-8",
    )
    cfg = load_config(tmp_path)
    assert cfg.llm_provider == "anthropic"
    assert cfg.llm_base_url == ""
    assert cfg.llm_model == "claude-haiku-4-5-20251001"
    assert cfg.llm_vision_model == "claude-sonnet-4-6"
    assert cfg.enrich_model == "claude-haiku-4-5-20251001"
    assert cfg.describe_model == "claude-sonnet-4-6"


def test_load_config_rejects_anthropic_model_when_provider_is_nanogpt(tmp_path: Path):
    (tmp_path / "config.toml").write_text(
        "[paths]\n"
        'vault = "/tmp/vault"\n'
        'output_subdir = "learnings/x-knowledge"\n'
        'data_dir = "data"\n'
        "[x]\n"
        'handle = "vgonpa"\n'
        "[llm]\n"
        'provider = "nanogpt"\n'
        'model = "claude-haiku-4-5-20251001"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Anthropic-native"):
        load_config(tmp_path)


def test_load_config_rejects_nanogpt_model_when_provider_is_anthropic(tmp_path: Path):
    (tmp_path / "config.toml").write_text(
        "[paths]\n"
        'vault = "/tmp/vault"\n'
        'output_subdir = "learnings/x-knowledge"\n'
        'data_dir = "data"\n'
        "[x]\n"
        'handle = "vgonpa"\n'
        "[llm]\n"
        'provider = "anthropic"\n'
        'model = "zai-org/glm-5.2"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not look like an Anthropic"):
        load_config(tmp_path)


def test_load_config_rejects_provider_mismatched_environment_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _write_repo(tmp_path)
    monkeypatch.setenv("NANOGPT_MODEL", "claude-haiku-4-5-20251001")

    with pytest.raises(ValueError, match="NANOGPT_MODEL"):
        load_config(tmp_path)


def test_load_config_ignores_nanogpt_base_url_when_provider_is_anthropic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    (tmp_path / "config.toml").write_text(
        "[paths]\n"
        'vault = "/tmp/vault"\n'
        'output_subdir = "learnings/x-knowledge"\n'
        'data_dir = "data"\n'
        "[x]\n"
        'handle = "vgonpa"\n'
        "[llm]\n"
        'provider = "anthropic"\n'
        'base_url = "https://nano-gpt.example/api/v1"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("NANOGPT_BASE_URL", "https://nano-gpt.env/api/v1")

    cfg = load_config(tmp_path)

    assert cfg.llm_base_url == ""


def test_load_config_reads_dotenv_without_overriding_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _write_repo(tmp_path)
    (tmp_path / ".env").write_text(
        'NANOGPT_MODEL="zai-org/glm-5.2:thinking"\n'
        'NANOGPT_VISION_MODEL="minimax/minimax-m3-pro"\n'
        'NANOGPT_BASE_URL="https://nano-gpt.example/api/v1"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("NANOGPT_MODEL", "zai-org/glm-5.2")

    try:
        cfg = load_config(tmp_path)
    finally:
        os.environ.pop("NANOGPT_BASE_URL", None)
        os.environ.pop("NANOGPT_VISION_MODEL", None)

    assert cfg.enrich_model == "zai-org/glm-5.2"
    assert cfg.describe_model == "minimax/minimax-m3-pro"
    assert cfg.llm_base_url == "https://nano-gpt.example/api/v1"


def test_load_config_allows_distinct_text_and_vision_models(tmp_path: Path):
    (tmp_path / "config.toml").write_text(
        "[paths]\n"
        'vault = "/tmp/vault"\n'
        'output_subdir = "learnings/x-knowledge"\n'
        'data_dir = "data"\n'
        "[x]\n"
        'handle = "vgonpa"\n'
        "[enrich]\n"
        'model = "zai-org/glm-5.2"\n'
        "[describe]\n"
        'model = "minimax/minimax-m3"\n',
        encoding="utf-8",
    )
    cfg = load_config(tmp_path)
    assert cfg.llm_model == "zai-org/glm-5.2"
    assert cfg.llm_vision_model == "minimax/minimax-m3"


def test_load_config_rejects_multiple_text_models(tmp_path: Path):
    (tmp_path / "config.toml").write_text(
        "[paths]\n"
        'vault = "/tmp/vault"\n'
        'output_subdir = "learnings/x-knowledge"\n'
        'data_dir = "data"\n'
        "[x]\n"
        'handle = "vgonpa"\n'
        "[llm]\n"
        'model = "zai-org/glm-5.2"\n'
        "[enrich]\n"
        'model = "zai-org/glm-5.2:thinking"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="text API LLM model"):
        load_config(tmp_path)


def test_load_config_rejects_multiple_vision_models(tmp_path: Path):
    (tmp_path / "config.toml").write_text(
        "[paths]\n"
        'vault = "/tmp/vault"\n'
        'output_subdir = "learnings/x-knowledge"\n'
        'data_dir = "data"\n'
        "[x]\n"
        'handle = "vgonpa"\n'
        "[llm]\n"
        'vision_model = "minimax/minimax-m3"\n'
        "[describe]\n"
        'model = "minimax/minimax-m3-pro"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="vision API LLM model"):
        load_config(tmp_path)


def test_load_config_rejects_unknown_executor(tmp_path: Path):
    (tmp_path / "config.toml").write_text(
        "[paths]\n"
        'vault = "/tmp/vault"\n'
        'output_subdir = "learnings/x-knowledge"\n'
        'data_dir = "data"\n'
        "[x]\n"
        'handle = "vgonpa"\n'
        "[enrich]\n"
        'executor = "gpt"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="executor must be"):
        load_config(tmp_path)


def test_load_config_rejects_zero_target_count(tmp_path: Path):
    (tmp_path / "config.toml").write_text(
        "[paths]\n"
        'vault = "/tmp/vault"\n'
        'output_subdir = "learnings/x-knowledge"\n'
        'data_dir = "data"\n'
        "[x]\n"
        'handle = "vgonpa"\n'
        "[vocab]\n"
        "target_count = 0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="target_count must be >= 1"):
        load_config(tmp_path)


def test_config_topics_threshold_defaults_to_25(tmp_path):
    from xbrain.config import load_config

    (tmp_path / "config.toml").write_text(
        '[paths]\nvault = "/v"\noutput_subdir = "o"\ndata_dir = "data"\n[x]\nhandle = "h"\n',
        encoding="utf-8",
    )
    cfg = load_config(tmp_path)
    assert cfg.topics_resynth_threshold == 25
    assert cfg.topics_path == tmp_path / "data" / "topics.json"


def test_config_topics_threshold_is_configurable(tmp_path):
    from xbrain.config import load_config

    (tmp_path / "config.toml").write_text(
        '[paths]\nvault = "/v"\noutput_subdir = "o"\ndata_dir = "data"\n'
        '[x]\nhandle = "h"\n'
        "[topics]\nresynth_threshold = 50\n",
        encoding="utf-8",
    )
    assert load_config(tmp_path).topics_resynth_threshold == 50


def test_load_config_defaults_topic_style_to_wikilink(tmp_path: Path):
    """No `[output] topic_style` key → wikilink default (backwards-compat)."""
    _write_repo(tmp_path)
    cfg = load_config(tmp_path)
    assert cfg.topic_style == "wikilink"


def test_load_config_round_trips_hashtag_topic_style(tmp_path: Path):
    """Explicit `topic_style = "hashtag"` round-trips."""
    (tmp_path / "config.toml").write_text(
        "[paths]\n"
        'vault = "/tmp/vault"\n'
        'output_subdir = "learnings/x-knowledge"\n'
        'data_dir = "data"\n'
        "[x]\n"
        'handle = "vgonpa"\n'
        "[output]\n"
        'topic_style = "hashtag"\n',
        encoding="utf-8",
    )
    cfg = load_config(tmp_path)
    assert cfg.topic_style == "hashtag"


def test_load_config_rejects_unknown_topic_style(tmp_path: Path):
    """Unknown topic_style fails fast with the supported list in the message."""
    (tmp_path / "config.toml").write_text(
        "[paths]\n"
        'vault = "/tmp/vault"\n'
        'output_subdir = "learnings/x-knowledge"\n'
        'data_dir = "data"\n'
        "[x]\n"
        'handle = "vgonpa"\n'
        "[output]\n"
        'topic_style = "bogus"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="topic_style"):
        load_config(tmp_path)


def test_load_config_describe_settings_have_defaults(tmp_path: Path):
    """No [describe] section → global vision LLM model + version v1."""
    _write_repo(tmp_path)
    cfg = load_config(tmp_path)
    assert cfg.describe_model == DEFAULT_NANOGPT_VISION_MODEL
    assert cfg.describe_version == "v1"


def test_load_config_round_trips_describe_overrides(tmp_path: Path):
    """Legacy [describe].model feeds the vision model."""
    (tmp_path / "config.toml").write_text(
        "[paths]\n"
        'vault = "/tmp/vault"\n'
        'output_subdir = "learnings/x-knowledge"\n'
        'data_dir = "data"\n'
        "[x]\n"
        'handle = "vgonpa"\n'
        "[describe]\n"
        'model = "minimax/minimax-m3-pro"\n'
        'version = "v3"\n',
        encoding="utf-8",
    )
    cfg = load_config(tmp_path)
    assert cfg.llm_model == DEFAULT_NANOGPT_MODEL
    assert cfg.enrich_model == DEFAULT_NANOGPT_MODEL
    assert cfg.llm_vision_model == "minimax/minimax-m3-pro"
    assert cfg.describe_model == "minimax/minimax-m3-pro"
    assert cfg.describe_version == "v3"
