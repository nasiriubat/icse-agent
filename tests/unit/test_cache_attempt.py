from memcheck.config import ExperimentConfig
from memcheck.models import content_hash
from memcheck.providers.base import ChatMessage, ChatRequest
from memcheck.providers.cache import request_key, stamp_attempt


def _request(**overrides: object) -> ChatRequest:
    fields: dict[str, object] = {
        "model_identifier": "gpt-5-mini",
        "system_prompt": "answer",
        "messages": [ChatMessage(role="user", content="hello")],
    }
    fields.update(overrides)
    return ChatRequest.model_validate(fields)


def test_attempt_zero_reproduces_the_pre_field_key() -> None:
    request = _request()
    legacy_payload = request.model_dump(mode="json")
    del legacy_payload["attempt"]
    legacy_key = content_hash({"provider": "openai", "request": legacy_payload})
    assert request_key("openai", request) == legacy_key


def test_attempt_changes_the_key() -> None:
    assert request_key("openai", _request()) != request_key("openai", _request(attempt=1))
    assert request_key("openai", _request(attempt=1)) != request_key("openai", _request(attempt=2))


def test_stamp_attempt_only_when_non_zero() -> None:
    request = _request()
    assert stamp_attempt(request, 0) is request
    assert stamp_attempt(request, 1).attempt == 1
    assert request.attempt == 0


def test_config_attempt_defaults_to_zero() -> None:
    config = ExperimentConfig.model_validate(
        {"name": "x", "workload": {"name": "synthetic_longitudinal", "version": "v2"}}
    )
    assert config.attempt == 0
