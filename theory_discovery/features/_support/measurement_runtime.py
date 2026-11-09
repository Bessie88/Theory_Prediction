"""Explicit model-provider factory using existing transport; never called on import."""
from .semantic_features import LegacyJSONFeatureBackend, ModelPermit
from theory_discovery.core.schemas import require


def configured_backend(model_settings, runtime, prompt_version):
    if (model_settings is None or not model_settings.enabled or
            not runtime.execution_authorized or not runtime.models_authorized):
        return None
    require(model_settings.prompt_version == prompt_version, "Central prompt version must match installed prompt")
    from pathlib import Path
    from theory_discovery.core.config import LLMConfig
    from theory_discovery.core.response_recovery import RecoverableQwenJSONClient
    from theory_discovery.core.storage import Cache
    config = LLMConfig(enabled=True, base_url=model_settings.endpoint, model=model_settings.model,
                       temperature=model_settings.temperature, max_tokens=model_settings.max_tokens,
                       retries=model_settings.retries, max_prompt_chars=runtime.max_text_characters,
                       timeout_seconds=900 if model_settings.max_tokens > 4096 else 300)
    client = RecoverableQwenJSONClient(config, Cache(Path(runtime.cache_directory) / "model_transport"))
    client.verify_model()  # Explicit future authorized factory only; no service startup/download.
    return LegacyJSONFeatureBackend(client, model_settings.model, prompt_version,
                                    ModelPermit(True, True))
