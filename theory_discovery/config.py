"""Validated configuration; paths resolve relative to the configuration file."""
from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path


@dataclass
class LLMConfig:
    enabled: bool = False
    base_url: str = ''
    model: str = ''
    api_key_env: str = 'THEORY_API_KEY'
    temperature: float = 0
    max_tokens: int = 4096
    max_prompt_chars: int = 48000
    timeout_seconds: float = 120
    retries: int = 2
    batch_size: int = 4
    max_calls: int = 100
    enable_thinking: bool = False
    semantic_boundaries: bool = False
    context_length: int = 32768
    context_safety_margin: int = 512
    tokenizer_path: str = ''
    condition_batch_size: int = 10


@dataclass
class Config:
    schema_version: str = '1'
    allow_real_data: bool = False
    input_path: str = ''
    input_format: str = 'jsonl'
    split: str = 'discovery'
    output_dir: str = 'outputs'
    cache_dir: str = 'outputs/cache'
    max_observations: int = 1000
    workers: int = 1
    min_support: int = 2
    min_action_frequency: float = 0.8
    max_condition_size: int = 3
    max_candidates_per_group: int = 100
    max_boundary_candidates: int = 40
    min_boundary_gain: float = 0.05
    min_retained_support: float = 0.5
    min_test_cases: int = 20
    pre_context_keys: list[str] = field(default_factory=list)
    rule_induction_mode: str = 'qwen_open_codes'
    boundary_proposal_mode: str = 'qwen_open_codes'
    llm: LLMConfig = field(default_factory=LLMConfig)

    def validate(self):
        if self.rule_induction_mode not in ('qwen_open_codes', 'statistical') or self.boundary_proposal_mode not in ('qwen_open_codes', 'statistical'):
            raise ValueError('Invalid rule/boundary execution mode')
        if self.schema_version != '1' or self.split != 'discovery':
            raise ValueError('Discovery requires schema version 1 and split=discovery')
        for name in ('workers', 'max_observations', 'min_support', 'max_condition_size',
                     'max_candidates_per_group', 'max_boundary_candidates', 'min_test_cases'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.max_condition_size > 8:
            raise ValueError('max_condition_size must be <= 8 (bounded search)')
        for name in ('min_action_frequency', 'min_boundary_gain', 'min_retained_support'):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f'{name} must be in [0,1]')
        if self.input_format not in ('jsonl', 'wnc_tsv'):
            raise ValueError('Unsupported input format')
        if any(type(k) is not str or not k for k in self.pre_context_keys):
            raise ValueError('Invalid pre-context allowlist')
        if len(set(self.pre_context_keys)) != len(self.pre_context_keys):
            raise ValueError('Duplicate pre-context keys')
        if self.llm.batch_size < 1 or self.llm.retries < 0 or self.llm.max_calls < 0:
            raise ValueError('Invalid inference budget')
        if self.llm.max_tokens < 1 or self.llm.max_prompt_chars < 1 or self.llm.timeout_seconds <= 0:
            raise ValueError('Invalid inference limits')
        if self.llm.enabled and (not self.llm.base_url or not self.llm.model):
            raise ValueError('Explicit endpoint and model identifier required')
        return self

    def to_dict(self):
        return asdict(self)


def load_config(path):
    path = Path(path).resolve()
    # JSON is a strict YAML subset; use this format without installing PyYAML.
    raw = json.loads(path.read_text(encoding='utf-8'))
    raw['llm'] = LLMConfig(**raw.get('llm', {}))
    cfg = Config(**raw).validate()
    for key in ('input_path', 'output_dir', 'cache_dir'):
        value = getattr(cfg, key)
        if value:
            setattr(cfg, key, str((path.parent / value).resolve()))
    return cfg


def configured_tokenizer_path():
    """Use an explicitly configured local tokenizer for prompt budgeting."""
    path = os.environ.get("SIMULATION_TOKENIZER_PATH")
    if not path:
        raise ValueError("Set SIMULATION_TOKENIZER_PATH to the served model's local tokenizer directory")
    return path
