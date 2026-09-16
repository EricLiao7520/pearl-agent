from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, NamedTuple

@dataclass
class TestCase:
    name: str
    args: List[Any]
    kwargs: Dict[str, Any]
    expected: Any

class ExecResult(NamedTuple):
    ok: bool
    stdout: str
    stderr: str
    exception: Optional[str]
    time_s: float

@dataclass
class VerificationResult:
    passed_all: bool
    num_passed: int
    num_total: int
    stdout: str
    stderr: str
    exception: Optional[str]
    time_s: float

@dataclass
class JudgeConfig:
    temperature: float = 0.7
    max_choices: int = 10