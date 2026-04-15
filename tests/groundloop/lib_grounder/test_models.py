from __future__ import annotations

import pytest
from pydantic import ValidationError

from groundloop.lib_grounder.models import GroundingReport, Symbol


def test_symbol_frozen() -> None:
    s = Symbol(module="os", attr=None, kind="import", resolved=True, line=1)
    with pytest.raises(ValidationError):
        s.module = "sys"  # type: ignore[misc]


def test_grounding_report_frozen() -> None:
    r = GroundingReport(total_symbols=0, grounded=(), ungrounded=(), groundedness=1.0)
    with pytest.raises(ValidationError):
        r.groundedness = 0.0  # type: ignore[misc]
