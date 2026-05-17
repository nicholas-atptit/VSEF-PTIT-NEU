from __future__ import annotations

from pathlib import Path

from scripts.check_provider_usage_policy import scan_file


def test_direct_provider_imports_in_unauthorized_script_are_detected(tmp_path: Path) -> None:
    path = tmp_path / "bad_fetch.py"
    path.write_text(
        "from vnstock_data import Quote\n"
        "df = Quote(source='VCI', symbol='FPT').history(start='2026-01-01', end='2026-01-02', interval='1H')\n",
        encoding="utf-8",
    )
    violations = scan_file(path)
    patterns = {violation.pattern for violation in violations}
    assert "from vnstock_data import" in patterns
    assert "Quote(" in patterns
    assert ".history(" in patterns


def test_gateway_file_is_allowed() -> None:
    violations = scan_file(Path("src/data/providers/vn_price_gateway.py"))
    assert violations == []
