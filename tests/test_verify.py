from pathlib import Path

import yaml

from tpa.verify import verify_run


def _create_run(tmp_path: Path, markdown: str, retrieved: list[dict], evidence: bool = True) -> Path:
    run_dir = tmp_path
    (run_dir / "section_transport.md").write_text(markdown, encoding="utf-8")
    (run_dir / "retrieved.json").write_text(__import__("json").dumps(retrieved, indent=2), encoding="utf-8")
    if evidence:
        evidence_dir = run_dir / "evidence"
        evidence_dir.mkdir(parents=True, exist_ok=True)
        (evidence_dir / "dummy.png").write_bytes(b"00")
    manifest = {
        "files": {
            "output_md": "section_transport.md",
            "retrieved": "retrieved.json",
            "prompt": "prompt.txt",
            "completion": "completion.md",
            "evidence_dir": "evidence" if evidence else None,
        },
        "output": {"save_zip_of_pages": evidence},
    }
    (run_dir / "RUN.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    (run_dir / "prompt.txt").write_text("PROMPT", encoding="utf-8")
    (run_dir / "completion.md").write_text(markdown, encoding="utf-8")
    return run_dir


def test_verify_fails_without_policy(tmp_path):
    run_dir = tmp_path / "fail_run"
    run_dir.mkdir()
    retrieved = [
        {"id": "APP:sample", "kind": "app", "score": 0.9, "path": str(Path("tests/data/app/sample_app.pdf")), "page": 1}
    ]
    markdown = "## Transport\n\nParagraph without policy."
    _create_run(run_dir, markdown, retrieved, evidence=False)
    assert not verify_run(run_dir)


def test_verify_pass(tmp_path):
    run_dir = tmp_path / "pass_run"
    run_dir.mkdir()
    retrieved = [
        {"id": "APP:sample", "kind": "app", "score": 0.9, "path": str(Path("tests/data/app/sample_app.pdf")), "page": 1},
        {"id": "APP:sample2", "kind": "app", "score": 0.85, "path": str(Path("tests/data/app/sample_app_2.pdf")), "page": 1},
        {"id": "POL:sample_policy", "kind": "policy", "score": 0.8, "path": str(Path("tests/data/policy/sample_policy.pdf")), "page": 1},
        {"id": "POL:sample_policy2", "kind": "policy", "score": 0.7, "path": str(Path("tests/data/policy/sample_policy_2.pdf")), "page": 1},
    ]
    markdown = (
        "## Transport\n\nParagraph with policy [POL:sample_policy].\n\n"
        "Another paragraph [APP:sample] [POL:sample_policy2].\n\n"
        "### Mini Planning Balance\nBalance text [POL:sample_policy] [APP:sample2]."
    )
    run_dir = _create_run(run_dir, markdown, retrieved, evidence=True)
    assert verify_run(run_dir)
