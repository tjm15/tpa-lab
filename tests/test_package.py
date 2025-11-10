import zipfile

import yaml

from tpa.package import package_run


def test_package_includes_extended_files(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    files = {
        "output_md": "section_transport.md",
        "retrieved": "retrieved.json",
        "prompt": "prompt.txt",
        "completion": "completion.md",
        "reasoning": "reasoning.json",
        "material_considerations": "material_considerations.json",
        "planning_balance_md": "planning_balance.md",
        "planning_balance_json": "planning_balance.json",
        "mc_conflicts": "mc_conflicts.json",
        "evidence_dir": None,
    }
    for rel in files.values():
        if rel:
            (run_dir / rel).write_text(rel, encoding="utf-8")
    manifest = {"files": files}
    (run_dir / "RUN.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    zip_path = package_run(run_dir)
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
    for rel in files.values():
        if rel:
            assert rel in names
