import os
from pathlib import Path

from typer.testing import CliRunner

from tpa.cli import app


def test_full_pipeline(tmp_path):
    runner = CliRunner()
    config_path = Path("CONFIG.sample.yml")
    os.environ["TPA_EMBED_MODE"] = "hash"

    index_result = runner.invoke(app, ["index", "--config", str(config_path)])
    assert index_result.exit_code == 0

    run_name = "test_run"
    report_result = runner.invoke(
        app,
        [
            "report",
            "--section",
            "transport",
            "--config",
            str(config_path),
            "--run",
            run_name,
        ],
    )
    assert report_result.exit_code == 0

    run_dir = Path("runs") / run_name
    assert (run_dir / "section_transport.md").exists()
    assert (run_dir / "retrieved.json").exists()

    verify_result = runner.invoke(app, ["verify", "--run", run_name])
    assert verify_result.exit_code == 0

    package_result = runner.invoke(app, ["package", "--run", run_name])
    assert package_result.exit_code == 0
    assert (Path("runs") / run_name / "package.zip").exists()
