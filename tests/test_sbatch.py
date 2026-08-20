"""Generated jobs must carry the directives the clusters actually need."""

import pytest

from src.sbatch import CLUSTERS, JobSpec, build_command, render, write


def _spec(**overrides):
    settings = dict(name="walk", email="you@example.org",
                    command="python video_to_trajectory.py clip.mp4")
    settings.update(overrides)
    return JobSpec(**settings)


def test_arrhenius_header(tmp_path):
    text = render(_spec(cluster="arrhenius"), tmp_path)
    assert "#SBATCH --partition=gpu" in text
    assert "#SBATCH -A naiss2025-22-1304-gpu" in text
    assert "#SBATCH --gpus 1" in text
    assert "#SBATCH --mail-user=you@example.org" in text


def test_disi_uses_gres(tmp_path):
    text = render(_spec(cluster="disi"), tmp_path)
    assert "#SBATCH --gres=gpu:1" in text
    assert "-A " not in text          # this cluster has no account


def test_work_dir_goes_to_node_local_scratch(tmp_path):
    text = render(_spec(), tmp_path)
    assert 'TMPDIR:-/tmp' in text
    assert 'trap' in text             # scratch is cleaned up on exit


def test_unknown_cluster_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="unknown cluster"):
        render(_spec(cluster="nowhere"), tmp_path)


def test_account_override(tmp_path):
    text = render(_spec(account="my-project"), tmp_path)
    assert "#SBATCH -A my-project" in text


def test_write_creates_log_dir(tmp_path):
    destination = tmp_path / "walk.sbatch"
    written = write(_spec(), destination)
    assert written.is_file()
    assert (tmp_path / "walk").is_dir()


def test_build_command_always_sets_work_dir():
    command = build_command(video="a.mp4", config=None, calibration="c.txt",
                            camera_height=1.8, output=None, depth_model=None)
    assert '--work-dir "$WORK_DIR"' in command
    assert "--camera-height 1.8" in command
