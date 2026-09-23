"""Generated jobs must carry the directives the clusters actually need."""

import pytest

from src.sbatch import (CLUSTERS, JobSpec, build_command, default_output_dir,
                         render, write)


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


def test_build_command_always_raises_dpvo_buffer_size():
    command = build_command(video="a.mp4", config=None, calibration="c.txt",
                            camera_height=1.8, output=None, depth_model=None)
    assert "--dpvo-opts BUFFER_SIZE 16384" in command


def test_default_output_dir_keeps_results_beside_the_video(tmp_path):
    videos_root = tmp_path / "videos"
    video = videos_root / "GH010050" / "GH010050.MP4"
    output = default_output_dir(video, videos_root, "2026-09-23_13-53-43")
    assert output == videos_root / "GH010050" / "results" / "2026-09-23_13-53-43"


def test_default_output_dir_is_none_outside_videos_root(tmp_path):
    videos_root = tmp_path / "videos"
    video = tmp_path / "elsewhere" / "clip.mp4"
    assert default_output_dir(video, videos_root, "2026-09-23_13-53-43") is None


def test_default_output_dir_is_none_without_a_video(tmp_path):
    assert default_output_dir(None, tmp_path / "videos", "2026-09-23_13-53-43") is None


def test_write_uses_the_given_stamp_for_the_log_file(tmp_path):
    destination = tmp_path / "walk.sbatch"
    written = write(_spec(), destination, stamp="2026-09-23_13-53-43")
    assert "walk_2026-09-23_13-53-43" in written.read_text()
