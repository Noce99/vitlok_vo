"""RunConfig layers CLI over YAML over defaults, and validates before running."""

import pytest
import yaml

from src.config import RunConfig, build_parser

CALIBRATION = "calibration/gopro_SW.txt"


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"not really a video")
    return path


def _config(video, **overrides):
    settings = {"video": video, "calibration": CALIBRATION}
    settings.update(overrides)
    return RunConfig(**settings)


def test_defaults(video):
    cfg = _config(video)
    assert cfg.depth_model == "metric3d"
    assert cfg.scaling == "depth_ratio"
    assert cfg.run_name == "clip"
    assert cfg.run_dir.name == "clip"


def test_cli_overrides_yaml(tmp_path, video):
    config_file = tmp_path / "run.yaml"
    config_file.write_text(yaml.safe_dump({
        "video": str(video),
        "calibration": CALIBRATION,
        "camera_height": 1.5,
        "depth_model": "metric3d",
    }))

    args = build_parser().parse_args(
        ["--config", str(config_file), "--camera-height", "2.0"]
    )
    cfg = RunConfig.from_args(args)

    assert cfg.camera_height == 2.0        # flag wins
    assert cfg.depth_model == "metric3d"   # yaml survives


def test_yaml_rejects_unknown_keys(tmp_path, video):
    config_file = tmp_path / "run.yaml"
    config_file.write_text(yaml.safe_dump({
        "video": str(video), "calibration": CALIBRATION, "nonsense": 1,
    }))
    args = build_parser().parse_args(["--config", str(config_file)])
    with pytest.raises(ValueError, match="unknown key"):
        RunConfig.from_args(args)


def test_missing_video_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="--video"):
        RunConfig(video=tmp_path / "absent.mp4", calibration=CALIBRATION)


@pytest.mark.parametrize("field,value", [
    ("depth_model", "banana"),
    ("scaling", "banana"),
    ("camera_height", -1.0),
    ("resize", 0.0),
    ("resize", 2.0),
    ("sas_stride", 0),
    ("random_patch_ratio", 1.5),
])
def test_validation_rejects(video, field, value):
    with pytest.raises(ValueError):
        _config(video, **{field: value})


def test_depthpro_needs_a_checkpoint(video):
    with pytest.raises(ValueError, match="depthpro-checkpoint"):
        _config(video, depth_model="depthpro")
