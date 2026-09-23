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


def test_run_name_expands_strftime_codes(video):
    import re

    cfg = _config(video, run_name="%Y_%m_%d_%H_%M_%S")
    assert re.fullmatch(r"\d{4}(_\d{2}){5}", cfg.run_name)
    assert cfg.run_dir == cfg.output / cfg.run_name


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


def test_calibration_not_required_in_360_mode(video):
    cfg = RunConfig(video=video, three_sixty_camera_model="gopromax")
    assert cfg.calibration is None


def test_calibration_still_required_without_360_mode(video):
    with pytest.raises(FileNotFoundError, match="--calibration"):
        RunConfig(video=video)


def test_360_mode_rejects_unknown_camera_without_params(video):
    with pytest.raises(ValueError, match="not a known camera"):
        RunConfig(video=video, three_sixty_camera_model="insta360x3")


def test_360_mode_accepts_custom_params(video):
    cfg = RunConfig(
        video=video,
        three_sixty_camera_model="custom",
        three_sixty_camera_params="fov_h=180,fov_v=90",
    )
    assert cfg.calibration is None


def test_360_bare_flag_defaults_to_gopromax(video):
    args = build_parser().parse_args([str(video), "--360-camera-model"])
    cfg = RunConfig.from_args(args)
    assert cfg.three_sixty_camera_model == "gopromax"


def test_from_cli_missing_calibration_mentions_360_escape_hatch(video):
    args = build_parser().parse_args([str(video)])
    with pytest.raises(ValueError, match="--360-camera-model"):
        RunConfig.from_args(args)


@pytest.mark.parametrize("key", ["gpx", "gps_csv", "gt_trajectory"])
def test_config_with_ground_truth_triggers_evaluation(tmp_path, key):
    """video_to_trajectory.py evaluates by itself when its config names ground truth."""
    from gpx_evaluation import config_has_ground_truth

    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump({"video": "clip.mp4", key: "truth.file"}))
    assert config_has_ground_truth(path)


def test_config_without_ground_truth_skips_evaluation(tmp_path):
    from gpx_evaluation import config_has_ground_truth

    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump({"video": "clip.mp4", "gpx": None}))
    assert not config_has_ground_truth(path)
