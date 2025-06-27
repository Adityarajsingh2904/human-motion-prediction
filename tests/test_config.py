import os
import sys
import pytest

from configs import config


def test_parse_args_defaults(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog"])
    args = config.parse_args()
    assert args.exp_name == "cmu"
    assert args.input_n == 10
    assert args.data_dir == ""


def test_parse_args_custom(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "--exp_name", "h36m", "--input_n", "5", "--data_dir", "/tmp"])
    args = config.parse_args()
    assert args.exp_name == "h36m"
    assert args.input_n == 5
    assert args.data_dir == "/tmp"


def test_config_requires_data_dir(monkeypatch):
    monkeypatch.delenv("H36M_DATA_DIR", raising=False)
    with pytest.raises(ValueError):
        config.Config(exp_name="h36m")


def test_config_env_var(monkeypatch, tmp_path):
    monkeypatch.setenv("CMU_DATA_DIR", str(tmp_path))
    cfg = config.Config(exp_name="cmu")
    assert cfg.base_data_dir == os.path.abspath(str(tmp_path))
