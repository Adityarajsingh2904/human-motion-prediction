import os
import pytest

from datas_dct.h36m.motiondataset import MotionDataset as H36MMotionDataset
from datas_dct.cmu.motiondataset import MotionDataset as CMUMotionDataset


def test_h36m_dataset_missing_dir(tmp_path):
    missing = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        H36MMotionDataset(str(missing), actions="walk", mode_name="train", input_n=10, output_n=10,
                          dct_used=15, split=0, sample_rate=2, test_manner="all", device="cpu")


def test_cmu_dataset_missing_dir(tmp_path):
    missing = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        CMUMotionDataset(str(missing), actions="walk", mode_name="train", input_n=10, output_n=10,
                         dct_used=15, split=0, sample_rate=2, test_manner="all", device="cpu")

