import pytest
np = pytest.importorskip('numpy')
torch = pytest.importorskip('torch')
from torch.utils.data import Dataset

from run.h36m_runner import H36MRunner
from run.cmu_runner import CMURunner


class DummySummaryWriter:
    def add_scalar(self, *args, **kwargs):
        pass
    def close(self):
        pass


class DummyDataset(Dataset):
    def __init__(self, *args, input_n=10, output_n=10, dct_used=15, **kwargs):
        self.length = 2
        seq_len = input_n + output_n
        dims = {'p32': 32*3, 'p22': 22*3, 'p12': 12*3, 'p7': 7*3, 'p4': 4*3}
        self.gt_all_scales = {k: np.random.randn(self.length, v, seq_len).astype(np.float32) for k, v in dims.items()}
        self.input_all_scales = {k: np.random.randn(self.length, v, dct_used).astype(np.float32) for k, v in dims.items()}
        self.global_max = 1.0
        self.global_min = -1.0

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        gts = {k: v[idx] for k, v in self.gt_all_scales.items()}
        inputs = {k: self.input_all_scales[k][idx] for k in self.input_all_scales}
        return inputs, gts


class DummyDatasetCMU(Dataset):
    def __init__(self, *args, input_n=10, output_n=10, dct_used=15, **kwargs):
        self.length = 2
        seq_len = input_n + output_n
        dims = {'p32': 38*3, 'p22': 25*3, 'p12': 12*3, 'p7': 7*3, 'p4': 4*3}
        self.gt_all_scales = {k: np.random.randn(self.length, v, seq_len).astype(np.float32) for k, v in dims.items()}
        self.input_all_scales = {k: np.random.randn(self.length, v, dct_used).astype(np.float32) for k, v in dims.items()}
        self.global_max = 1.0
        self.global_min = -1.0

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        gts = {k: v[idx] for k, v in self.gt_all_scales.items()}
        inputs = {k: self.input_all_scales[k][idx] for k in self.input_all_scales}
        return inputs, gts


def test_h36m_runner_forward(monkeypatch):
    monkeypatch.setattr('run.h36m_runner.H36MMotionDataset', DummyDataset, raising=False)
    monkeypatch.setattr('run.h36m_runner.define_actions', lambda x: ['walk'])
    monkeypatch.setattr('run.h36m_runner.SummaryWriter', lambda *a, **k: DummySummaryWriter())

    runner = H36MRunner(device='cpu', epochs=1, batch_size=2)
    inputs, _ = next(iter(runner.train_loader))
    outputs = runner.model(inputs)

    assert isinstance(outputs, dict)
    assert outputs['p22'].shape[0] == 2
    assert outputs['p22'].shape[1] == 22*3
    assert outputs['p22'].shape[2] == runner.cfg.dct_n


def test_cmu_runner_forward(monkeypatch):
    monkeypatch.setattr('run.cmu_runner.CMUMotionDataset', DummyDatasetCMU, raising=False)
    monkeypatch.setattr('run.cmu_runner.define_actions_cmu', lambda x: ['walk'])
    monkeypatch.setattr('run.cmu_runner.SummaryWriter', lambda *a, **k: DummySummaryWriter())

    runner = CMURunner(device='cpu', epochs=1, batch_size=2)
    inputs, _ = next(iter(runner.train_loader))
    outputs = runner.model(inputs)

    assert isinstance(outputs, dict)
    assert outputs['p22'].shape[0] == 2
    assert outputs['p22'].shape[1] == 25*3
    assert outputs['p22'].shape[2] == runner.cfg.dct_n
