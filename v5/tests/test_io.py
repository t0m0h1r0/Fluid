import pytest
import numpy as np
import os
import tempfile
from src.io.checkpoint import CheckpointHandler
from src.io.visualizer import Visualizer

@pytest.fixture
def sample_state():
    return {
        'time': 0.1,
        'step': 100,
        'fields': {
            'phi': np.random.randn(32, 32, 64),
            'pressure': np.random.randn(32, 32, 64),
            'velocity': [
                np.random.randn(32, 32, 64),
                np.random.randn(32, 32, 64),
                np.random.randn(32, 32, 64)
            ]
        }
    }

def test_checkpoint_save_load(sample_state):
    with tempfile.TemporaryDirectory() as tmpdir:
        handler = CheckpointHandler(tmpdir)
        
        # 保存
        filename = handler.save(sample_state, sample_state['step'])
        assert os.path.exists(filename)
        
        # 読み込み
        loaded_state = handler.load(filename)
        
        # 検証
        assert loaded_state['time'] == sample_state['time']
        assert loaded_state['step'] == sample_state['step']
        assert np.allclose(loaded_state['fields']['phi'], 
                          sample_state['fields']['phi'])
        assert np.allclose(loaded_state['fields']['pressure'], 
                          sample_state['fields']['pressure'])
        for v1, v2 in zip(loaded_state['fields']['velocity'], 
                         sample_state['fields']['velocity']):
            assert np.allclose(v1, v2)

def test_visualizer(sample_state):
    with tempfile.TemporaryDirectory() as tmpdir:
        vis = Visualizer(tmpdir)
        
        fields = {
            'phi': sample_state['fields']['phi'],
            'p': sample_state['fields']['pressure'],
            'u': sample_state['fields']['velocity'][0],
            'v': sample_state['fields']['velocity'][1],
            'w': sample_state['fields']['velocity'][2]
        }
        
        vis.save_plots(fields, sample_state['time'], sample_state['step'])
        
        # 出力ファイルの確認
        assert os.path.exists(f"{tmpdir}/phase_{sample_state['step']:06d}.png")
        assert os.path.exists(f"{tmpdir}/velocity_{sample_state['step']:06d}.png")
        assert os.path.exists(f"{tmpdir}/pressure_{sample_state['step']:06d}.png")