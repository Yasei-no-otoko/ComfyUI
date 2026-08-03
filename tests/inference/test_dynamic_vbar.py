import torch

import comfy.model_patcher
import comfy.ops


def test_dynamic_vbar_reserves_only_streamed_parameter_storage():
    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.large = comfy.ops.disable_weight_init.Linear(128, 128)
            self.small = comfy.ops.disable_weight_init.Linear(2, 2)

    patcher = object.__new__(comfy.model_patcher.ModelPatcherDynamic)
    patcher.model = ToyModel()
    patcher.patches = {}

    loading = patcher._load_list(for_dynamic=True, default_device=torch.device("cpu"))

    weight_size = 128 * 128 * torch.empty((), dtype=torch.float32).element_size()
    expected = ((weight_size + 1023) // 1024 * 1024) + 1024
    assert patcher._vbar_size(loading) == expected
