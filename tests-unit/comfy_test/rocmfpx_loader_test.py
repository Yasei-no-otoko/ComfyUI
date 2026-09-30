from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import safetensors.torch
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

from comfy.text_encoders import rocmfpx
from comfy_extras import nodes_rocmfpx


@pytest.fixture
def loader(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.version, "hip", "test")
    (tmp_path / "llama.dll").touch()
    gguf_path = tmp_path / "model.gguf"
    gguf_path.touch()
    captured = {}

    def init_clip(self, **kwargs):
        captured.update(kwargs)
        self.patcher = SimpleNamespace(load_device=torch.device("cpu"), offload_device=torch.device("cpu"))

    monkeypatch.setattr(rocmfpx.comfy.sd.CLIP, "__init__", init_clip)
    monkeypatch.setattr(rocmfpx, "FpxModelPatcher", Mock())
    return SimpleNamespace(gguf_path=gguf_path, directory=tmp_path, captured=captured)


@pytest.mark.parametrize("extension", [".safetensors", ".sft"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("include_decoder", [False, True])
def test_h3_companion_uses_retained_embedding_dtype(loader, extension, dtype, include_decoder):
    companion_path = loader.directory / ("companion" + extension)
    retained = {
        "model.embed_tokens.weight": torch.ones(2, 3, dtype=dtype),
        "visual.patch_embed.proj.weight": torch.ones(2, 3, dtype=torch.float32),
    }
    state_dict = dict(retained)
    if include_decoder:
        state_dict["model.layers.0.input_layernorm.weight"] = torch.ones(3, dtype=torch.float64)
    safetensors.torch.save_file(state_dict, companion_path)

    rocmfpx.ROCmFPXCLIP(loader.gguf_path, loader.directory, "minimax_h3", companion_path)

    assert loader.captured["model_options"]["dtype"] == dtype
    assert loader.captured["target"].clip is rocmfpx.H3TE
    loaded = loader.captured["state_dict"][0]
    assert loaded.keys() == retained.keys()
    for key, tensor in retained.items():
        torch.testing.assert_close(loaded[key], tensor)


def test_h3_companion_missing_embeddings_has_clear_error(loader):
    companion_path = loader.directory / "companion.safetensors"
    safetensors.torch.save_file({
        "visual.patch_embed.proj.weight": torch.ones(2, 3),
        "model.layers.0.input_layernorm.weight": torch.ones(3),
    }, companion_path)

    with pytest.raises(ValueError, match=r"MiniMax H3 companion is missing model\.embed_tokens\.weight"):
        rocmfpx.ROCmFPXCLIP(loader.gguf_path, loader.directory, "minimax_h3", companion_path)
    assert not loader.captured


@pytest.mark.parametrize("filename", ["companion.pt", "companion.ckpt", "companion.bin", "companion", None])
def test_h3_companion_rejects_unsupported_format_before_open(loader, monkeypatch, filename):
    companion_path = loader.directory / filename if filename is not None else None
    safe_open = Mock(side_effect=AssertionError("Unsupported companion must not be opened"))
    monkeypatch.setattr(rocmfpx.safetensors, "safe_open", safe_open)

    with pytest.raises(ValueError, match=r"MiniMax H3 companion must be a safetensors"):
        rocmfpx.ROCmFPXCLIP(loader.gguf_path, loader.directory, "minimax_h3", companion_path)
    safe_open.assert_not_called()
    assert not loader.captured


def test_companion_options_only_include_safetensors(monkeypatch):
    filenames = ["encoder.pt", "encoder.ckpt", "encoder.bin", "b.sft", "a.safetensors", "nested/c.SAFETENSORS"]
    monkeypatch.setattr(nodes_rocmfpx.folder_paths, "get_filename_list",
                        lambda folder: filenames if folder == "text_encoders" else ["model.gguf"])

    schema = nodes_rocmfpx.ROCmFPXCLIPLoader.define_schema()
    companion = next(input for input in schema.inputs if input.id == "companion_name")

    assert companion.options == ["none", "a.safetensors", "b.sft", "nested/c.SAFETENSORS"]


def test_node_rejects_companion_format_when_combo_is_bypassed(loader, monkeypatch):
    companion_path = loader.directory / "companion.pt"
    safetensors.torch.save_file({"model.embed_tokens.weight": torch.ones(2, 3)}, companion_path)
    monkeypatch.setenv("ROCMFPX_LIBRARY_PATH", str(loader.directory))
    monkeypatch.setitem(nodes_rocmfpx.folder_paths.folder_names_and_paths,
                        "rocmfpx_text_encoders", ([str(loader.directory)], {".gguf"}))
    monkeypatch.setitem(nodes_rocmfpx.folder_paths.folder_names_and_paths,
                        "text_encoders", ([str(loader.directory)], {".pt", ".safetensors", ".sft"}))

    with pytest.raises(ValueError, match=r"MiniMax H3 companion must be a safetensors"):
        nodes_rocmfpx.ROCmFPXCLIPLoader.execute("model.gguf", "minimax_h3", "companion.pt")
    assert not loader.captured


def test_qwen_image_does_not_require_companion(loader):
    rocmfpx.ROCmFPXCLIP(loader.gguf_path, loader.directory, "qwen_image_21")

    assert loader.captured["target"].clip is rocmfpx.QwenImageTE
    assert loader.captured["state_dict"] == []
    assert loader.captured["model_options"]["dtype"] == torch.float32
