import ctypes as ct
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from comfy.text_encoders import rocmfpx_runtime
from comfy.text_encoders.rocmfpx_runtime import ContextParams, FpxSession, ModelParams


@pytest.fixture
def session():
    # Bypass DLL/model loading; exercise the real validation and ctypes batch path.
    result = FpxSession.__new__(FpxSession)
    result.hidden_size = 5120
    result.headless = True
    result.kv_type = 0
    result.flash_attn = 0
    result.model = 1
    result.context = None
    return result


def h3_inputs():
    packed = np.arange(3 * 40960, dtype=np.float64).reshape(3, 40960)[:, ::2]
    positions = np.array([[1, 1, 1], [0, 0, 1], [0, 1, 0], [0, 0, 0]], dtype=np.int64)
    return packed, positions


class BatchProbe:
    def __init__(self, packed, positions, status):
        self.packed = packed
        self.positions = positions
        self.status = status
        self.output = np.arange(3 * 5120, dtype=np.float32).reshape(1, 3, 5120)
        self.events = []

    def llama_context_default_params(self):
        return ContextParams()

    def llama_init_from_model(self, model, params):
        assert model == 1
        assert params.embeddings and params.n_seq_max == 1
        assert params.n_batch == params.n_ubatch == 3
        self.events.append("init")
        return 123

    def llama_decode(self, context, batch):
        assert context == 123 and batch.n_tokens == 3 and not batch.token
        np.testing.assert_array_equal(np.ctypeslib.as_array(batch.embd, shape=(3 * 20480,)), self.packed.astype(np.float32).ravel())
        np.testing.assert_array_equal(np.ctypeslib.as_array(batch.pos, shape=(12,)), self.positions.ravel())
        np.testing.assert_array_equal(np.ctypeslib.as_array(batch.n_seq_id, shape=(3,)), [1, 1, 1])
        np.testing.assert_array_equal(np.ctypeslib.as_array(batch.logits, shape=(3,)), [1, 1, 1])
        assert [batch.seq_id[i][0] for i in range(3)] == [0, 0, 0]
        self.events.append("decode")
        return self.status

    def llama_synchronize(self, context):
        assert context == 123
        self.events.append("sync")

    def llama_get_embeddings(self, context):
        assert context == 123
        self.events.append("output")
        return self.output.ctypes.data_as(ct.POINTER(ct.c_float))

    def llama_free(self, context):
        assert context == 123
        self.events.append("free")
        self.output.fill(np.nan)


@pytest.mark.parametrize("status", [0, 1])
def test_h3_batch_and_context_lifetime(session, status):
    packed, positions = h3_inputs()
    original_packed, original_positions = packed.copy(), positions.copy()
    session.llama = BatchProbe(packed, positions, status)
    expected = session.llama.output.copy()

    if status:
        with pytest.raises(RuntimeError, match="prefill failed: 1"):
            session.encode_embeddings(packed, positions)
        assert session.llama.events == ["init", "decode", "sync", "free"]
    else:
        hidden = session.encode_embeddings(packed, positions)
        np.testing.assert_array_equal(hidden, expected)
        assert not np.shares_memory(hidden, session.llama.output)
        assert session.llama.events == ["init", "decode", "sync", "output", "free"]

    assert session.context is None
    np.testing.assert_array_equal(packed, original_packed)
    np.testing.assert_array_equal(positions, original_positions)


@pytest.mark.parametrize("case, message", [
    ("width", "packed embeddings"),
    ("empty", "packed embeddings"),
    ("axes", "integer mRoPE positions"),
    ("float_positions", "integer mRoPE positions"),
    ("negative", "Invalid mRoPE positions"),
    ("overflow", "Invalid mRoPE positions"),
    ("fourth_axis", "Invalid mRoPE positions"),
    ("noncausal", "index-causal mask"),
])
def test_h3_invalid_inputs_fail_before_loading_a_context(session, case, message):
    packed, positions = h3_inputs()
    if case == "width":
        packed = packed[:, :-1]
    elif case == "empty":
        packed, positions = packed[:0], positions[:, :0]
    elif case == "axes":
        positions = positions[:3]
    elif case == "float_positions":
        positions = positions.astype(np.float32)
    elif case == "negative":
        positions[0, 0] = -1
    elif case == "overflow":
        positions[0, 0] = 2 ** 31
    elif case == "fourth_axis":
        positions[3, 0] = 1
    elif case == "noncausal":
        positions[:, 1] = positions[:, 0]

    with pytest.raises(ValueError, match=message):
        session.encode_embeddings(packed, positions)
    assert session.context is None


@pytest.fixture
def runtime(monkeypatch):
    environment = {"GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F": "1", "UNCHANGED": "value"}
    directories = []
    events = []
    ggml = Mock(_handle=10)
    ggml.ggml_version.return_value = b"0.11.1"
    ggml.ggml_commit.return_value = b"3fca7f4bb"
    llama = Mock(_handle=20)
    llama.llama_model_default_params.side_effect = ModelParams
    llama.llama_backend_init.side_effect = lambda: events.append("backend_init")
    llama.llama_backend_free.side_effect = lambda: events.append("backend_free")
    llama.llama_model_load_from_file.return_value = 1
    llama.llama_model_free.side_effect = lambda model: events.append("model_free")
    llama.llama_free.side_effect = lambda context: events.append("context_free")
    llama.llama_model_n_embd.return_value = 4096
    llama.llama_model_n_layer.return_value = 36
    llama.llama_vocab_n_tokens.return_value = 151936

    def metadata(model, key, value, size):
        value.value = b"qwen3vl"
        return len(value.value)

    def add_directory(path):
        handle = Mock()
        directories.append(handle)
        return handle

    llama.llama_model_meta_val_str.side_effect = metadata
    monkeypatch.setattr(rocmfpx_runtime, "os", SimpleNamespace(
        name="nt", environ=environment, fsencode=lambda value: str(value).encode(),
        add_dll_directory=add_directory))
    monkeypatch.setattr(rocmfpx_runtime.importlib.util, "find_spec", lambda name: None)
    loader = Mock(side_effect=lambda path: ggml if path.endswith("ggml-base.dll") else llama)
    monkeypatch.setattr(rocmfpx_runtime.ct, "CDLL", loader)
    monkeypatch.setattr(rocmfpx_runtime, "_backend_users", {})
    return SimpleNamespace(environment=environment, ggml=ggml, llama=llama,
                           directories=directories, events=events, loader=loader)


@pytest.mark.parametrize("flags", [
    {},
    {"GGML_CUDA_FORCE_CUBLAS_COMPUTE_16F": "1"},
    {"GGML_CUDA_FORCE_CUBLAS_COMPUTE_16F": "0", "GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F": "1"},
    {"GGML_CUDA_FORCE_CUBLAS_COMPUTE_16F": "", "GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F": "1"},
])
def test_conflicting_precision_flags_are_preserved_and_rejected(runtime, flags):
    runtime.environment.clear()
    runtime.environment.update(flags)
    before = runtime.environment.copy()
    with pytest.raises(RuntimeError, match="restart ComfyUI"):
        FpxSession("model.gguf", "dlls", "qwen_image_21")
    assert runtime.environment == before
    runtime.loader.assert_not_called()
    assert not runtime.directories


@pytest.mark.parametrize("value", ["1", "0", ""])
def test_precision_flags_are_never_changed(runtime, value):
    # Upstream getenv checks presence, not the string value.
    runtime.environment["GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F"] = value
    before = runtime.environment.copy()
    session = FpxSession("model.gguf", "dlls", "qwen_image_21")
    assert runtime.environment == before
    session.close()
    assert runtime.environment == before
    assert not rocmfpx_runtime._backend_users


def test_backend_is_kept_until_last_session_closes(runtime):
    first = FpxSession("first.gguf", "dlls", "qwen_image_21")
    second = FpxSession("second.gguf", "dlls", "qwen_image_21")
    runtime.llama.llama_backend_init.assert_called_once()
    first.context = 123
    first.close()
    first.close()
    assert runtime.events == ["backend_init", "context_free", "model_free"]
    assert rocmfpx_runtime._backend_users == {10: 1}
    second.close()
    second.close()
    assert runtime.events == ["backend_init", "context_free", "model_free", "model_free", "backend_free"]
    assert not rocmfpx_runtime._backend_users
    for directory in runtime.directories:
        directory.close.assert_called_once()

    reloaded = FpxSession("third.gguf", "dlls", "qwen_image_21")
    assert runtime.llama.llama_backend_init.call_count == 2
    reloaded.close()
    assert runtime.llama.llama_backend_free.call_count == 2


def test_failed_second_session_does_not_free_live_backend(runtime):
    first = FpxSession("first.gguf", "dlls", "qwen_image_21")
    runtime.llama.llama_model_load_from_file.return_value = None
    before = runtime.environment.copy()
    with pytest.raises(RuntimeError, match="could not load"):
        FpxSession("broken.gguf", "dlls", "qwen_image_21")
    assert runtime.environment == before
    runtime.llama.llama_backend_free.assert_not_called()
    assert rocmfpx_runtime._backend_users == {10: 1}
    runtime.directories[1].close.assert_called_once()
    first.close()
    runtime.llama.llama_backend_free.assert_called_once()
    assert not rocmfpx_runtime._backend_users


@pytest.mark.parametrize("handle", [10, 11])
def test_backend_ownership_uses_native_handle_not_python_wrapper(runtime, handle):
    first = FpxSession("first.gguf", "dlls", "qwen_image_21")
    second_ggml = Mock(_handle=handle)
    second_ggml.ggml_version.return_value = b"0.11.1"
    second_ggml.ggml_commit.return_value = b"3fca7f4bb"
    runtime.loader.side_effect = lambda path: second_ggml if path.endswith("ggml-base.dll") else runtime.llama
    second = FpxSession("second.gguf", "dlls", "qwen_image_21")
    assert runtime.llama.llama_backend_init.call_count == (1 if handle == 10 else 2)
    first.close()
    assert runtime.llama.llama_backend_free.call_count == (0 if handle == 10 else 1)
    second.close()
    assert runtime.llama.llama_backend_free.call_count == (1 if handle == 10 else 2)
    assert not rocmfpx_runtime._backend_users


@pytest.mark.parametrize("failure", ["version", "load", "dimensions"])
def test_failed_initialization_releases_only_owned_resources(runtime, failure):
    if failure == "version":
        runtime.ggml.ggml_commit.return_value = b"unsupported"
    elif failure == "load":
        runtime.llama.llama_model_load_from_file.return_value = None
    else:
        runtime.llama.llama_model_n_embd.return_value = 5120
    before = runtime.environment.copy()
    with pytest.raises((RuntimeError, ValueError)):
        FpxSession("model.gguf", "dlls", "qwen_image_21")
    assert runtime.environment == before
    assert not rocmfpx_runtime._backend_users
    runtime.directories[0].close.assert_called_once()
    assert runtime.llama.llama_backend_free.call_count == (failure != "version")
    assert runtime.llama.llama_model_free.call_count == (failure == "dimensions")
