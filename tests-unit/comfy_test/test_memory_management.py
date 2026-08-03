from contextlib import nullcontext
from types import SimpleNamespace

from comfy.memory_management import TensorFileSlice, read_tensor_file_slice_into


class FakeStorage:
    pass


class FakeTensor:
    def __init__(self, device_type, data_ptr, size=16, device_index=None):
        self.device = SimpleNamespace(type=device_type, index=device_index)
        self._data_ptr = data_ptr
        self._size = size
        self._storage = FakeStorage()
        self.copy_calls = []

    def untyped_storage(self):
        return self._storage

    def numel(self):
        return self._size

    def element_size(self):
        return 1

    def storage_offset(self):
        return 0

    def is_contiguous(self):
        return True

    def data_ptr(self):
        return self._data_ptr

    def copy_(self, source, non_blocking=False):
        self.copy_calls.append((source, non_blocking))


class FakeHostBuffer:
    def __init__(self, address):
        self.address = address
        self.calls = []

    def get_raw_address(self):
        return self.address

    def read_file_slice(self, *args, **kwargs):
        self.calls.append((args, kwargs))


def file_backed_tensor(size=16):
    tensor = FakeTensor("cpu", 0, size=size)
    tensor._storage._comfy_tensor_file_slice = TensorFileSlice(object(), nullcontext(), 32, size)
    return tensor


def test_host_buffer_uses_blocking_copy_when_no_stream_is_available():
    tensor = file_backed_tensor()
    destination = FakeTensor("cpu", 1040)
    hostbuf = FakeHostBuffer(1024)
    destination._storage._comfy_hostbuf = hostbuf
    device_destination = FakeTensor("cuda", 4096, device_index=0)

    assert read_tensor_file_slice_into(tensor, destination, destination2=device_destination)

    _, kwargs = hostbuf.calls[0]
    assert kwargs["offset"] == 16
    assert kwargs["stream"] == 0
    assert kwargs["device_ptr"] == 0
    assert device_destination.copy_calls == [(destination, False)]


def test_host_buffer_streams_directly_to_device_with_non_default_stream():
    tensor = file_backed_tensor()
    destination = FakeTensor("cpu", 1040)
    hostbuf = FakeHostBuffer(1024)
    destination._storage._comfy_hostbuf = hostbuf
    device_destination = FakeTensor("cuda", 4096, device_index=0)
    stream = SimpleNamespace(cuda_stream=1234)

    assert read_tensor_file_slice_into(tensor, destination, stream=stream, destination2=device_destination)

    _, kwargs = hostbuf.calls[0]
    assert kwargs["stream"] == 1234
    assert kwargs["device_ptr"] == 4096
    assert device_destination.copy_calls == []
