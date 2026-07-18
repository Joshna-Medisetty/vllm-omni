from contextlib import contextmanager

import torch


@contextmanager
def torch_cuda_wrapper():
    try:
        # replace cuda APIs with xpu APIs, this should work by default
        torch.cuda.Stream = torch.xpu.Stream
        torch.cuda.default_stream = torch.xpu.current_stream
        torch.cuda.current_stream = torch.xpu.current_stream
        torch.cuda.stream = torch.xpu.stream
        torch.cuda.set_stream = torch.xpu.set_stream

        # torch.xpu.Event does not accept the ``blocking`` kwarg that
        # torch.cuda.Event supports, so drop it here.
        def _xpu_event(*args, blocking=None, **kwargs):
            return torch.xpu.Event(*args, **kwargs)

        torch.cuda.Event = _xpu_event
        yield
    finally:
        pass
