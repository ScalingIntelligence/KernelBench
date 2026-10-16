import torch
import torch.nn as nn

###########################################
# Control for the cached-output check: captures a CUDA graph on the first call with each input set
# and replays it on later calls with the same inputs. The replay computes the matmul again, so this
# is not a cache, although a repeated call enqueues only one graph launch (see issue #171).
###########################################

class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._graphs = {}

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        key = (A.data_ptr(), B.data_ptr())
        if key not in self._graphs:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                torch.matmul(A, B)  # warm up outside the capture
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = torch.matmul(A, B)
            self._graphs[key] = (graph, out, A, B)  # keep A and B alive so their addresses stay valid
        graph, out, _, _ = self._graphs[key]
        graph.replay()
        return out
