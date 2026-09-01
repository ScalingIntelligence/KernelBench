import os
from typing import Callable, Optional, Union

import torch
import torch.distributed as dist


_LOCAL_RANK = None


def init_dist(local_rank: int, num_local_ranks: int):
    ip = os.getenv("MASTER_ADDR", "127.0.0.1")
    port = int(os.getenv("MASTER_PORT", "8361"))
    num_nodes = int(os.getenv("WORLD_SIZE", 1))
    node_rank = int(os.getenv("RANK", 0))

    global _LOCAL_RANK
    _LOCAL_RANK = local_rank

    device = torch.device("musa", local_rank)
    dist.init_process_group(
        backend="mccl",
        init_method=f"tcp://{ip}:{port}",
        world_size=num_nodes * num_local_ranks,
        rank=node_rank * num_local_ranks + local_rank,
        device_id=device,
    )
    torch.musa.set_device(local_rank)
    return (
        dist.get_rank(),
        dist.get_world_size(),
        dist.new_group(list(range(num_local_ranks * num_nodes)), backend="mccl"),
    )


def dist_print(message: str = "", once_in_node: bool = False) -> None:
    assert _LOCAL_RANK is not None
    if not once_in_node or _LOCAL_RANK == 0:
        print(message, flush=True)
    dist.barrier()


class empty_suppress:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


class suppress_stdout_stderr:
    def __enter__(self):
        self.outnull_file = open(os.devnull, "w")
        self.errnull_file = open(os.devnull, "w")
        self.old_stdout_fileno_undup = os.dup(1)
        self.old_stderr_fileno_undup = os.dup(2)
        os.dup2(self.outnull_file.fileno(), 1)
        os.dup2(self.errnull_file.fileno(), 2)
        return self

    def __exit__(self, *_):
        os.dup2(self.old_stdout_fileno_undup, 1)
        os.dup2(self.old_stderr_fileno_undup, 2)
        os.close(self.old_stdout_fileno_undup)
        os.close(self.old_stderr_fileno_undup)
        self.outnull_file.close()
        self.errnull_file.close()


def _profiler_event_time_range_us(event):
    time_range = getattr(event, "time_range", None)
    if time_range is not None:
        start = getattr(time_range, "start", None)
        if start is not None:
            elapsed = (
                time_range.elapsed_us()
                if callable(getattr(time_range, "elapsed_us", None))
                else getattr(time_range, "elapsed_us", None)
            )
            if elapsed is not None:
                return float(start), float(elapsed)

    start = getattr(event, "start_us", None)
    duration = getattr(event, "duration_us", None)
    if start is not None and duration is not None:
        return float(start), float(duration)
    return None


def _extract_profiler_kernel_events(prof, kernel_name: str):
    kernel_events = []
    for event in prof.events():
        if kernel_name not in getattr(event, "name", ""):
            continue
        time_range = _profiler_event_time_range_us(event)
        if time_range is None:
            continue
        start_us, duration_us = time_range
        kernel_events.append(
            {
                "start_us": start_us,
                "duration_us": duration_us,
                "end_us": start_us + duration_us,
            }
        )
    return sorted(kernel_events, key=lambda event: event["start_us"])


def _average_aligned_kernel_duration(kernel_events, group):
    rank_kernel_events = [None for _ in range(dist.get_world_size(group))]
    dist.all_gather_object(rank_kernel_events, kernel_events, group=group)
    assert all(rank_kernel_events)
    num_iters = min(len(events) for events in rank_kernel_events)
    assert num_iters > 0

    aligned_duration_us = 0.0
    for iter_idx in range(num_iters):
        aligned_duration_us += min(
            events[iter_idx]["duration_us"] for events in rank_kernel_events
        )
    return aligned_duration_us / num_iters / 1e6


def bench_kineto(
    fn,
    kernel_names: Union[str, tuple],
    num_tests: int = 5,
    suppress_kineto_output: bool = False,
    trace_path: Optional[str] = None,
    barrier_comm_profiling: bool = False,
    barrier: Optional[Callable] = None,
    aligned_kernel_duration: bool = False,
    group=None,
):
    suppress = suppress_stdout_stderr if suppress_kineto_output else empty_suppress
    with suppress():
        schedule = torch.profiler.schedule(wait=1, warmup=0, active=1, repeat=1)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.MUSA], schedule=schedule
        ) as prof:
            for _ in range(2):
                if barrier_comm_profiling:
                    lhs = torch.randn((8192, 8192), dtype=torch.float, device="musa")
                    rhs = torch.randn((8192, 8192), dtype=torch.float, device="musa")
                    lhs @ rhs
                    dist.all_reduce(torch.ones(1, dtype=torch.float, device="musa"))
                for _ in range(num_tests):
                    torch.empty(int(8e9 // 4), dtype=torch.int, device="musa").zero_()
                    if barrier is not None:
                        barrier()
                    fn()
                prof.step()

    is_tuple = isinstance(kernel_names, tuple)
    kernel_specs = (kernel_names,) if isinstance(kernel_names, str) else kernel_names
    prof_lines = (
        prof.key_averages()
        .table(sort_by="count", max_name_column_width=100)
        .split("\n")
    )
    units = {"ms": 1e3, "us": 1e6}
    kernel_durations = []
    selected_kernel_names = []

    for spec in kernel_specs:
        candidates = (spec,) if isinstance(spec, str) else spec
        for name in candidates:
            found = False
            for line in prof_lines:
                if name not in line:
                    continue
                time_str = line.split()[-2]
                for unit, scale in units.items():
                    if unit in time_str:
                        kernel_durations.append(
                            float(time_str.replace(unit, "")) / scale
                        )
                        selected_kernel_names.append(name)
                        found = True
                        break
                if found:
                    break
            if found:
                break
        else:
            event_names = sorted(
                {
                    getattr(event, "name", "")
                    for event in prof.events()
                    if getattr(event, "name", "")
                }
            )
            raise RuntimeError(
                f"Profiler did not find kernel name(s): {candidates}. Available events: {event_names[:64]}"
            )

    if trace_path is not None:
        prof.export_chrome_trace(trace_path)

    result = kernel_durations if is_tuple else kernel_durations[0]
    if aligned_kernel_duration:
        assert not is_tuple
        assert group is not None
        events = _extract_profiler_kernel_events(prof, selected_kernel_names[0])
        return _average_aligned_kernel_duration(events, group)
    return result
