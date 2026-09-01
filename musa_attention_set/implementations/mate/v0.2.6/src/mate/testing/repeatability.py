from __future__ import annotations

import copy
import functools
import inspect
import random
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeVar, cast

import torch

F = TypeVar("F", bound=Callable[..., Any])
CompareMode = Literal["bitwise", "equal"]
Synchronize = bool | Callable[[], None]


@dataclass
class _RngState:
    python: object
    numpy: object | None
    torch_cpu: torch.Tensor
    accelerators: dict[str, tuple[torch.Tensor, ...]]


@dataclass
class _TensorSnapshot:
    value: torch.Tensor
    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    layout: torch.layout
    stride: tuple[int, ...] | None


@dataclass
class _SequenceSnapshot:
    value_type: type[Any]
    values: tuple[Any, ...]


@dataclass
class _MappingSnapshot:
    value_type: type[Any]
    values: tuple[tuple[Any, Any], ...]


@dataclass
class _LeafSnapshot:
    value_type: type[Any]
    value: Any


@dataclass
class _ScopedCall:
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    rng_state: _RngState | None
    captured: dict[str, Any] | None = None
    missing_names: tuple[str, ...] = ()
    available_names: tuple[str, ...] = ()


def _normalize_output_names(outputs: str | Sequence[str]) -> tuple[str, ...]:
    names = (outputs,) if isinstance(outputs, str) else tuple(outputs)
    if not names:
        raise ValueError("outputs must contain at least one local variable name")
    for index, name in enumerate(names):
        if not isinstance(name, str) or not name:
            raise TypeError(
                f"outputs[{index}] must be a non-empty string, got {name!r}"
            )
    if len(set(names)) != len(names):
        raise ValueError(f"outputs must not contain duplicate names, got {names!r}")
    return names


def _get_numpy_rng_state() -> object | None:
    try:
        import numpy as np
    except ImportError:
        return None
    return copy.deepcopy(np.random.get_state())


def _set_numpy_rng_state(state: object | None) -> None:
    if state is None:
        return
    try:
        import numpy as np
    except ImportError:
        return
    np.random.set_state(cast(tuple[Any, ...], state))


def _capture_rng_state() -> _RngState:
    accelerator_states: dict[str, tuple[torch.Tensor, ...]] = {}
    for backend_name in ("cuda", "musa", "xpu"):
        backend = getattr(torch, backend_name, None)
        if backend is None:
            continue
        is_available = getattr(backend, "is_available", None)
        get_rng_state_all = getattr(backend, "get_rng_state_all", None)
        set_rng_state_all = getattr(backend, "set_rng_state_all", None)
        if not callable(is_available) or not callable(get_rng_state_all):
            continue
        if not callable(set_rng_state_all) or not is_available():
            continue
        accelerator_states[backend_name] = tuple(
            state.clone() for state in get_rng_state_all()
        )

    return _RngState(
        python=random.getstate(),
        numpy=_get_numpy_rng_state(),
        torch_cpu=torch.random.get_rng_state().clone(),
        accelerators=accelerator_states,
    )


def _restore_rng_state(state: _RngState) -> None:
    random.setstate(cast(tuple[Any, ...], state.python))
    _set_numpy_rng_state(state.numpy)
    torch.random.set_rng_state(state.torch_cpu)
    for backend_name, states in state.accelerators.items():
        backend = getattr(torch, backend_name)
        backend.set_rng_state_all(list(states))


def _iter_tensors(value: Any):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _iter_tensors(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _iter_tensors(item)


def _synchronize_outputs(values: Mapping[str, Any], synchronize: Synchronize) -> None:
    if callable(synchronize):
        synchronize()
        return
    if not synchronize:
        return

    devices = {
        tensor.device for value in values.values() for tensor in _iter_tensors(value)
    }
    for device in sorted(devices, key=str):
        if device.type == "cpu":
            continue
        backend = getattr(torch, device.type, None)
        backend_synchronize = getattr(backend, "synchronize", None)
        if callable(backend_synchronize):
            backend_synchronize(device)


def _check_finite(value: Any, *, path: str, iteration: int, repeat: int) -> None:
    for tensor in _iter_tensors(value):
        if not (tensor.is_floating_point() or tensor.is_complex()):
            continue
        finite = torch.isfinite(tensor)
        if bool(finite.all().item()):
            continue
        non_finite = int((~finite).sum().item())
        raise AssertionError(
            f"repeat_check found non-finite values at {path} on "
            f"repeat={iteration + 1}/{repeat}: {non_finite}/{tensor.numel()}"
        )


def _snapshot(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        stride = tuple(value.stride()) if value.layout == torch.strided else None
        return _TensorSnapshot(
            value=value.detach().clone(),
            shape=tuple(value.shape),
            dtype=value.dtype,
            device=value.device,
            layout=value.layout,
            stride=stride,
        )
    if isinstance(value, Mapping):
        return _MappingSnapshot(
            value_type=type(value),
            values=tuple(
                (copy.deepcopy(key), _snapshot(item)) for key, item in value.items()
            ),
        )
    if isinstance(value, (tuple, list)):
        return _SequenceSnapshot(
            value_type=type(value),
            values=tuple(_snapshot(item) for item in value),
        )
    return _LeafSnapshot(value_type=type(value), value=copy.deepcopy(value))


def _tensor_metadata(snapshot: _TensorSnapshot) -> tuple[Any, ...]:
    return (
        snapshot.shape,
        snapshot.dtype,
        snapshot.device,
        snapshot.layout,
        snapshot.stride,
    )


def _tensor_mismatch_detail(baseline: torch.Tensor, current: torch.Tensor) -> str:
    if baseline.numel() == 0:
        return "changed=0/0"
    changed = int((baseline != current).sum().item())
    detail = f"changed={changed}/{baseline.numel()}"
    if baseline.is_floating_point() or baseline.is_complex():
        max_abs = float((baseline - current).abs().max().item())
        detail += f", max_abs={max_abs:.9g}"
    return detail


def _compare_snapshots(
    baseline: Any,
    current: Any,
    *,
    path: str,
    iteration: int,
    repeat: int,
    compare: CompareMode,
) -> None:
    prefix = f"repeat_check mismatch at {path} on repeat={iteration + 1}/{repeat}"
    if type(baseline) is not type(current):
        raise AssertionError(
            f"{prefix}: snapshot types differ "
            f"({type(baseline).__name__} != {type(current).__name__})"
        )

    if isinstance(baseline, _TensorSnapshot):
        if _tensor_metadata(baseline) != _tensor_metadata(current):
            raise AssertionError(
                f"{prefix}: tensor metadata differs "
                f"({_tensor_metadata(baseline)!r} != {_tensor_metadata(current)!r})"
            )
        # The existing MATE repeatability tests use torch.equal as their
        # bitwise-repeatability contract. Keep that behavior for compatibility.
        if not torch.equal(baseline.value, current.value):
            detail = _tensor_mismatch_detail(baseline.value, current.value)
            raise AssertionError(f"{prefix}: {detail} (compare={compare})")
        return

    if isinstance(baseline, _SequenceSnapshot):
        if baseline.value_type is not current.value_type:
            raise AssertionError(
                f"{prefix}: container types differ "
                f"({baseline.value_type.__name__} != {current.value_type.__name__})"
            )
        if len(baseline.values) != len(current.values):
            raise AssertionError(
                f"{prefix}: sequence lengths differ "
                f"({len(baseline.values)} != {len(current.values)})"
            )
        for index, (baseline_item, current_item) in enumerate(
            zip(baseline.values, current.values)
        ):
            _compare_snapshots(
                baseline_item,
                current_item,
                path=f"{path}[{index}]",
                iteration=iteration,
                repeat=repeat,
                compare=compare,
            )
        return

    if isinstance(baseline, _MappingSnapshot):
        if baseline.value_type is not current.value_type:
            raise AssertionError(
                f"{prefix}: mapping types differ "
                f"({baseline.value_type.__name__} != {current.value_type.__name__})"
            )
        baseline_keys = tuple(key for key, _ in baseline.values)
        current_keys = tuple(key for key, _ in current.values)
        if baseline_keys != current_keys:
            raise AssertionError(
                f"{prefix}: mapping keys differ ({baseline_keys!r} != {current_keys!r})"
            )
        for (key, baseline_item), (_, current_item) in zip(
            baseline.values, current.values
        ):
            _compare_snapshots(
                baseline_item,
                current_item,
                path=f"{path}[{key!r}]",
                iteration=iteration,
                repeat=repeat,
                compare=compare,
            )
        return

    if not isinstance(baseline, _LeafSnapshot):
        raise AssertionError(f"{prefix}: unsupported snapshot type {type(baseline)!r}")
    if baseline.value_type is not current.value_type:
        raise AssertionError(
            f"{prefix}: value types differ "
            f"({baseline.value_type.__name__} != {current.value_type.__name__})"
        )
    try:
        is_equal = bool(baseline.value == current.value)
    except (TypeError, ValueError):
        is_equal = False
    if not is_equal:
        raise AssertionError(
            f"{prefix}: values differ ({baseline.value!r} != {current.value!r})"
        )


def _call_and_capture(
    function: F,
    target_code: Any,
    target_name: str,
    output_names: tuple[str, ...],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    captures: list[dict[str, Any]] = []
    missing_names: list[str] = []
    available_names: list[str] = []
    previous_profile = sys.getprofile()

    def profile(frame, event, arg):
        if previous_profile is not None:
            previous_profile(frame, event, arg)
        if frame.f_code is not target_code or event != "return":
            return

        available_names[:] = sorted(frame.f_locals)
        missing_names[:] = [name for name in output_names if name not in frame.f_locals]
        if not missing_names:
            captures.append({name: frame.f_locals[name] for name in output_names})

    try:
        sys.setprofile(profile)
        result = function(*args, **kwargs)
    finally:
        sys.setprofile(previous_profile)

    if missing_names:
        raise AssertionError(
            "repeat_check could not capture local variable(s) "
            f"{missing_names!r} from {target_name}; "
            f"available locals: {available_names!r}"
        )
    if len(captures) != 1:
        raise AssertionError(
            f"repeat_check expected one return from {target_name}, "
            f"captured {len(captures)}"
        )
    return result, captures[0]


def _arguments_from_frame(
    frame: Any, signature: inspect.Signature
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    args: list[Any] = []
    kwargs: dict[str, Any] = {}
    for parameter in signature.parameters.values():
        value = frame.f_locals[parameter.name]
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            args.append(value)
        elif parameter.kind == inspect.Parameter.VAR_POSITIONAL:
            args.extend(value)
        elif parameter.kind == inspect.Parameter.KEYWORD_ONLY:
            kwargs[parameter.name] = value
        elif parameter.kind == inspect.Parameter.VAR_KEYWORD:
            kwargs.update(value)
    return tuple(args), kwargs


def _call_once_and_capture_scoped_calls(
    function: F,
    scope_target: Callable[..., Any],
    target_code: Any,
    target_name: str,
    output_names: tuple[str, ...],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    reset_rng: bool,
) -> tuple[Any, list[_ScopedCall]]:
    signature = inspect.signature(scope_target)
    calls: list[_ScopedCall] = []
    active_calls: dict[int, _ScopedCall] = {}
    previous_profile = sys.getprofile()

    def profile(frame, event, arg):
        if previous_profile is not None:
            previous_profile(frame, event, arg)
        if frame.f_code is not target_code:
            return
        if event == "call":
            call_args, call_kwargs = _arguments_from_frame(frame, signature)
            call = _ScopedCall(
                args=call_args,
                kwargs=call_kwargs,
                rng_state=_capture_rng_state() if reset_rng else None,
            )
            calls.append(call)
            active_calls[id(frame)] = call
        elif event == "return":
            call = active_calls.pop(id(frame))
            call.available_names = tuple(sorted(frame.f_locals))
            call.missing_names = tuple(
                name for name in output_names if name not in frame.f_locals
            )
            if not call.missing_names:
                call.captured = {name: frame.f_locals[name] for name in output_names}

    try:
        sys.setprofile(profile)
        result = function(*args, **kwargs)
    finally:
        sys.setprofile(previous_profile)

    if not calls:
        raise AssertionError(
            f"repeat_check did not observe a call to scope {target_name}"
        )
    for call_index, call in enumerate(calls):
        if call.missing_names:
            raise AssertionError(
                "repeat_check could not capture local variable(s) "
                f"{list(call.missing_names)!r} from {target_name} call[{call_index}]; "
                f"available locals: {list(call.available_names)!r}"
            )
        if call.captured is None:
            raise AssertionError(
                f"repeat_check did not observe a return from {target_name} "
                f"call[{call_index}]"
            )
    return result, calls


def _snapshot_captured(
    captured: Mapping[str, Any],
    *,
    path_prefix: str,
    iteration: int,
    repeat: int,
    finite: bool,
    synchronize: Synchronize,
) -> dict[str, Any]:
    _synchronize_outputs(captured, synchronize)
    snapshots: dict[str, Any] = {}
    for name, value in captured.items():
        path = f"{path_prefix}{name}"
        if finite:
            _check_finite(value, path=path, iteration=iteration, repeat=repeat)
        snapshots[name] = _snapshot(value)
    return snapshots


def repeat_check(
    *,
    outputs: str | Sequence[str],
    repeat: int = 2,
    compare: CompareMode = "bitwise",
    finite: bool = True,
    reset_rng: bool = True,
    synchronize: Synchronize = True,
    scope: Callable[..., Any] | None = None,
) -> Callable[[F], F]:
    """Repeat a test and compare local variables captured at function return.

    ``outputs`` contains local variable names to compare. Without ``scope``, the
    whole test body is repeated and variables are read from the test's return
    frame. With ``scope``, the test body runs once; every call to that helper is
    recorded and only those calls are replayed for subsequent repetitions.
    This keeps setup and reference calculations single-shot while repeating the
    operation under test.

    Python, NumPy, Torch CPU, and available accelerator RNG states are restored
    before every repetition. Tensor outputs are detached and cloned before
    comparison so reused output buffers are safe.
    ``compare="bitwise"`` follows the suite's existing exact-comparison
    convention and uses ``torch.equal``.

    Pytest fixtures and parametrization remain compatible because the wrapper
    preserves the original function signature. Mutable fixtures, globals, and
    explicit ``torch.Generator`` objects are not reset automatically.
    """
    output_names = _normalize_output_names(outputs)
    if isinstance(repeat, bool) or not isinstance(repeat, int):
        raise TypeError(f"repeat must be an integer, got {type(repeat).__name__}")
    if repeat < 2:
        raise ValueError(f"repeat must be at least 2, got {repeat}")
    if compare not in ("bitwise", "equal"):
        raise ValueError(f"compare must be 'bitwise' or 'equal', got {compare!r}")
    if not isinstance(finite, bool):
        raise TypeError(f"finite must be bool, got {type(finite).__name__}")
    if not isinstance(reset_rng, bool):
        raise TypeError(f"reset_rng must be bool, got {type(reset_rng).__name__}")
    if not isinstance(synchronize, bool) and not callable(synchronize):
        raise TypeError("synchronize must be bool or a zero-argument callable")
    if scope is not None and not callable(scope):
        raise TypeError(f"scope must be callable or None, got {type(scope).__name__}")

    def decorator(function: F) -> F:
        scope_callable = scope
        target = inspect.unwrap(
            scope_callable if scope_callable is not None else function
        )
        target_code = getattr(target, "__code__", None)
        if target_code is None:
            raise TypeError(
                "repeat_check scope must resolve to a Python function with a code object"
            )
        target_name = target.__qualname__

        @functools.wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if scope_callable is None:
                initial_rng = _capture_rng_state() if reset_rng else None
                post_first_rng: _RngState | None = None
                baseline: dict[str, Any] | None = None
                last_result: Any = None
                try:
                    for iteration in range(repeat):
                        if iteration > 0 and initial_rng is not None:
                            _restore_rng_state(initial_rng)

                        result, captured = _call_and_capture(
                            function,
                            target_code,
                            target_name,
                            output_names,
                            args,
                            kwargs,
                        )
                        if iteration == 0 and initial_rng is not None:
                            post_first_rng = _capture_rng_state()

                        current = _snapshot_captured(
                            captured,
                            path_prefix="",
                            iteration=iteration,
                            repeat=repeat,
                            finite=finite,
                            synchronize=synchronize,
                        )
                        if baseline is None:
                            baseline = current
                        else:
                            for name in output_names:
                                _compare_snapshots(
                                    baseline[name],
                                    current[name],
                                    path=name,
                                    iteration=iteration,
                                    repeat=repeat,
                                    compare=compare,
                                )
                        last_result = result
                finally:
                    if post_first_rng is not None:
                        _restore_rng_state(post_first_rng)
                return last_result

            result, scoped_calls = _call_once_and_capture_scoped_calls(
                function,
                target,
                target_code,
                target_name,
                output_names,
                args,
                kwargs,
                reset_rng=reset_rng,
            )
            post_test_rng = _capture_rng_state() if reset_rng else None
            call_count = len(scoped_calls)
            baselines: list[dict[str, Any]] = []
            try:
                for call_index, call in enumerate(scoped_calls):
                    path_prefix = f"call[{call_index}]." if call_count > 1 else ""
                    baselines.append(
                        _snapshot_captured(
                            cast(dict[str, Any], call.captured),
                            path_prefix=path_prefix,
                            iteration=0,
                            repeat=repeat,
                            finite=finite,
                            synchronize=synchronize,
                        )
                    )

                for iteration in range(1, repeat):
                    for call_index, call in enumerate(scoped_calls):
                        if call.rng_state is not None:
                            _restore_rng_state(call.rng_state)
                        _, captured = _call_and_capture(
                            scope_callable,
                            target_code,
                            target_name,
                            output_names,
                            call.args,
                            call.kwargs,
                        )
                        path_prefix = f"call[{call_index}]." if call_count > 1 else ""
                        current = _snapshot_captured(
                            captured,
                            path_prefix=path_prefix,
                            iteration=iteration,
                            repeat=repeat,
                            finite=finite,
                            synchronize=synchronize,
                        )
                        for name in output_names:
                            _compare_snapshots(
                                baselines[call_index][name],
                                current[name],
                                path=f"{path_prefix}{name}",
                                iteration=iteration,
                                repeat=repeat,
                                compare=compare,
                            )
            finally:
                if post_test_rng is not None:
                    _restore_rng_state(post_test_rng)
            return result

        return cast(F, wrapper)

    return decorator
