"""API logging utilities for MATE.

This module provides the @mate_api decorator for logging API calls,
including input/output tensors and metadata for debugging purposes.

Level 10 adds tensor dumping functionality for reproducibility.
"""

import enum
import fnmatch
import functools
import inspect
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Tuple, Optional
import importlib
import threading
import torch

# Read environment variables once at module load time
_API_LOG_LEVEL = int(os.environ.get("MATE_LOGLEVEL", "0"))
_API_LOG_DEST = os.environ.get("MATE_LOGDEST", "stdout")
_API_LOG_DEST = _API_LOG_DEST.replace("%i", str(os.getpid()))

# Configuration for Level 10 tensor dumping
_DUMP_DIR = os.environ.get("MATE_DUMP_DIR", "mate_dumps")
_DUMP_MAX_SIZE_GB = float(os.environ.get("MATE_DUMP_MAX_SIZE_GB", "20"))
_DUMP_MAX_COUNT = int(os.environ.get("MATE_DUMP_MAX_COUNT", "1000"))

# Dump filtering: include/exclude patterns (fnmatch-style, comma-separated)
_DUMP_INCLUDE = os.environ.get("MATE_DUMP_INCLUDE", "")
_DUMP_EXCLUDE = os.environ.get("MATE_DUMP_EXCLUDE", "")
_DUMP_INCLUDE_PATTERNS = [p.strip() for p in _DUMP_INCLUDE.split(",") if p.strip()]
_DUMP_EXCLUDE_PATTERNS = [p.strip() for p in _DUMP_EXCLUDE.split(",") if p.strip()]

# SafeTensors format option (default: use torch.save which preserves stride/contiguity)
_DUMP_SAFETENSORS = os.environ.get("MATE_DUMP_SAFETENSORS", "0") == "1"

# Global tracking for dump limits (reset per process)
_dump_count = 0
_dump_total_size_bytes = 0
_dump_call_counter: Dict[str, int] = {}
_dump_lock = threading.Lock()

# Create logger using Python's logging library
_logger = logging.getLogger("mate.api")


def _setup_logger():
    """Set up the logger based on environment variables."""
    if _API_LOG_LEVEL == 0:
        _logger.addHandler(logging.NullHandler())
        _logger.setLevel(logging.CRITICAL + 1)
        return

    _logger.setLevel(logging.DEBUG)
    _logger.handlers.clear()

    if _API_LOG_DEST == "stdout":
        handler = logging.StreamHandler(sys.stdout)
    elif _API_LOG_DEST == "stderr":
        handler = logging.StreamHandler(sys.stderr)
    else:
        handler = logging.FileHandler(_API_LOG_DEST, mode="a")

    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    _logger.addHandler(handler)
    _logger.propagate = False


_setup_logger()


def get_api_logger() -> logging.Logger:
    """Return the configured MATE API logger."""
    return _logger


def _get_timestamp() -> str:
    """Get current timestamp in the format [YYYY-MM-DD HH:MM:SS]."""
    return datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")


def _warn_dump():
    """Warn users about security implications of Level 10 logging."""
    if _API_LOG_LEVEL >= 10:
        lines = ["=" * 80]
        lines.append(
            "WARNING: MATE API Logging is set to Level 10 (Tensor Dumping).\n"
            "This will dump ALL input and outputs including tensors for MATE APIs to disk in\n"
            "the configured dump directory. Ensure that you are NOT processing sensitive data\n"
            "or that the dump directory is secure. To disable dumping, unset MATE_LOGLEVEL or\n"
            "set it to below 10."
        )
        lines.append(f"Current dump directory is: {_DUMP_DIR}")
        if _DUMP_SAFETENSORS:
            lines.append(
                "SAFETENSORS mode enabled: tensor stride/non-contiguity will NOT be preserved.\n"
                "    Tensors will be saved as contiguous. Use torch.save (default) to preserve strides."
            )
        if _DUMP_INCLUDE_PATTERNS:
            lines.append(f"Include filter: {_DUMP_INCLUDE_PATTERNS}")
        if _DUMP_EXCLUDE_PATTERNS:
            lines.append(f"Exclude filter: {_DUMP_EXCLUDE_PATTERNS}")
        lines.append("=" * 80)
        _logger.warning("\n".join(lines))


def _should_dump_function(func_name: str) -> bool:
    """Check if a function should be dumped based on include/exclude filters."""
    if _DUMP_INCLUDE_PATTERNS:
        if not any(fnmatch.fnmatch(func_name, pat) for pat in _DUMP_INCLUDE_PATTERNS):
            return False
    if _DUMP_EXCLUDE_PATTERNS:
        if any(fnmatch.fnmatch(func_name, pat) for pat in _DUMP_EXCLUDE_PATTERNS):
            return False
    return True


def _append_to_jsonl(filepath: Path, record: Dict[str, Any]) -> None:
    """Append a JSON record as a single line to a JSONL file."""
    with open(filepath, "a") as f:
        f.write(json.dumps(record) + "\n")


def _read_jsonl_last_record(filepath: Path) -> Optional[Dict[str, Any]]:
    """Read the last record from a JSONL file."""
    if not filepath.exists():
        return None
    last_line = None
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                last_line = line
    if last_line:
        return json.loads(last_line)
    return None


def _get_tensor_size_bytes(tensor: torch.Tensor) -> int:
    """Calculate the size of a tensor in bytes."""
    return tensor.element_size() * tensor.nelement()


def _get_tensor_va_range(tensor: torch.Tensor) -> Tuple[int, int]:
    """Calculate the half-open virtual address range used by a tensor view."""
    data_ptr = int(tensor.data_ptr())
    if tensor.numel() == 0:
        return data_ptr, data_ptr

    min_offset = 0
    max_offset = 0
    for size, stride in zip(tensor.shape, tensor.stride()):
        if size <= 1:
            continue
        extent = (int(size) - 1) * int(stride)
        if extent < 0:
            min_offset += extent
        else:
            max_offset += extent

    element_size = int(tensor.element_size())
    va_start = data_ptr + min_offset * element_size
    va_end = data_ptr + max_offset * element_size + element_size
    return va_start, va_end


def _serialize_value(value: Any) -> Any:
    """Convert a non-tensor value to a JSON-serializable format."""
    try:
        if isinstance(value, torch.dtype):
            return {"type": "torch.dtype", "value": str(value)}
        elif isinstance(value, enum.Enum):
            return {
                "type": "enum",
                "name": f"{type(value).__name__}.{value.name}",
                "value": value.value,
            }
        elif isinstance(value, (int, float, str, bool, type(None))):
            return value
        elif isinstance(value, (list, tuple, dict)):
            return {"type": type(value).__name__, "value": str(value)[:1000]}
        else:
            return {"type": type(value).__name__, "repr": str(value)[:1000]}
    except Exception:
        return {"type": type(value).__name__, "repr": "<not serializable>"}


def _extract_tensors_and_metadata(
    args: tuple, kwargs: dict
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Extract tensors and non-tensor metadata from function arguments."""
    tensors = {}
    metadata = {}

    for i, arg in enumerate(args):
        key = f"arg_{i}"
        if isinstance(arg, torch.Tensor):
            tensors[key] = arg.cpu()
        else:
            metadata[key] = _serialize_value(arg)

    for key, value in kwargs.items():
        kwarg_key = f"kwarg_{key}"
        if isinstance(value, torch.Tensor):
            tensors[kwarg_key] = value.cpu()
        else:
            metadata[kwarg_key] = _serialize_value(value)

    return tensors, metadata


def _dump_function_inputs(
    func: Callable,
    func_name: str,
    args: tuple,
    kwargs: dict,
    self_id: Optional[int] = None,
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Dump function inputs to disk BEFORE execution (crash-safe).

    Returns:
        Tuple of (dump_dir, metadata) on success, None if skipped or failed.
        The returned metadata dict is passed directly to _dump_function_outputs
        to avoid re-reading from disk.
    """
    global _dump_count, _dump_total_size_bytes

    if not _should_dump_function(func_name):
        _logger.debug(f"Skipping dump for {func_name} (filtered)")
        return None

    # Claim a slot under the lock before doing any I/O
    with _dump_lock:
        if _dump_count >= _DUMP_MAX_COUNT:
            _logger.warning(
                f"Dump limit reached ({_DUMP_MAX_COUNT}). Skipping {func_name}."
            )
            return None
        _dump_call_counter[func_name] = _dump_call_counter.get(func_name, 0) + 1
        call_seq = _dump_call_counter[func_name]

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        pid = os.getpid()
        dump_name = f"{timestamp}_pid{pid}_{func_name}_call{call_seq:04d}"
        dump_dir = Path(_DUMP_DIR) / dump_name
        dump_dir.mkdir(parents=True, exist_ok=True)

        input_tensors, input_metadata = _extract_tensors_and_metadata(args, kwargs)
        input_size = sum(_get_tensor_size_bytes(t) for t in input_tensors.values())

        max_size_bytes = _DUMP_MAX_SIZE_GB * 1024 * 1024 * 1024
        with _dump_lock:
            if _dump_total_size_bytes + input_size > max_size_bytes:
                _logger.warning(f"Dump size limit reached ({_DUMP_MAX_SIZE_GB} GB).")
                dump_dir.rmdir()
                return None
            _dump_count += 1
            _dump_total_size_bytes += input_size

        if input_tensors:
            if _DUMP_SAFETENSORS:
                try:
                    from safetensors.torch import save_file

                    tensors_contiguous = {
                        k: v.contiguous() for k, v in input_tensors.items()
                    }
                    save_file(tensors_contiguous, str(dump_dir / "inputs.safetensors"))
                except ImportError:
                    _logger.error("safetensors not installed. pip install safetensors")
                    raise
            else:
                torch.save(input_tensors, dump_dir / "inputs.pt")

        metadata: Dict[str, Any] = {
            "function_name": func_name,
            "module": func.__module__ if hasattr(func, "__module__") else "",
            "call_sequence": call_seq,
            "timestamp": timestamp,
            "process_id": pid,
            "input_metadata": input_metadata,
            "output_metadata": {},
            "tensor_info": {
                "input_tensor_keys": list(input_tensors.keys()),
                "output_tensor_keys": [],
                "input_size_bytes": input_size,
                "input_size_mb": input_size / (1024 * 1024),
            },
            "tensor_details": {},
            "tensor_format": "safetensors" if _DUMP_SAFETENSORS else "torch",
            "function_signature": str(inspect.signature(func)),
            "versions": {
                "torch": torch.__version__,
                "python": sys.version,
            },
            "execution_status": "inputs_saved",
        }

        if self_id is not None:
            metadata["self_id"] = self_id

        for key, tensor in input_tensors.items():
            metadata["tensor_details"][key] = {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "stride": list(tensor.stride()),
                "device": str(tensor.device),
            }

        try:
            from ._build_meta import __version__ as mate_version

            metadata["versions"]["mate"] = mate_version
        except Exception:
            metadata["versions"]["mate"] = "<unavailable>"

        metadata["dump_dir"] = str(dump_dir)

        _append_to_jsonl(dump_dir / "metadata.jsonl", metadata)
        _append_to_jsonl(Path(_DUMP_DIR) / "session.jsonl", metadata)

        _logger.debug(
            f"Dumped inputs to: {dump_dir} "
            f"(size: {input_size / (1024 * 1024):.2f} MB, "
            f"total: {_dump_count}/{_DUMP_MAX_COUNT} dumps)"
        )

        return str(dump_dir), metadata

    except Exception as e:
        _logger.error(f"Failed to dump function call {func_name}: {e}")
        import traceback

        _logger.error(traceback.format_exc())
        return None


def _dump_function_outputs(
    dump_dir: str, metadata: Dict[str, Any], result: Any
) -> None:
    """Add function outputs to an existing dump directory (crash-safe).

    Args:
        dump_dir: Path to the dump directory created by _dump_function_inputs.
        metadata: The metadata dict returned by _dump_function_inputs; passed
                  directly to avoid re-reading metadata.jsonl from disk.
        result: The return value of the decorated function.
    """
    global _dump_total_size_bytes

    try:
        dump_path = Path(dump_dir)
        if not dump_path.exists():
            _logger.error(f"Dump directory not found: {dump_dir}")
            return

        output_tensors = {}
        output_metadata = {}
        if isinstance(result, torch.Tensor):
            output_tensors["result"] = result.cpu()
        elif isinstance(result, tuple):
            for i, item in enumerate(result):
                if isinstance(item, torch.Tensor):
                    output_tensors[f"result_{i}"] = item.cpu()
                else:
                    output_metadata[f"result_{i}"] = _serialize_value(item)
        else:
            output_metadata["result"] = _serialize_value(result)

        output_size = sum(_get_tensor_size_bytes(t) for t in output_tensors.values())

        # Check size limit before writing output tensors (problem 10)
        max_size_bytes = _DUMP_MAX_SIZE_GB * 1024 * 1024 * 1024
        with _dump_lock:
            if _dump_total_size_bytes + output_size > max_size_bytes:
                _logger.warning(
                    f"Dump size limit reached ({_DUMP_MAX_SIZE_GB} GB). "
                    f"Skipping output dump for {dump_dir}."
                )
                return
            _dump_total_size_bytes += output_size

        if output_tensors:
            if _DUMP_SAFETENSORS:
                from safetensors.torch import save_file

                tensors_contiguous = {
                    k: v.contiguous() for k, v in output_tensors.items()
                }
                save_file(tensors_contiguous, str(dump_path / "outputs.safetensors"))
            else:
                torch.save(output_tensors, dump_path / "outputs.pt")

        metadata["output_metadata"] = output_metadata
        metadata["tensor_info"]["output_tensor_keys"] = list(output_tensors.keys())
        metadata["tensor_info"]["output_size_bytes"] = output_size
        metadata["tensor_info"]["output_size_mb"] = output_size / (1024 * 1024)
        metadata["tensor_info"]["total_size_bytes"] = (
            metadata["tensor_info"]["input_size_bytes"] + output_size
        )
        metadata["tensor_info"]["total_size_mb"] = metadata["tensor_info"][
            "total_size_bytes"
        ] / (1024 * 1024)
        metadata["execution_status"] = "completed"

        if "tensor_details" not in metadata:
            metadata["tensor_details"] = {}
        for key, tensor in output_tensors.items():
            metadata["tensor_details"][key] = {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "stride": list(tensor.stride()),
                "device": str(tensor.device),
            }

        metadata_jsonl_path = dump_path / "metadata.jsonl"
        _append_to_jsonl(metadata_jsonl_path, metadata)
        _append_to_jsonl(Path(_DUMP_DIR) / "session.jsonl", metadata)

        _logger.debug(
            f"Dumped outputs to: {dump_dir} "
            f"(output size: {output_size / (1024 * 1024):.2f} MB)"
        )

    except Exception as e:
        _logger.error(f"Failed to dump outputs to {dump_dir}: {e}")
        import traceback

        _logger.error(traceback.format_exc())


def _reconstruct_value(value: Any) -> Any:
    """Reconstruct special types from metadata format."""
    if isinstance(value, dict):
        value_type = value.get("type")
        if value_type == "torch.dtype":
            dtype_str = value.get("value", "")
            dtype_name = dtype_str.replace("torch.", "")
            try:
                return getattr(torch, dtype_name)
            except AttributeError:
                _logger.warning(f"Could not reconstruct dtype: {dtype_str}")
                return value
        return value
    return value


def _resolve_function(module_name: str, function_name: str) -> Optional[Callable]:
    """Resolve a function from module name and function name."""
    try:
        module = importlib.import_module(module_name)
        parts = function_name.split(".")
        obj: Any = module
        for part in parts:
            obj = getattr(obj, part)
        if not callable(obj):
            return None
        return obj
    except Exception as e:
        _logger.warning(
            f"Could not resolve function {module_name}.{function_name}: {e}"
        )
        return None


def replay_from_dump(
    dump_dir: str,
    compare_outputs: bool = False,
    device: str = "musa",
    run: bool = False,
    object_registry: Optional[Dict[Tuple[int, int], Any]] = None,
) -> Dict[str, Any]:
    """Replay a function call from a dumped directory.

    Args:
        dump_dir: Path to the dump directory
        compare_outputs: If True, load and compare with saved outputs
        device: Target device for tensors ("musa", "cpu", "musa:N")
        run: If True, try to resolve and execute the function
        object_registry: Registry of stateful objects mapped by (process_id, self_id)

    Returns:
        Dictionary containing args, kwargs, metadata, and optional execution results
    """
    dump_path = Path(dump_dir)
    if not dump_path.exists():
        raise FileNotFoundError(f"Dump directory not found: {dump_dir}")

    metadata_jsonl_path = dump_path / "metadata.jsonl"
    if not metadata_jsonl_path.exists():
        raise FileNotFoundError(f"metadata.jsonl not found in {dump_dir}")

    metadata = _read_jsonl_last_record(metadata_jsonl_path)
    if metadata is None:
        raise ValueError(f"metadata.jsonl is empty in {dump_dir}")

    func_name = metadata["function_name"]

    # Load input tensors
    inputs_pt_path = dump_path / "inputs.pt"
    inputs_safetensors_path = dump_path / "inputs.safetensors"

    if inputs_pt_path.exists():
        # weights_only=False is required because dumps contain dict metadata, not just tensors
        input_tensors = torch.load(
            str(inputs_pt_path), map_location="cpu", weights_only=False
        )
    elif inputs_safetensors_path.exists():
        try:
            from safetensors.torch import load_file

            input_tensors = load_file(str(inputs_safetensors_path), device="cpu")
        except ImportError:
            raise ImportError(
                "safetensors not installed. pip install safetensors"
            ) from None
    else:
        raise FileNotFoundError(
            f"Neither inputs.pt nor inputs.safetensors found in {dump_dir}"
        )

    for key, tensor in input_tensors.items():
        input_tensors[key] = tensor.to(device)

    # Reconstruct args and kwargs
    args = []
    kwargs = {}
    input_metadata = metadata.get("input_metadata", {})

    max_arg_idx = -1
    for key in input_tensors.keys():
        if key.startswith("arg_"):
            idx = int(key.split("_")[1])
            max_arg_idx = max(max_arg_idx, idx)
    for key in input_metadata.keys():
        if key.startswith("arg_"):
            idx = int(key.split("_")[1])
            max_arg_idx = max(max_arg_idx, idx)

    for i in range(max_arg_idx + 1):
        key = f"arg_{i}"
        if key in input_tensors:
            args.append(input_tensors[key])
        elif key in input_metadata:
            args.append(_reconstruct_value(input_metadata[key]))
        else:
            _logger.warning(f"Missing argument {i} in dump.")
            args.append(None)

    for key in input_tensors.keys():
        if key.startswith("kwarg_"):
            kwarg_name = key[len("kwarg_") :]
            kwargs[kwarg_name] = input_tensors[key]

    for key in input_metadata.keys():
        if key.startswith("kwarg_"):
            kwarg_name = key[len("kwarg_") :]
            if kwarg_name not in kwargs:
                kwargs[kwarg_name] = _reconstruct_value(input_metadata[key])

    _logger.info(f"Replaying {func_name} from {dump_dir}")
    _logger.info(f"  Args: {len(args)}, Kwargs: {list(kwargs.keys())}")

    result_dict: Dict[str, Any] = {"args": args, "kwargs": kwargs, "metadata": metadata}

    # Load expected outputs if needed
    expected_outputs = {}
    output_metadata = {}
    _outputs_file_missing = False
    if compare_outputs:
        outputs_pt_path = dump_path / "outputs.pt"
        outputs_safetensors_path = dump_path / "outputs.safetensors"

        if outputs_pt_path.exists():
            # weights_only=False is required because dumps contain dict metadata, not just tensors
            expected_outputs = torch.load(
                str(outputs_pt_path), map_location="cpu", weights_only=False
            )
        elif outputs_safetensors_path.exists():
            try:
                from safetensors.torch import load_file

                expected_outputs = load_file(
                    str(outputs_safetensors_path), device="cpu"
                )
            except ImportError:
                raise ImportError("safetensors not installed") from None
        else:
            _logger.warning(
                f"compare_outputs=True but no output file found in {dump_dir}. "
                "The dump may be incomplete (e.g. the process crashed after saving inputs). "
                "Comparison result will be False."
            )
            _outputs_file_missing = True

        for key, tensor in expected_outputs.items():
            expected_outputs[key] = tensor.to(device)

        output_metadata = metadata.get("output_metadata", {})
        result_dict["expected_tensors"] = expected_outputs
        result_dict["expected_metadata"] = output_metadata

    if run:
        module_name = metadata.get("module")
        self_id = metadata.get("self_id")
        process_id = metadata.get("process_id")

        func = None
        obj = None

        if self_id is not None:
            registry_key = (process_id, self_id)
            if func_name.endswith(".__init__"):
                class_name = func_name.split(".")[-2]
                cls_obj = _resolve_function(module_name, class_name)
                if cls_obj and callable(cls_obj):
                    real_args = args[1:] if len(args) > 0 else []
                    try:
                        _logger.info(f"Instantiating {class_name}...")
                        obj = cls_obj(*real_args, **kwargs)
                        if object_registry is not None:
                            object_registry[registry_key] = obj
                        execution_result = None
                        result_dict["execution_result"] = execution_result
                        if compare_outputs:
                            result_dict["comparison_match"] = True
                        return result_dict
                    except Exception as e:
                        _logger.error(f"Failed to instantiate {class_name}: {e}")
                        result_dict["execution_error"] = str(e)
                        return result_dict
            else:
                if object_registry is not None and registry_key in object_registry:
                    obj = object_registry[registry_key]
                    method_name = func_name.split(".")[-1]
                    if hasattr(obj, method_name):
                        func = getattr(obj, method_name)
                        args = args[1:] if len(args) > 0 else []
                    else:
                        _logger.warning(f"Object {obj} has no method {method_name}")
                else:
                    _logger.warning(
                        f"Object (PID: {process_id}, ID: {self_id}) not found."
                    )

        if func is None:
            func = _resolve_function(module_name, func_name)

        if func:
            try:
                _logger.info(f"Executing {module_name}.{func_name}...")
                execution_result = func(*args, **kwargs)
                result_dict["execution_result"] = execution_result

                if compare_outputs:
                    actual_outputs = {}
                    if isinstance(execution_result, torch.Tensor):
                        actual_outputs["result"] = execution_result
                    elif isinstance(execution_result, (tuple, list)):
                        for i, item in enumerate(execution_result):
                            if isinstance(item, torch.Tensor):
                                actual_outputs[f"result_{i}"] = item

                    match = not _outputs_file_missing
                    if match and expected_outputs:
                        for key in expected_outputs:
                            if key in actual_outputs:
                                if not torch.allclose(
                                    expected_outputs[key],
                                    actual_outputs[key],
                                    rtol=1e-3,
                                    atol=1e-3,
                                ):
                                    match = False
                                    break
                            else:
                                match = False
                                break

                    result_dict["comparison_match"] = match
                    if match:
                        _logger.info("Replay comparison passed!")
                    else:
                        _logger.warning("Replay comparison FAILED.")

            except Exception as e:
                _logger.error(f"Execution failed: {e}")
                import traceback

                _logger.error(traceback.format_exc())
                result_dict["execution_error"] = str(e)
        else:
            _logger.warning(
                f"Skipping execution: could not resolve {module_name}.{func_name}"
            )

    return result_dict


def replay_sequence(root_dir: str, device: str = "musa") -> list:
    """Replay a sequence of API calls from a root dump directory.

    Args:
        root_dir: Path to the root directory containing dump subdirectories
        device: Target device for execution

    Returns:
        List of results from replay_from_dump calls
    """
    root_path = Path(root_dir)
    if not root_path.exists():
        raise FileNotFoundError(f"Root dump directory not found: {root_dir}")

    dump_dirs = []
    for item in root_path.iterdir():
        if item.is_dir() and (item / "metadata.jsonl").exists():
            dump_dirs.append(item)

    dump_dirs.sort(key=lambda x: x.name)

    results = []
    total = len(dump_dirs)
    _logger.info(f"Found {total} dumps to replay from {root_dir}")

    object_registry: Dict[Tuple[int, int], Any] = {}

    for i, dump_dir in enumerate(dump_dirs):
        _logger.info(f"[{i + 1}/{total}] Replaying {dump_dir.name}...")
        try:
            res = replay_from_dump(
                str(dump_dir),
                compare_outputs=True,
                device=device,
                run=True,
                object_registry=object_registry,
            )
            res["dump_dir"] = str(dump_dir)
            results.append(res)
        except Exception as e:
            _logger.error(f"Failed to replay {dump_dir.name}: {e}")
            results.append({"error": str(e), "dump_dir": str(dump_dir)})

    return results


def _format_value(value: Any, level: int, indent: int = 0) -> str:
    """Format a value for logging based on the log level."""
    indent_str = "  " * indent

    if value is None:
        return f"{indent_str}None"

    if isinstance(value, enum.Enum):
        return (
            f"{indent_str}{value.__class__.__name__}.{value.name} (value={value.value})"
        )

    if isinstance(value, torch.Tensor):
        if level == 1:
            return f"{indent_str}Tensor(...)"

        lines = [f"{indent_str}Tensor("]
        lines.append(f"{indent_str}  shape={tuple(value.shape)}")
        lines.append(f"{indent_str}  stride={tuple(value.stride())}")
        lines.append(f"{indent_str}  dtype={value.dtype}")
        lines.append(f"{indent_str}  device={value.device}")
        try:
            va_start, va_end = _get_tensor_va_range(value)
            lines.append(f"{indent_str}  va_range=[0x{va_start:x}, 0x{va_end:x})")
        except Exception as e:
            lines.append(f"{indent_str}  va_range=<unavailable: {e}>")
        lines.append(f"{indent_str}  requires_grad={value.requires_grad}")
        lines.append(f"{indent_str}  is_contiguous={value.is_contiguous()}")

        if level >= 5 and value.numel() > 0:
            try:
                is_capturing = False
                # Only check for MUSA graph capture
                device_type = str(value.device).split(":")[0]
                if device_type == "musa":
                    try:
                        import torch_musa

                        if hasattr(torch_musa, "is_current_stream_capturing"):
                            is_capturing = torch_musa.is_current_stream_capturing()
                    except Exception:
                        pass

                # Skip statistics during MUSA graph capture (avoid sync issues)
                if is_capturing:
                    lines.append(
                        f"{indent_str}  [statistics skipped: MUSA graph capture in progress]"
                    )
                elif value.dtype in [
                    torch.float16,
                    torch.float32,
                    torch.float64,
                    torch.bfloat16,
                    torch.float8_e4m3fn,
                    torch.float8_e5m2,
                ]:
                    val_float = value.float()
                    lines.append(f"{indent_str}  min={val_float.min().item():.6f}")
                    lines.append(f"{indent_str}  max={val_float.max().item():.6f}")
                    lines.append(f"{indent_str}  mean={val_float.mean().item():.6f}")
                    nan_count = torch.isnan(val_float).sum().item()
                    lines.append(f"{indent_str}  nan_count={nan_count}")
                    inf_count = torch.isinf(val_float).sum().item()
                    lines.append(f"{indent_str}  inf_count={inf_count}")
                elif value.dtype in [
                    torch.int8,
                    torch.int16,
                    torch.int32,
                    torch.int64,
                    torch.uint8,
                ]:
                    lines.append(f"{indent_str}  min={value.min().item()}")
                    lines.append(f"{indent_str}  max={value.max().item()}")
                    lines.append(
                        f"{indent_str}  mean={value.float().mean().item():.6f}"
                    )
            except Exception as e:
                lines.append(f"{indent_str}  [statistics error: {e}]")

        lines.append(f"{indent_str})")
        return "\n".join(lines)

    if isinstance(value, list):
        if len(value) == 0:
            return f"{indent_str}[]"
        if level == 1:
            return f"{indent_str}[list with {len(value)} items]"
        lines = [f"{indent_str}["]
        for i, item in enumerate(value):
            lines.append(
                f"{indent_str}  [{i}]: {_format_value(item, level, indent + 1)}"
            )
        lines.append(f"{indent_str}]")
        return "\n".join(lines)

    if isinstance(value, tuple):
        if len(value) == 0:
            return f"{indent_str}()"
        if level == 1:
            return f"{indent_str}(tuple with {len(value)} items)"
        lines = [f"{indent_str}("]
        for i, item in enumerate(value):
            lines.append(
                f"{indent_str}  [{i}]: {_format_value(item, level, indent + 1)}"
            )
        lines.append(f"{indent_str})")
        return "\n".join(lines)

    if isinstance(value, dict):
        if len(value) == 0:
            return f"{indent_str}{{}}"
        if level == 1:
            return f"{indent_str}{{dict with {len(value)} keys}}"
        lines = [f"{indent_str}{{"]
        for key, val in value.items():
            lines.append(
                f"{indent_str}  {repr(key)}: {_format_value(val, level, indent + 1)}"
            )
        lines.append(f"{indent_str}}}")
        return "\n".join(lines)

    if isinstance(value, (int, float, bool, complex)):
        return f"{indent_str}{value}"

    if isinstance(value, str):
        return f"{indent_str}{repr(value)}"

    try:
        return f"{indent_str}{repr(value)}"
    except Exception:
        return f"{indent_str}<{type(value).__name__} object>"


def _get_default_params(func: Callable, args: tuple, kwargs: dict) -> dict:
    """Extract parameters that have default values but were not explicitly provided."""
    try:
        sig = inspect.signature(func)
        default_params = {}
        # Track positional slot index separately; do NOT use the enumeration index
        # because VAR_POSITIONAL/*args and KEYWORD_ONLY params break positional counting.
        pos_idx = 0
        for param_name, param in sig.parameters.items():
            kind = param.kind
            if kind == inspect.Parameter.VAR_POSITIONAL:
                # *args consumes all remaining positional arguments
                pos_idx = len(args)
                continue
            if kind == inspect.Parameter.VAR_KEYWORD:
                # **kwargs has no reportable default
                continue
            if param.default is inspect.Parameter.empty:
                if kind in (
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.POSITIONAL_ONLY,
                ):
                    pos_idx += 1
                continue
            # Parameter has a default — determine whether the caller supplied it
            if kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.POSITIONAL_ONLY,
            ):
                provided = pos_idx < len(args) or param_name in kwargs
                pos_idx += 1
            else:
                # KEYWORD_ONLY: can only be provided via kwargs
                provided = param_name in kwargs
            if not provided:
                default_params[param_name] = param.default
        return default_params
    except Exception:
        return {}


def _log_function_inputs(
    func: Callable, func_name: str, args: tuple, kwargs: dict, level: int
) -> None:
    """Log function inputs BEFORE execution for crash safety."""
    lines = []
    lines.append("=" * 80)
    lines.append(f"{_get_timestamp()} MATE API Call: {func_name}")
    lines.append("-" * 80)

    if args or kwargs:
        if args:
            lines.append("Positional input arguments:")
            for i, arg in enumerate(args):
                lines.append(f"  arg[{i}]:")
                lines.append(_format_value(arg, level, indent=2))
        if kwargs:
            lines.append("Keyword input arguments:")
            for key, value in kwargs.items():
                lines.append(f"  {key}=")
                lines.append(_format_value(value, level, indent=2))
    else:
        lines.append("(No explicit arguments)")

    default_params = _get_default_params(func, args, kwargs)
    if default_params:
        lines.append("Default parameters (not explicitly provided):")
        for param_name, default_value in default_params.items():
            lines.append(f"  {param_name}= [DEFAULT]")
            lines.append(_format_value(default_value, level, indent=2))

    _logger.debug("\n".join(lines))


def _log_function_outputs(func_name: str, result: Any, level: int) -> None:
    """Log function outputs AFTER successful execution."""
    lines = []
    lines.append("Output value:")
    lines.append(_format_value(result, level, indent=1))
    lines.append("=" * 80)
    lines.append("")
    _logger.debug("\n".join(lines))


def _log_system_info():
    """Log system information once at module initialization."""
    if _API_LOG_LEVEL == 0:
        return

    lines = []
    lines.append("=" * 80)
    lines.append(f"{_get_timestamp()} MATE API Logging - System Information")
    lines.append("=" * 80)

    try:
        try:
            from ._build_meta import __version__ as mate_version

            lines.append(f"MATE version: {mate_version}")
        except Exception:
            lines.append("MATE version: <unavailable>")

        # torch_musa version and GPU information
        try:
            import torch_musa

            lines.append(f"torch_musa version: {torch_musa.__version__}")
            if torch_musa.is_available():
                device_count = torch_musa.device_count()
                lines.append(f"Number of MUSA GPUs: {device_count}")
                for i in range(device_count):
                    try:
                        gpu_name = torch_musa.get_device_name(i)
                        lines.append(f"  MUSA GPU {i}: {gpu_name}")
                    except Exception as e:
                        lines.append(f"  MUSA GPU {i}: <error: {e}>")
            else:
                lines.append("MUSA: Not available")
        except ImportError:
            lines.append("torch_musa: <not available>")

        lines.append(f"PyTorch version: {torch.__version__}")

    except Exception as e:
        lines.append(f"Error gathering system information: {e}")

    lines.append("=" * 80)
    lines.append("")
    _logger.debug("\n".join(lines))


_log_system_info()
_warn_dump()


def mate_api(func: Optional[Callable] = None) -> Callable:
    """Decorator for MATE APIs.

    This decorator provides API logging functionality for MATE functions.
    It integrates with Python's standard logging infrastructure while
    maintaining zero overhead when disabled (MATE_LOGLEVEL=0).

    Environment Variables:
        MATE_LOGLEVEL (int, default: 0):
            - 0: No logging (zero overhead - decorator returns original function)
            - 1: Log function name only (logged BEFORE execution - crash-safe)
            - 3: Log function name + inputs/outputs with metadata,
                 including tensor shape/stride/dtype/device/VA range
            - 5: Level 3 logging + tensor statistics
            - 10: Level 5 logging + dump metadata and input/output tensors to disk
                  for reproducibility (preserves stride/contiguity)

    Notes:
        - At level 5, tensor statistics (min/max/mean/nan_count/inf_count) are
          automatically skipped during MUSA graph capture to avoid synchronization
          issues. The message "[statistics skipped: graph capture in progress]"
          will be logged in this case.

        MATE_LOGDEST (str, default: "stdout"):
            - "stdout": Log to standard output
            - "stderr": Log to standard error
            - <path>: Log to specified file path (supports %i for PID)

    Level 10 Tensor Dumping (additional variables):
        MATE_DUMP_DIR (str, default: "mate_dumps"):
            Directory where tensor dumps are saved
        MATE_DUMP_MAX_SIZE_GB (float, default: 20):
            Maximum total size of dumps in GB
        MATE_DUMP_MAX_COUNT (int, default: 1000):
            Maximum number of function call dumps
        MATE_DUMP_SAFETENSORS (int, default: 0):
            - 0: Use torch.save format (preserves stride/contiguity)
            - 1: Use safetensors format (no pickle, but loses stride info)
        MATE_DUMP_INCLUDE (str, default: ""):
            Comma-separated list of patterns to include for dumping (fnmatch-style)
        MATE_DUMP_EXCLUDE (str, default: ""):
            Comma-separated list of patterns to exclude for dumping (fnmatch-style)

    Replay Functions:
        replay_from_dump(dump_dir, compare_outputs=False, device="musa", run=False):
            Replay a single function call from a dump directory
        replay_sequence(root_dir, device="musa"):
            Replay a sequence of API calls from a root dump directory

    Examples:
        Basic usage::

            @mate_api
            def my_function(x, y):
                return x + y

        Level 10 with filtering::

            # Only dump functions matching pattern
            MATE_LOGLEVEL=10 MATE_DUMP_INCLUDE="*gemm*,*attention*" python script.py
    """
    if _API_LOG_LEVEL == 0:
        if func is None:
            return lambda f: f
        return func

    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            func_name = f.__name__
            self_id = None
            if args and hasattr(args[0], "__class__"):
                try:
                    class_name = args[0].__class__.__name__
                    if hasattr(args[0], f.__name__):
                        func_name = f"{class_name}.{func_name}"
                        self_id = id(args[0])
                except Exception:
                    pass

            # Level 10: Dump inputs BEFORE execution (crash-safe)
            dump_dir = None
            dump_metadata = None
            if _API_LOG_LEVEL >= 10:
                try:
                    dump_result = _dump_function_inputs(
                        f, func_name, args, kwargs, self_id=self_id
                    )
                    if dump_result:
                        dump_dir, dump_metadata = dump_result
                        _logger.debug(f"Inputs dumped to: {dump_dir}")
                except Exception as e:
                    _logger.error(f"[DUMP ERROR (inputs) in {func_name}]: {e}")

            # Log BEFORE execution (crash-safe for all levels!)
            try:
                if _API_LOG_LEVEL == 1:
                    _logger.debug(f"{_get_timestamp()} MATE API Call: {func_name}")
                elif _API_LOG_LEVEL >= 3:
                    effective_level = min(_API_LOG_LEVEL, 5)
                    _log_function_inputs(f, func_name, args, kwargs, effective_level)
            except Exception as e:
                _logger.error(f"[LOGGING ERROR in {func_name} (pre-execution)]: {e}")

            # Call the original function
            result = f(*args, **kwargs)

            # Log outputs AFTER successful execution (level 3+ only)
            try:
                if _API_LOG_LEVEL >= 3:
                    effective_level = min(_API_LOG_LEVEL, 5)
                    _log_function_outputs(func_name, result, effective_level)
            except Exception as e:
                _logger.error(f"[LOGGING ERROR in {func_name} (outputs)]: {e}")

            # Level 10: Dump outputs AFTER successful execution
            if _API_LOG_LEVEL >= 10 and dump_dir and dump_metadata is not None:
                try:
                    _dump_function_outputs(dump_dir, dump_metadata, result)
                    _logger.info(f"Outputs dumped to: {dump_dir}")
                except Exception as e:
                    _logger.error(f"[DUMP ERROR (outputs) in {func_name}]: {e}")

            return result

        return wrapper

    if func is None:
        return decorator
    return decorator(func)
