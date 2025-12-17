####
# Profiling Related Functions
# TODO: @kesavan @simon @arya
####

import torch
import pandas as pd

# wrapper with tool to measure hardware metric


# Check if nsight-python is available
# To patch nsight-python to support multiple metrics and fix the "cannot insert Annotation" bug
try:
    import nsight
    NSIGHT_AVAILABLE = True
    
    # Patch 1: Multiple metrics support
    _orig_ncu_init = nsight.collection.ncu.NCUCollector.__init__
    nsight.collection.ncu.NCUCollector.__init__ = lambda self, metric="gpu__time_duration.sum", *a, **kw: \
        _orig_ncu_init(self, ",".join(metric) if isinstance(metric, (list, tuple)) else metric, *a, **kw)
    
    # Patch 2: Extract all metrics from comma-separated string
    _orig_extract = nsight.extraction.extract_df_from_report
    def _patched_extract(path, metric, *a, **kw):
        if "," not in metric:
            return _orig_extract(path, metric, *a, **kw)
        rows = []
        for m in metric.split(","):
            df = _orig_extract(path, m.strip(), *a, **kw)
            if df is not None and not df.empty:
                df = df.copy()
                df['Metric Name'] = m.strip()
                rows.extend(df.to_dict('records'))
        return pd.DataFrame(rows) if rows else None
    nsight.extraction.extract_df_from_report = _patched_extract
    
    # Patch 3: Fix "cannot insert Annotation" bug
    _orig_agg = nsight.transformation.aggregate_data
    def _patched_agg(df, func, norm, progress):
        _orig_gb = pd.DataFrame.groupby
        pd.DataFrame.groupby = lambda self, *a, **kw: _orig_gb(self, *a, **{**kw, 'as_index': False})
        try:
            return _orig_agg(df, func, norm, progress)
        finally:
            pd.DataFrame.groupby = _orig_gb
    nsight.transformation.aggregate_data = _patched_agg

except ImportError:
    NSIGHT_AVAILABLE = False


def profile_with_nsight(func, metrics=None, *args, **kwargs):
    """Profile a PyTorch function. Returns {metric_name: value}."""
    if not NSIGHT_AVAILABLE:
        raise RuntimeError("nsight-python not available")
    
    metrics = [metrics] if isinstance(metrics, str) else (metrics or ['sm__cycles_active.avg'])
    
    @nsight.analyze.kernel(metric=metrics, runs=1, configs=[(0,)], 
                           combine_kernel_metrics=lambda a, b: (a or 0) + (b or 0))
    def profiled(_):
        with nsight.annotate("kernel"):
            return func(*args, **kwargs)
    
    try:
        result = profiled()
        if result is None:
            return {m: None for m in metrics}
        
        df = result.to_dataframe()
        if df is None or df.empty:
            return {m: None for m in metrics}
        
        if 'Metric Name' in df.columns:
            return {row['Metric Name']: float(row['AvgValue']) for _, row in df.iterrows()}
        return {metrics[0]: float(df['AvgValue'].iloc[0])}
    except Exception as e:
        print(f"Error profiling: {e}")
        return {m: None for m in metrics}


# example function to profile with nsight
def example_ncu_python_profile():
    # Test with simple kernel
    def test_kernel(x, y):
        """Simple matmul kernel."""
        return x @ y
    
    print("Creating test tensors...")
    a = torch.randn(256, 256, device="cuda")
    b = torch.randn(256, 256, device="cuda")
    
    print("Running nsight profiling...")
    metric_values = profile_with_nsight(
        test_kernel, 
        ['sm__cycles_active.avg', 'sm__cycles_elapsed.sum', "smsp__inst_executed_pipe_tensor_op_hmma.sum"],
        a, b
    )
    
    print("\nProfiling results:")
    for metric_name, value in metric_values.items():
        print(f"  {metric_name}: {value}")
    return
    

def check_ncu_available() -> bool:
    from shutil import which
    return which('ncu') is not None


@nsight.analyze.kernel
def benchmark_matmul(n):
    """Standard benchmark following nsight-python docs."""
    a = torch.randn(n, n, device="cuda")
    b = torch.randn(n, n, device="cuda")
    with nsight.annotate("matmul"):
        c = a @ b
    return c


if __name__ == "__main__":
    if not check_ncu_available():
        print("ncu not found in PATH. Install Nsight Compute.")
        exit(1)
    
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        exit(1)
        
    
    # test the example_ncu_python_profile
    example_ncu_python_profile()
    
    

# pytorch profiler
# migrate from old repo during ICML / caesar repo