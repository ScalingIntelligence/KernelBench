import torch
from torch.profiler import profile, record_function, ProfilerActivity
from level3.10_ResNet101_SOL import Model, get_inputs

# Initialize model and capture graph first
model = Model().cuda().eval()
inputs = get_inputs()[0]
model.capture_graph() # Pre-capture so it's not in the profile

# Now Profile the Replay
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_flops=True  # Important for SOL calculation
) as prof:
    with record_function("model_inference"):
        # This now executes the entire ResNet101 as 1 Graph Replay
        output = model(inputs)

# Print the same table as before
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

