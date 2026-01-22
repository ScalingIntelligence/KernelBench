import json
import matplotlib.pyplot as plt
import numpy as np

# 1. Load the data from your JSON
with open('../sol_metrics_from_model.json', 'r') as f:
    data = json.load(f)

peak_bw = data['peak_bw'] / 1e12  # Convert to TB/s
peak_flops = data['peak_flops'] / 1e12  # Convert to TFLOPS
kernel = data['kernels']['softmax_dropout_model']

# 2. Setup the Roofline parameters
# Ridge point is where BW * AI = Peak Flops
ridge_point = peak_flops / peak_bw 
ai_range = np.logspace(-2, 2, 100)

# Calculate Roofline boundary
roofline = np.minimum(peak_flops, peak_bw * ai_range)

# 3. Create the Plot
plt.figure(figsize=(10, 6))
plt.loglog(ai_range, roofline, label='H100 Peak SOL', color='red', lw=2)

# Plot the specific kernel point
plt.scatter(kernel['ai'], kernel['achieved_tflops'], color='blue', s=100, zorder=5)
plt.annotate(f" Softmax-Dropout\n AI: {kernel['ai']:.2f}", 
             (kernel['ai'], kernel['achieved_tflops']), verticalalignment='bottom')

# 4. Formatting
plt.title('H100 Speed-of-Light (SOL) Roofline Analysis')
plt.xlabel('Arithmetic Intensity (FLOPs/Byte)')
plt.ylabel('Performance (TFLOPS)')
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.legend()

# Save the new version
plt.savefig('h100_roofline_final.png')
print("Successfully generated: h100_roofline_final.png")
