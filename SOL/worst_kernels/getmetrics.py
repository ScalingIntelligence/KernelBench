import torch
import torch.nn as nn
import triton
import triton.language as tl
import importlib.util

# ==========================================
# 1. SUPER-FUSED TRITON KERNEL
# ==========================================
@triton.jit
def fused_bn_add_relu_kernel(
    x_ptr, ident_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr, out_ptr,
    n_elements, eps, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Fused Loads
    x = tl.load(x_ptr + offsets, mask=mask)
    ident = tl.load(ident_ptr + offsets, mask=mask)
    mean = tl.load(mean_ptr + offsets, mask=mask)
    var = tl.load(var_ptr + offsets, mask=mask)
    weight = tl.load(weight_ptr + offsets, mask=mask)
    bias = tl.load(bias_ptr + offsets, mask=mask)

    # BatchNorm: (x - mean) / sqrt(var + eps) * weight + bias
    inv_std = 1.0 / tl.sqrt(var + eps)
    bn_out = (x - mean) * inv_std * weight + bias
    
    # Residual Add + ReLU
    output = tl.maximum(bn_out + ident, 0.0)
    
    # Final Store
    tl.store(out_ptr + offsets, output, mask=mask)

def triton_bn_add_relu(x, ident, bn_module):
    n_elements = x.numel()
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    fused_bn_add_relu_kernel[grid](
        x, ident, 
        bn_module.running_mean, bn_module.running_var, 
        bn_module.weight, bn_module.bias, 
        out, n_elements, bn_module.eps, 
        BLOCK_SIZE=1024
    )
    return out

# ==========================================
# 2. MODEL DEFINITION
# ==========================================
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.conv3(out) # bn3 is fused below
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # SUPER-FUSION: BatchNorm3 + Add + ReLU
        return triton_bn_add_relu(out, identity, self.bn3)

class ResNet101(nn.Module):
    def __init__(self, layers=[3, 4, 23, 3], num_classes=1000):
        super(ResNet101, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * Bottleneck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * Bottleneck.expansion),
            )
        layers = [Bottleneck(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

# ==========================================
# 3. CUDA GRAPH WRAPPER
# ==========================================
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = ResNet101().cuda().eval()
        self.static_input = torch.randn(10, 3, 224, 224).cuda()
        self.static_output = None
        self.graph = None

    def capture_graph(self):
        print("Capturing CUDA Graph...")
        # Warmup
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(10):
                self.static_output = self.base_model(self.static_input)
        torch.cuda.current_stream().wait_stream(stream)

        # Record
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_output = self.base_model(self.static_input)
        print("Graph Capture Complete.")

    def forward(self, x):
        if self.graph is None:
            self.capture_graph()
        
        self.static_input.copy_(x)
        self.graph.replay()
        return self.static_output

def get_inputs():
    return [torch.randn(10, 3, 224, 224).cuda()]

# ==========================================
# 4. EXECUTION
# ==========================================
if __name__ == "__main__":
    model = Model()
    inputs = get_inputs()[0]
    
    # First call will trigger Graph Capture
    output = model(inputs)
    print(f"Inference Successful. Output shape: {output.shape}")

