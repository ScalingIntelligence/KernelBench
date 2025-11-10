import torch
import traceback

def main():
    try:
        print("torch:", torch.__version__)
        print("cuda available:", torch.cuda.is_available())
        if not torch.cuda.is_available():
            return

        i = 0
        print("device name:", torch.cuda.get_device_name(i))
        print("compute capability:", torch.cuda.get_device_capability(i))
        props = torch.cuda.get_device_properties(i)
        print("total memory (GB):", props.total_memory / 1024**3)

        # Simple alloc / ops to invoke kernels
        a = torch.randn(1024, 1024, device="cuda", dtype=torch.float32)
        b = torch.randn(1024, 1024, device="cuda", dtype=torch.float32)
        print("running matmul...")
        c = a @ b
        print("matmul done, norm:", c.norm().item())

        # A few elementwise ops + reduction
        x = torch.randn(10_000_000, device="cuda")
        y = torch.relu(x)
        print("relu + sum:", y.sum().item())

        # small CUDA kernel via einsum to exercise other codepaths
        d = torch.einsum("ij,jk->ik", a[:512, :512], b[:512, :512])
        print("einsum done, shape:", d.shape)

        print("All GPU ops succeeded.")
    except Exception:
        print("Exception during GPU test:")
        traceback.print_exc()

if __name__ == "__main__":
    main()