"""Exercise the shipped ViT reference through image tokens and evaluation."""

import re
import runpy
from pathlib import Path

import pytest
import torch
from torch import nn

REFERENCE_PATH = (
    Path(__file__).resolve().parents[3] / "KernelBench/level3/28_VisionTransformer.py"
)


def explicit_patches(image, patch_size):
    return torch.stack(
        [
            image[:, :, row : row + patch_size, column : column + patch_size].flatten(1)
            for row in range(0, image.shape[2], patch_size)
            for column in range(0, image.shape[3], patch_size)
        ],
        dim=1,
    )


def sequence_first_output(model, image, dim, heads, mlp_dim, depth):
    encoder = (
        nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim, heads, mlp_dim, dropout=0.0),
            num_layers=depth,
            enable_nested_tensor=False,
        )
        .to(device=image.device, dtype=image.dtype)
        .eval()
    )
    encoder.load_state_dict(model.transformer.state_dict())
    embedded = model.patch_to_embedding(explicit_patches(image, model.patch_size))
    tokens = torch.cat(
        (model.cls_token.expand(image.shape[0], -1, -1), embedded), dim=1
    )
    tokens = model.dropout(tokens + model.pos_embedding)
    encoded = encoder(tokens.transpose(0, 1)).transpose(0, 1)
    return model.mlp_head(encoded[:, 0])


@pytest.mark.parametrize(
    "batch_size,channels,patch_size", [(1, 3, 2), (2, 1, 2), (3, 3, 4)]
)
def test_image_tokens_match_sequence_first_reference(batch_size, channels, patch_size):
    reference = runpy.run_path(str(REFERENCE_PATH))
    torch.manual_seed(17)
    model = (
        reference["Model"](8, patch_size, 5, 16, 2, 4, 32, channels, 0.0, 0.0)
        .double()
        .eval()
    )
    image = torch.arange(batch_size * channels * 64, dtype=torch.float64)
    image = image.reshape(batch_size, channels, 8, 8).transpose(2, 3) / image.numel()
    image.requires_grad_()
    captured = []
    hook = model.patch_to_embedding.register_forward_pre_hook(
        lambda module, inputs: captured.append(inputs[0].detach().clone())
    )
    actual = model(image)
    hook.remove()
    patches = explicit_patches(image, patch_size)
    if not torch.equal(captured[0], patches):
        pytest.fail("An image token mixes pixels from different spatial patches")
    oracle_image = image.detach().clone().requires_grad_()
    expected = sequence_first_output(model, oracle_image, 16, 4, 32, 2)
    if not torch.allclose(actual, expected, atol=1e-10, rtol=1e-10):
        pytest.fail(
            "ViT output differs from attention over each image's token sequence"
        )
    weights = torch.arange(1, 6, dtype=torch.float64)
    actual_grads = torch.autograd.grad(
        (actual * weights).sum(), (image, model.patch_to_embedding.weight)
    )
    expected_grads = torch.autograd.grad(
        (expected * weights).sum(), (oracle_image, model.patch_to_embedding.weight)
    )
    for name, actual_grad, expected_grad in zip(
        ("image", "patch projection"), actual_grads, expected_grads
    ):
        if (
            not torch.isfinite(actual_grad).all()
            or torch.count_nonzero(actual_grad) == 0
        ):
            pytest.fail(f"The class output has no finite, nonzero {name} gradient")
        if not torch.allclose(actual_grad, expected_grad, atol=1e-10, rtol=1e-10):
            pytest.fail(f"The {name} gradient differs from the independent reference")
    with torch.no_grad():
        separate = torch.cat([model(sample.unsqueeze(0)) for sample in image], dim=0)
    if not torch.allclose(actual, separate, atol=1e-10, rtol=1e-10):
        pytest.fail("Class predictions depend on which other images share the batch")


def test_shipped_vit_depends_on_image_values():
    reference = runpy.run_path(str(REFERENCE_PATH))
    torch.manual_seed(42)
    model = reference["Model"](*reference["get_init_inputs"]()).eval()
    image = reference["get_inputs"]()[0]
    with torch.no_grad():
        actual = model(image)
        changed = model(10 * image)
    if actual.shape != (image.shape[0], reference["num_classes"]):
        pytest.fail(f"Unexpected class prediction shape {actual.shape}")
    if not torch.isfinite(actual).all() or not torch.isfinite(changed).all():
        pytest.fail("The shipped ViT produced non-finite class predictions")
    if torch.allclose(actual, changed, atol=1e-4, rtol=1e-4):
        pytest.fail("The shipped ViT ignores a change in its image inputs")


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="KernelBench evaluation requires a CUDA or ROCm device",
)
def test_gpu_evaluator_rejects_image_independent_vit():
    from kernelbench.eval import eval_kernel_against_ref

    source = REFERENCE_PATH.read_text()
    matching_source = re.sub(r"\bModel\b", "ModelNew", source)
    image_independent_source = matching_source.replace(
        "        p = self.patch_size",
        "        img = torch.zeros_like(img)\n        p = self.patch_size",
    )
    for candidate, expected in [
        (matching_source, True),
        (image_independent_source, False),
    ]:
        result = eval_kernel_against_ref(
            original_model_src=source,
            custom_model_src=candidate,
            num_correct_trials=3,
            measure_performance=False,
            device=torch.device("cuda:0"),
            backend="hip" if torch.version.hip else "cuda",
        )
        if not result.compiled or result.correctness != expected:
            pytest.fail(f"Unexpected ViT evaluation result {result}")
