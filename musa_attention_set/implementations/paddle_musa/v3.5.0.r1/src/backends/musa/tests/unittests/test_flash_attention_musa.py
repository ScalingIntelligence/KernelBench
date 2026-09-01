import logging
import os
import re
import unittest
import math

import numpy as np
from op_test import get_device_place, is_custom_device

import paddle
import paddle.nn.functional as F
from paddle import base
from paddle.nn.functional.flash_attention import (
    flash_attention,
    flash_attn_unpadded
)
from paddle.nn.functional import scaled_dot_product_attention
from paddle.nn.attention import SDPBackend, sdpa_kernel

import custom_setup_ops

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")


def get_cuda_version():
    result = os.popen("nvcc --version").read()
    regex = r'release (\S+),'
    match = re.search(regex, result)
    if match:
        num = str(match.group(1))
        integer, decimal = num.split('.')
        return int(integer) * 1000 + int(float(decimal) * 10)
    else:
        return -1


def attention_naive(q, k, v, causal=False):
    """
    Differentiable, stable reference:
    - scores/matmul in fp32
    - softmax implemented manually in fp32 (avoid backend softmax kernel NaN)
    - cast output back to original dtype
    q,k,v: [B, L, H, D]
    return: [B, L, H, D]
    """
    orig_dtype = q.dtype

    # [B,H,L,D]
    qt = paddle.transpose(q, [0, 2, 1, 3]).astype('float32')
    kt = paddle.transpose(k, [0, 2, 1, 3]).astype('float32')
    vt = paddle.transpose(v, [0, 2, 1, 3]).astype('float32')

    scale = 1.0 / np.sqrt(float(q.shape[-1]))
    s = paddle.matmul(qt * scale, paddle.transpose(kt, [0, 1, 3, 2]))  # [B,H,L,L] fp32

    if causal:
        L = s.shape[-1]
        m = paddle.triu(paddle.ones([L, L], dtype='float32'), diagonal=1) * (-1e9)
        s = s + m

    s = s - paddle.max(s, axis=-1, keepdim=True)   # max-shift
    e = paddle.exp(s)                              # exp in fp32
    denom = paddle.sum(e, axis=-1, keepdim=True)   # sum in fp32
    p = e / denom                                  # [B,H,L,L] fp32

    o = paddle.matmul(p, vt)                       # [B,H,L,D] fp32
    out = paddle.transpose(o, [0, 2, 1, 3]).astype(orig_dtype)
    return out

def attention_naive_with_mask(q, k, v, attn_bias):
    """
    Try to match kernel behavior where scale applies to (QK^T + attn_bias):
      softmax( (QK^T + attn_bias) * scale ) @ V
    """
    orig_dtype = q.dtype
    D = q.shape[-1]

    qt = paddle.transpose(q, [0, 2, 1, 3]).astype("float32")
    kt = paddle.transpose(k, [0, 2, 1, 3]).astype("float32")
    vt = paddle.transpose(v, [0, 2, 1, 3]).astype("float32")

    s = paddle.matmul(qt, paddle.transpose(kt, [0, 1, 3, 2]))  # fp32

    if attn_bias is not None:
        # IMPORTANT: add bias BEFORE scaling
        s = s + attn_bias.astype("float32")

    scale = 1.0 / np.sqrt(float(D))
    s = s * scale

    # manual stable softmax fp32
    s = s - paddle.max(s, axis=-1, keepdim=True)
    e = paddle.exp(s)
    denom = paddle.sum(e, axis=-1, keepdim=True)
    denom = paddle.where(denom == 0, paddle.ones_like(denom), denom)
    p = e / denom

    o = paddle.matmul(p, vt)
    out = paddle.transpose(o, [0, 2, 1, 3]).astype(orig_dtype)
    return out


def repeat_kv_for_gqa(k, v, num_heads_q):
    b, sk, hk, d = k.shape
    assert num_heads_q % hk == 0
    rep = num_heads_q // hk
    k2 = paddle.repeat_interleave(k, rep, axis=2)
    v2 = paddle.repeat_interleave(v, rep, axis=2)
    return k2, v2


def attention_naive_kvcache(q, k, v, causal=False, scale=None, seqused_k=None):
    orig_dtype = q.dtype
    B, Sq, H, D = q.shape
    Sk = k.shape[1]
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    qt = paddle.transpose(q, [0, 2, 1, 3]).astype("float32")  # [B,H,Sq,D]
    kt = paddle.transpose(k, [0, 2, 1, 3]).astype("float32")  # [B,H,Sk,D]
    vt = paddle.transpose(v, [0, 2, 1, 3]).astype("float32")  # [B,H,Sk,D]

    s = paddle.matmul(qt * scale, paddle.transpose(kt, [0, 1, 3, 2]))  # [B,H,Sq,Sk]

    mask = None

    if seqused_k is not None:
        seqlen = seqused_k.astype("int64").reshape([B, 1, 1, 1])  # [B,1,1,1]
        j = paddle.arange(Sk, dtype="int64").reshape([1, 1, 1, Sk])  # [1,1,1,Sk]
        pad_mask = (j >= seqlen)
        mask = pad_mask if mask is None else (mask | pad_mask)

    if causal:
        Sq = s.shape[-2]
        Sk = s.shape[-1]
        offset = Sk - Sq
        i = paddle.arange(Sq, dtype="int64").reshape([1, 1, Sq, 1])
        j = paddle.arange(Sk, dtype="int64").reshape([1, 1, 1, Sk])
        causal_mask = (j > (offset + i))
        s = s + paddle.where(causal_mask, paddle.full_like(s, -1e9), paddle.zeros_like(s))

    if mask is not None:
        s = s + paddle.where(mask, paddle.full_like(s, -1e9), paddle.zeros_like(s))

    s = s - paddle.max(s, axis=-1, keepdim=True)
    e = paddle.exp(s)
    denom = paddle.sum(e, axis=-1, keepdim=True)
    p = e / denom

    o = paddle.matmul(p, vt)  # [B,H,Sq,D]
    out = paddle.transpose(o, [0, 2, 1, 3]).astype(orig_dtype)
    return out


def make_paged_kv_from_contiguous(k, v, block_size, page_perm):
    B, Sk, Hk, D = k.shape
    assert Sk % block_size == 0
    nblocks = Sk // block_size
    num_pages = B * nblocks

    k_blk = k.reshape([B, nblocks, block_size, Hk, D])
    v_blk = v.reshape([B, nblocks, block_size, Hk, D])

    k_cache = paddle.empty([num_pages, block_size, Hk, D], dtype=k.dtype)
    v_cache = paddle.empty([num_pages, block_size, Hk, D], dtype=v.dtype)

    block_table = paddle.empty([B, nblocks], dtype="int32")

    page_perm = np.asarray(page_perm, dtype=np.int32)
    assert page_perm.shape[0] == num_pages

    for b in range(B):
        for blk in range(nblocks):
            logical_pid = b * nblocks + blk
            phys_pid = int(page_perm[logical_pid])
            k_cache[phys_pid] = k_blk[b, blk]
            v_cache[phys_pid] = v_blk[b, blk]
            block_table[b, blk] = phys_pid

    return k_cache, v_cache, block_table

def gather_paged_kv_to_contiguous(k_cache, v_cache, block_table, Sk, block_size):
    B, nblocks = block_table.shape
    _, _, Hk, D = k_cache.shape
    assert Sk == nblocks * block_size

    k_out = paddle.empty([B, Sk, Hk, D], dtype=k_cache.dtype)
    v_out = paddle.empty([B, Sk, Hk, D], dtype=v_cache.dtype)

    for b in range(B):
        for blk in range(nblocks):
            pid = int(block_table[b, blk].item())
            k_out[b, blk*block_size:(blk+1)*block_size] = k_cache[pid]
            v_out[b, blk*block_size:(blk+1)*block_size] = v_cache[pid]
    return k_out, v_out


is_sm80 = True
is_sm8x = False
is_sm90 = False
is_sm_supported = is_sm8x or is_sm90

def is_flashattn_supported():
    # if (
    #     not (core.is_compiled_with_cuda() or is_custom_device())
    #     or get_cuda_version() < 11040
    #     or not is_sm_supported
    # ):
    #     return False
    return True

class TestFlashAttentionAPI(unittest.TestCase):
    def setUp(self):
        self.place = get_device_place()
        self.shape = (1, 256, 8, 256)
        # self.shape = (1, 32768, 16, 256)
        # self.shape = (1, 8192, 16, 256)
        # self.shape = (1, 32768, 16, 128)
        self.dtype = 'float16'
        self.dropout = 0.0
        self.causal = False
        self.return_softmax = False
        self.use_sdp_kernel = False
        self.use_sdp_api = False

    def test_all(self):
        # print(
        #     f"Test case shape {self.shape} dtype {self.dtype} causal {self.causal}"
        # )
        # test dynamic
        paddle.disable_static()

        query = np.random.random(self.shape)
        key = np.random.random(self.shape)
        value = np.random.random(self.shape)

        q = paddle.to_tensor(
            query, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        k = paddle.to_tensor(
            key, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        v = paddle.to_tensor(
            value, place=self.place, dtype=self.dtype, stop_gradient=False
        )

        q_ = paddle.to_tensor(
            query, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        k_ = paddle.to_tensor(
            key, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        v_ = paddle.to_tensor(
            value, place=self.place, dtype=self.dtype, stop_gradient=False
        )

        if self.use_sdp_kernel:
            with paddle.nn.functional.sdp_kernel(
                enable_math=self.enable_math,
                enable_flash=self.enable_flash,
                enable_mem_efficient=self.enable_mem_efficient,
            ):
                if self.use_sdp_api:
                    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                        out = scaled_dot_product_attention(
                            q, k, v, None, self.dropout, self.causal
                        )
                else:
                    out, _ = flash_attention(
                        q, k, v, self.dropout, self.causal, self.return_softmax
                    )

        else:
            out, _ = flash_attention(
                q, k, v, self.dropout, self.causal, self.return_softmax
            )

        out_ = attention_naive(q_, k_, v_, self.causal)

        np.testing.assert_allclose(out.numpy(), out_, rtol=5e-03, atol=1e-03)

        out.backward()
        out_.backward()

        self.assertEqual(q.grad.shape, q.shape)
        self.assertEqual(q_.grad.shape, q.shape)

        np.testing.assert_allclose(
            q.grad.numpy(), q_.grad.numpy(), rtol=5e-03, atol=2e-03
        )

        # test static
        paddle.enable_static()

        with paddle.static.program_guard(paddle.static.Program()):
            qs = paddle.static.data(
                name="q", shape=self.shape, dtype=self.dtype
            )
            ks = paddle.static.data(
                name="k", shape=self.shape, dtype=self.dtype
            )
            vs = paddle.static.data(
                name="v", shape=self.shape, dtype=self.dtype
            )

            if self.use_sdp_kernel:
                with paddle.nn.functional.sdp_kernel(
                    enable_math=self.enable_math,
                    enable_flash=self.enable_flash,
                    enable_mem_efficient=self.enable_mem_efficient,
                ):
                    if self.use_sdp_api:
                        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                            outs = scaled_dot_product_attention(
                                qs, ks, vs, None, self.dropout, self.causal
                            )
                    else:
                        outs, softmax = flash_attention(
                            qs,
                            ks,
                            vs,
                            self.dropout,
                            self.causal,
                            self.return_softmax,
                        )
            else:
                outs, softmax = flash_attention(
                    qs, ks, vs, self.dropout, self.causal, self.return_softmax
                )

            exe = base.Executor(self.place)
            fetches_result = exe.run(
                feed={
                    "q": query.astype('float16'),
                    "k": key.astype('float16'),
                    "v": value.astype('float16'),
                },
                fetch_list=[outs],
            )

            np.testing.assert_allclose(
                fetches_result[0], out_, rtol=5e-03, atol=1e-03
            )

        paddle.disable_static()


class TestFlashAttentionWithMaskAPI(unittest.TestCase):
    def setUp(self):
        self.place = get_device_place()
        self.shape = (2, 128, 8, 32)
        self.dtype = 'float16'
        self.dropout = 0.0
        self.causal = False

    def test_dot_scale_product(self):
        # test dynamic
        paddle.disable_static()

        query = np.random.random(self.shape)
        key = np.random.random(self.shape)
        value = np.random.random(self.shape)

        q = paddle.to_tensor(
            query, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        k = paddle.to_tensor(
            key, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        v = paddle.to_tensor(
            value, place=self.place, dtype=self.dtype, stop_gradient=False
        )

        q_ = paddle.to_tensor(
            query, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        k_ = paddle.to_tensor(
            key, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        v_ = paddle.to_tensor(
            value, place=self.place, dtype=self.dtype, stop_gradient=False
        )

        mask_shape = (self.shape[0], 1, self.shape[1], self.shape[1])
        mask = np.random.random(mask_shape)
        m = paddle.to_tensor(
            mask, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            out = scaled_dot_product_attention(
                q, k, v, m, self.dropout, self.causal
            )
        out_ = attention_naive_with_mask(q_, k_, v_, m)

        np.testing.assert_allclose(out.numpy(), out_, rtol=5e-03, atol=1e-03)

        out.backward()
        out_.backward()


class TestFlashAttentionAPITest1(TestFlashAttentionAPI):
    def setUp(self):
        self.place = get_device_place()
        self.shape = (2, 128, 8, 16)
        self.dtype = 'float16'
        self.dropout = 0.0
        self.causal = False
        self.return_softmax = False
        self.use_sdp_kernel = False


class TestFlashAttentionAPITest2(TestFlashAttentionAPI):
    def setUp(self):
        self.place = get_device_place()
        self.shape = (2, 256, 8, 16)
        self.dtype = 'float16'
        self.dropout = 0.0
        self.causal = False
        self.return_softmax = False
        self.use_sdp_kernel = False


class TestFlashAttentionAPITest3(TestFlashAttentionAPI):
    def setUp(self):
        self.place = get_device_place()
        self.shape = (2, 512, 8, 16)
        self.dtype = 'float16'
        self.dropout = 0.0
        self.causal = True
        self.return_softmax = False
        self.use_sdp_kernel = False


class TestFlashAttentionAPITest4(TestFlashAttentionAPI):
    def setUp(self):
        self.place = get_device_place()
        self.shape = (8, 1024, 16, 128)
        self.dtype = 'float16'
        self.dropout = 0.0
        self.causal = False
        self.return_softmax = False
        self.use_sdp_kernel = False


class TestFlashAttentionAPITest5(TestFlashAttentionAPI):
    def setUp(self):
        self.place = get_device_place()
        self.shape = (
            (8, 1024, 16, 256) if (is_sm80 or is_sm90) else (8, 1024, 16, 192)
        )
        self.dtype = 'float16'
        self.dropout = 0.0
        self.causal = False
        self.return_softmax = False
        self.use_sdp_kernel = False


class TestMathAttentionAPITest(TestFlashAttentionAPI):
    def setUp(self):
        self.place = get_device_place()
        self.shape = (8, 1024, 16, 128)
        self.dtype = 'float16'
        self.dropout = 0.0
        self.causal = False
        self.return_softmax = False
        self.use_sdp_kernel = True
        self.use_sdp_api = False
        self.enable_math = True
        self.enable_flash = False
        self.enable_mem_efficient = False


class TestSDPAttentionAPITest(TestFlashAttentionAPI):
    def setUp(self):
        self.place = get_device_place()
        self.shape = (8, 1024, 16, 128)
        self.dtype = 'float16'
        self.dropout = 0.0
        self.causal = False
        self.return_softmax = False
        self.use_sdp_kernel = True
        self.use_sdp_api = True
        self.enable_math = True
        self.enable_flash = False
        self.enable_mem_efficient = False


class TestFlashAttentionWithMaskAPITest(TestFlashAttentionWithMaskAPI):
    def setUp(self):
        self.place = get_device_place()
        self.shape = (8, 1024, 16, 128)
        self.dtype = 'float16'
        self.dropout = 0.0
        self.causal = False


class TestFlashAttentionGQA(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.num_head = 8
        self.seq_len = 8192
        self.head_dim = 128
        self.num_group = 1
        self.dtype = 'bfloat16'

    def gen_unpadded_data(self, dtype):
        seq_len_q = np.random.randint(
            low=1, high=self.seq_len, size=[self.batch_size]
        )
        seq_len_k = np.random.randint(
            low=1, high=self.seq_len, size=[self.batch_size]
        )
        cu_seqlen_q = paddle.to_tensor(
            [0, *np.cumsum(seq_len_q).tolist()], dtype=paddle.int32
        )
        cu_seqlen_k = paddle.to_tensor(
            [0, *np.cumsum(seq_len_k).tolist()], dtype=paddle.int32
        )

        qs, ks, vs = [], [], []
        for i in range(self.batch_size):
            tmp_q = (
                paddle.randn(
                    [seq_len_q[i] * self.num_head * self.head_dim], dtype=dtype
                )
                / 1e2
            )
            tmp_k = (
                paddle.randn(
                    [
                        seq_len_k[i]
                        * self.num_head
                        * self.head_dim
                        // self.num_group
                    ],
                    dtype=dtype,
                )
                / 1e2
            )
            tmp_v = (
                paddle.randn(
                    [
                        seq_len_k[i]
                        * self.num_head
                        * self.head_dim
                        // self.num_group
                    ],
                    dtype=dtype,
                )
                / 1e2
            )
            qs.append(tmp_q)
            ks.append(tmp_k)
            vs.append(tmp_v)

        q = paddle.concat(qs, axis=0).reshape(
            [-1, self.num_head, self.head_dim]
        )
        k = paddle.concat(ks, axis=0).reshape(
            [-1, self.num_head // self.num_group, self.head_dim]
        )
        v = paddle.concat(vs, axis=0).reshape(
            [-1, self.num_head // self.num_group, self.head_dim]
        )
        return q, k, v, cu_seqlen_q, cu_seqlen_k

    def gen_test_data(self, dtype, use_unpadded):
        assert self.num_head % self.num_group == 0
        if use_unpadded:
            q, k, v, cu_seqlen_q, cu_seqlen_k = self.gen_unpadded_data(dtype)
        else:
            q = (
                paddle.randn(
                    [
                        self.batch_size,
                        self.seq_len,
                        self.num_head,
                        self.head_dim,
                    ],
                    dtype=dtype,
                )
                / 1e2
            )
            k = (
                paddle.randn(
                    [
                        self.batch_size,
                        self.seq_len,
                        self.num_head // self.num_group,
                        self.head_dim,
                    ],
                    dtype=dtype,
                )
                / 1e2
            )
            v = (
                paddle.randn(
                    [
                        self.batch_size,
                        self.seq_len,
                        self.num_head // self.num_group,
                        self.head_dim,
                    ],
                    dtype=dtype,
                )
                / 1e2
            )
            cu_seqlen_q = None
            cu_seqlen_k = None
        out_grad = paddle.randn(q.shape, dtype=dtype) / 1e2
        return q, k, v, cu_seqlen_q, cu_seqlen_k, out_grad

    def clone_tensor(self, tensor):
        if tensor is None:
            return None
        elif isinstance(tensor, (list, tuple)):
            return [self.clone_tensor(t) for t in tensor]
        else:
            tensor = tensor.detach().clone()
            tensor.stop_gradient = False
            return tensor

    @paddle.no_grad()
    def convert_dtype(self, tensors):
        ret = []
        for t in tensors:
            if t.dtype in [paddle.float16, paddle.bfloat16]:
                t = t.astype(paddle.float32)
            t = t.numpy()
            ret.append(t)
        return ret

    def calc_fa(
        self, q, k, v, cu_seqlen_q, cu_seqlen_k, out_grad, causal, use_unpadded
    ):
        q, k, v = self.clone_tensor([q, k, v])
        if use_unpadded:
            scale = self.head_dim ** (-0.5)
            out = flash_attn_unpadded(
                q,
                k,
                v,
                cu_seqlens_q=cu_seqlen_q,
                cu_seqlens_k=cu_seqlen_k,
                max_seqlen_q=self.seq_len,
                max_seqlen_k=self.seq_len,
                scale=scale,
                causal=causal,
            )
        else:
            out = flash_attention(q, k, v, causal=causal)
        out = out[0]
        out.backward(out_grad)
        return self.convert_dtype([out, q.grad, k.grad, v.grad])

    def calc_raw_attn(
        self, q, k, v, cu_seqlen_q, cu_seqlen_k, out_grad, causal, use_unpadded
    ):
        def ref_softmax(x, axis=-1):
            x = x.astype("float32")
            m = paddle.max(x, axis=axis, keepdim=True)
            y = paddle.exp(x - m)
            s = paddle.sum(y, axis=axis, keepdim=True)
            return y / s
        
        q, k, v = self.clone_tensor([q, k, v])
        if use_unpadded:
            qq, q_mask = self.pad(q, cu_seqlen_q, self.seq_len)
            kk, k_mask = self.pad(k, cu_seqlen_k, self.seq_len)
            vv, _ = self.pad(v, cu_seqlen_k, self.seq_len)
            qk_mask = paddle.matmul(q_mask, k_mask, transpose_y=True)
            qk_mask = qk_mask.reshape(
                [self.batch_size, 1, self.seq_len, self.seq_len]
            )
            qk_mask[qk_mask == 0] = -1e4
            qk_mask[qk_mask == 1] = 0
        else:
            qq, kk, vv = q, k, v

        assert len(qq.shape) == 4, qq.shape
        assert len(kk.shape) == 4, kk.shape
        assert len(vv.shape) == 4, vv.shape
        perm = [0, 2, 1, 3]
        qq = paddle.transpose(qq, perm)
        kk = paddle.transpose(kk, perm)
        kk = paddle.stack([kk] * self.num_group, axis=2).reshape(qq.shape)
        vv = paddle.transpose(vv, perm)
        vv = paddle.stack([vv] * self.num_group, axis=2).reshape(qq.shape)
        scale = self.head_dim ** (-0.5)
        weight = paddle.matmul(qq * scale, kk, transpose_y=True)
        if use_unpadded:
            weight += qk_mask
        if causal:
            shape = weight.shape[-2:]
            mask = paddle.full(shape, -np.inf, dtype=weight.dtype)
            mask = paddle.triu(mask, diagonal=1)
            weight += mask

        weight = weight.astype(paddle.float32)
        
        weight = ref_softmax(weight, axis=-1)
        
        out = paddle.matmul(weight.astype(vv.dtype), vv)
        out = paddle.transpose(out, perm)
        if use_unpadded:
            out = self.unpad(out, cu_seqlen_q)
        out.backward(out_grad)
        return self.convert_dtype([out, q.grad, k.grad, v.grad])

    def pad(self, x, cu_seqlen, max_seqlen):
        cu_seqlen_cpu = cu_seqlen.numpy()
        split_sections = []
        for i in range(len(cu_seqlen_cpu) - 1):
            split_sections.append(cu_seqlen_cpu[i + 1] - cu_seqlen_cpu[i])

        tmp_xs = paddle.split(x, split_sections)
        batch_size = len(tmp_xs)
        tmp_masks = []
        tmp_x_pads = []
        for i in range(batch_size):
            tmp_mask = paddle.ones([max_seqlen], dtype=x.dtype)
            tmp_mask[split_sections[i] :] = 0
            tmp_mask = tmp_mask.reshape([1, -1, 1])
            tmp_masks.append(tmp_mask)

            tmp_shape = tmp_xs[i].shape
            tmp_pad = paddle.zeros(
                [max_seqlen - tmp_shape[0], *tmp_shape[1:]], dtype=x.dtype
            )
            tmp_x = paddle.concat([tmp_xs[i], tmp_pad]).unsqueeze(0)
            tmp_x_pads.append(tmp_x)

        x_pad = paddle.concat(tmp_x_pads)
        mask = paddle.concat(tmp_masks)
        return x_pad, mask

    def unpad(self, x, cu_seqlen):
        cu_seqlen_cpu = cu_seqlen.numpy()
        xs = paddle.split(x, x.shape[0])
        tmp_xs = []
        for i in range(len(cu_seqlen_cpu) - 1):
            tmp = xs[i].squeeze(0)[: cu_seqlen_cpu[i + 1] - cu_seqlen_cpu[i]]
            tmp_xs.append(tmp)
        unpad_x = paddle.concat(tmp_xs)
        return unpad_x

    def test_main(self):
        # test dynamic
        paddle.disable_static()

        for causal in [False, True]:
            for use_unpadded in [True, False]:
                (
                    q,
                    k,
                    v,
                    cu_seqlen_q,
                    cu_seqlen_k,
                    out_grad,
                ) = self.gen_test_data(self.dtype, use_unpadded)
                fa_out = self.calc_fa(
                    q,
                    k,
                    v,
                    cu_seqlen_q,
                    cu_seqlen_k,
                    out_grad,
                    causal,
                    use_unpadded,
                )
                raw_out = self.calc_raw_attn(
                    q,
                    k,
                    v,
                    cu_seqlen_q,
                    cu_seqlen_k,
                    out_grad,
                    causal,
                    use_unpadded,
                )
                assert len(fa_out) == len(raw_out)

                for t1, t2 in zip(fa_out, raw_out):
                    np.testing.assert_allclose(t1, t2, atol=1e-2, rtol=1e-2)
                print("done now")


class TestFlashAttnKVCacheMate(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    def run_one(self, dtype="bfloat16", causal=False):
        B = 2
        Sq = 8192
        Sk = 8192
        Hq = 8
        Hk = 2  
        D = 128 

        q = paddle.randn([B, Sq, Hq, D], dtype=dtype)
        k = paddle.randn([B, Sk, Hk, D], dtype=dtype)
        v = paddle.randn([B, Sk, Hk, D], dtype=dtype)

        scale = 1.0 / math.sqrt(D)

        out, lse = custom_setup_ops.flash_attn_kvcache_mate(
            q_=q, k_=k, v_=v,
            k_new_=None, v_new_=None, q_v_=None, out_=None,
            cu_seqlens_q_=None, cu_seqlens_k_=None, cu_seqlens_k_new_=None,
            seqused_q_=None, seqused_k_=None,
            page_table_=None, kv_batch_idx_=None, leftpad_k_=None,
            rotary_cos_=None, rotary_sin_=None, seqlens_rotary_=None,
            q_descale_=None, k_descale_=None, v_descale_=None,
            scheduler_metadata_=None,
            max_seqlen_q_=-1, max_seqlen_k_=-1,
            softmax_scale_=scale, is_causal=causal,
            window_size_left=-1, window_size_right=-1,
            attention_chunk=0, softcap=0.0, is_rotary_interleaved=False,
            num_splits=1, pack_gqa_=-1, sm_margin=0
        )
        paddle.device.synchronize()

        k_ref, v_ref = repeat_kv_for_gqa(k, v, Hq)
        out_ref = attention_naive_kvcache(q, k_ref, v_ref, causal)

        out_np = out.astype("float32").cpu().numpy()
        ref_np = out_ref.astype("float32").cpu().numpy()

        np.testing.assert_allclose(
            out_np, ref_np,
            rtol=5e-2,
            atol=5e-2
        )

        self.assertTrue(paddle.isfinite(lse).all().item())

    def test_bf16_noncausal(self):
        self.run_one(dtype="bfloat16", causal=False)
    
    def test_bf16_causal(self):
        self.run_one(dtype="bfloat16", causal=True)


class TestFlashAttnKVCacheMatePaged(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    def _build_paged_inputs(self, B, pages_per_seq, Hk, D, dtype, seed=123):
        page_size = 64
        Sk = pages_per_seq * page_size

        rng = np.random.default_rng(seed)

        num_pages = B * pages_per_seq

        perm = rng.permutation(num_pages).astype(np.int32)
        page_table_np = perm.reshape(B, pages_per_seq)  # [B, pages_per_seq]
        page_table = paddle.to_tensor(page_table_np, dtype="int32")

        # page pool
        k_pool = paddle.randn([num_pages, page_size, Hk, D], dtype=dtype)
        v_pool = paddle.randn([num_pages, page_size, Hk, D], dtype=dtype)

        k_contig = paddle.empty([B, Sk, Hk, D], dtype=dtype)
        v_contig = paddle.empty([B, Sk, Hk, D], dtype=dtype)

        for b in range(B):
            for p in range(pages_per_seq):
                pid = int(page_table_np[b, p])
                s0 = p * page_size
                s1 = (p + 1) * page_size
                k_contig[b, s0:s1] = k_pool[pid]
                v_contig[b, s0:s1] = v_pool[pid]

        return k_pool, v_pool, page_table, k_contig, v_contig, Sk

    def run_one_paged(self, dtype="bfloat16", causal=False):
        B = 2
        Sq = 64
        Hq = 8
        Hk = 2
        D = 128
        pages_per_seq = 4  # => Sk = 4*64 = 256
        scale = 1.0 / math.sqrt(D)

        q = paddle.randn([B, Sq, Hq, D], dtype=dtype)

        k_pool, v_pool, page_table, k_contig, v_contig, Sk = self._build_paged_inputs(
            B=B, pages_per_seq=pages_per_seq, Hk=Hk, D=D, dtype=dtype, seed=123
        )

        seqused_k = paddle.full([B], Sk, dtype="int32")

        out, lse = custom_setup_ops.flash_attn_kvcache_mate(
            q_=q, k_=k_pool, v_=v_pool,
            k_new_=None, v_new_=None, q_v_=None, out_=None,
            cu_seqlens_q_=None, cu_seqlens_k_=None, cu_seqlens_k_new_=None,
            seqused_q_=None, seqused_k_=seqused_k,
            max_seqlen_q_=Sq, max_seqlen_k_=Sk,   
            page_table_=page_table,               
            kv_batch_idx_=None,
            leftpad_k_=None,
            rotary_cos_=None, rotary_sin_=None, seqlens_rotary_=None,
            q_descale_=None, k_descale_=None, v_descale_=None,
            softmax_scale_=scale, is_causal=causal,
            window_size_left=-1, window_size_right=-1,
            attention_chunk=0, softcap=0.0, is_rotary_interleaved=False,
            scheduler_metadata_=None,
            num_splits=1, pack_gqa_=-1, sm_margin=0
        )

        out_np = out.astype("float32").cpu().numpy()

        # ===== reference =====
        k_ref, v_ref = repeat_kv_for_gqa(k_contig, v_contig, Hq)
        out_ref = attention_naive_kvcache(q, k_ref, v_ref, causal=causal, scale=scale, seqused_k=seqused_k)
        ref_np = out_ref.astype("float32").cpu().numpy()

        np.testing.assert_allclose(out_np, ref_np, rtol=5e-2, atol=5e-2)
        self.assertTrue(paddle.isfinite(lse).all().item())

    def test_bf16_paged_noncausal(self):
        self.run_one_paged(dtype="bfloat16", causal=False)

    def test_bf16_paged_causal(self):
        self.run_one_paged(dtype="bfloat16", causal=True)


if __name__ == '__main__':
    unittest.main()
