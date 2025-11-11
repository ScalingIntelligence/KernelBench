# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from pykernel import (
    PyKernelOp,
    reader_thread,
    writer_thread,
    compute_thread,
    CircularBuffer,
)

from math import ceil

import ttnn
import torch
import torch.nn as nn


class VecAddMulticorePyKernelOp(PyKernelOp):
    def __init__(self, max_core_ranges=None):
        super().__init__()
        self.max_core_ranges = max_core_ranges

    # KERNEL DEFINITIONS
    @compute_thread()
    def add_multicore(
        cb_in0: CircularBuffer,
        cb_in1: CircularBuffer,
        cb_out: CircularBuffer,
        num_tiles,
        start_tile_id,
    ):
        binary_op_init_common(cb_in0, cb_in1, cb_out)
        add_tiles_init(cb_in0, cb_in1)

        end_tile_id = start_tile_id + num_tiles
        dst_reg = 0

        for i in range(start_tile_id, end_tile_id, 1):
            cb_wait_front(cb_in0, 1)
            cb_wait_front(cb_in1, 1)
            tile_regs_acquire()
            add_tiles(cb_in0, cb_in1, 0, 0, dst_reg)
            tile_regs_commit()

            cb_reserve_back(cb_out, 1)
            tile_regs_wait()
            pack_tile(dst_reg, cb_out, 0)
            tile_regs_release()

            cb_push_back(cb_out, 1)
            cb_pop_front(cb_in0, 1)
            cb_pop_front(cb_in1, 1)
            tile_regs_release()
        return

    @writer_thread()
    def writer_multicore(
        cb_out: CircularBuffer,
        dst_addr,
        num_tiles,
        start_id,
    ):
        onetile = 1

        tile_bytes = get_tile_size(cb_out)
        tensor_accessor_args = TensorAccessorArgs(1, 0)
        s0 = TensorAccessor(tensor_accessor_args, dst_addr, tile_bytes)

        end_id = start_id + num_tiles
        for i in range(start_id, end_id, onetile):
            cb_wait_front(cb_out, onetile)
            l1_read_addr = get_read_ptr(cb_out)
            noc_async_write_tile(i, s0, l1_read_addr)
            noc_async_write_barrier()
            cb_pop_front(cb_out, onetile)
        return

    @reader_thread()
    def reader_binary_interleaved(
        cb_in0: CircularBuffer,
        cb_in1: CircularBuffer,
        src_addr0,
        src_addr1,
        num_tiles,
        start_id,
    ):
        onetile = 1

        tile_bytes0 = get_tile_size(cb_in0)
        tensor_accessor_args = TensorAccessorArgs(2, 0)
        s0 = TensorAccessor(tensor_accessor_args, src_addr0, tile_bytes0)

        tile_bytes1 = get_tile_size(cb_in1)
        tensor_accessor_args = TensorAccessorArgs(2, 0)
        s1 = TensorAccessor(tensor_accessor_args, src_addr1, tile_bytes1)

        end_id = start_id + num_tiles
        for i in range(start_id, end_id, onetile):
            cb_reserve_back(cb_in0, onetile)
            cb_reserve_back(cb_in1, onetile)

            src0_write_addr = get_write_ptr(cb_in0)
            src1_write_addr = get_write_ptr(cb_in1)

            noc_async_read_tile(i, s0, src0_write_addr)
            noc_async_read_tile(i, s1, src1_write_addr)

            noc_async_read_barrier()
            cb_push_back(cb_in0, onetile)
            cb_push_back(cb_in1, onetile)
        return

    def define_core_ranges(self, tensors, options):
        core_0 = ttnn.CoreCoord(0, 0)
        if self.max_core_ranges is None:
            core_1 = ttnn.CoreCoord(1, 1)
        else:
            core_1 = self.max_core_ranges
        return ttnn.CoreRangeSet([ttnn.CoreRange(core_0, core_1)])

    def invoke(
        self,  # super() has invoke signature (*tensors, **options)
        a_tensor,
        b_tensor,
        out_tensor,  # Tensor Definitions are positional args
    ):
        cb_in0 = self.create_cb(a_tensor, 0)
        cb_in1 = self.create_cb(b_tensor, 1)
        cb_out = self.create_cb(out_tensor, 2)
        start_id = 0

        self.set_tensor_accessor_config(a_tensor)

        num_tiles = ceil(
            max(map(lambda t: t.volume(), [a_tensor, b_tensor, out_tensor])) / 1024
        )

        num_cores = self.get_core_ranges().num_cores()
        num_tiles_per_core = num_tiles / num_cores

        if num_tiles_per_core % 1 != 0:
            # uneven distro of work, just break down and cry.
            raise Exception(":sad_hamster:")

        num_tiles_per_core = int(num_tiles_per_core)

        # Define the multicore runtime arguments
        start_id_multicore = []

        # Go row-wise
        bb = self.get_core_ranges().bounding_box()
        for i in range(bb.start.x, bb.end.x + 1):
            start_id_multicore.append([])
            for j in range(bb.start.y, bb.end.y + 1):
                # Set for each core
                start_id_multicore[-1].append([start_id])
                start_id += 1

        kernels = [
            self.create_kernel(
                VecAddMulticorePyKernelOp.add_multicore,
                cb_in0,
                cb_in1,
                cb_out,
                num_tiles_per_core,
                start_id_multicore,
            ),
            self.create_kernel(
                VecAddMulticorePyKernelOp.writer_multicore,
                cb_out,
                out_tensor.buffer_address(),
                num_tiles_per_core,
                start_id_multicore,
            ),
            self.create_kernel(
                VecAddMulticorePyKernelOp.reader_binary_interleaved,
                cb_in0,
                cb_in1,
                a_tensor.buffer_address(),
                b_tensor.buffer_address(),
                num_tiles_per_core,
                start_id_multicore,
            ),
        ]

        return self.create_program(kernels, [cb_in0, cb_in1, cb_out])


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.vecadd_op = VecAddMulticorePyKernelOp()

    def forward(self, A: ttnn.Tensor, B: ttnn.Tensor, device: ttnn.MeshDevice) -> ttnn.Tensor:
        shape = list(A.shape)

        output_tensor = ttnn.allocate_tensor_on_device(
            ttnn.Shape(shape),
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        output = self.vecadd_op(A, B, output_tensor)
        return output