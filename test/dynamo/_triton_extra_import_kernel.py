# Owner(s): ["module: dynamo"]

import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def triton_kernel_with_extra_import(x_ptr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = libdevice.exp2(tl.load(x_ptr + offsets))
    tl.store(x_ptr + offsets, values)
