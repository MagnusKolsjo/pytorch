# Owner(s): ["module: inductor"]
import ast
import textwrap

import torch
import torch.utils._pytree as pytree
from torch._inductor import compile_to_python
from torch._inductor.standalone_compile import NoRunnableInductorModuleError
from torch.fx.experimental.proxy_tensor import make_fx
from torch.nn.utils import stateless
from torch.testing._internal.common_utils import run_tests, TestCase


def _capture(m, x):
    """Trace ``m(x)`` into a flat-input ATen graph (params+buffers then ``x`` lifted to
    inputs), mirroring how ``torch.compiler.precompile`` feeds a post-AOTAutograd inner
    graph to ``torch._inductor.compile_to_python``. Tracing runs ``m(x)`` once, so pass
    a throwaway module."""
    pnames = [n for n, _ in m.named_parameters()]
    bnames = [n for n, _ in m.named_buffers()]
    pb = [p for _, p in m.named_parameters()] + [b for _, b in m.named_buffers()]
    k = len(pnames)

    def flat_fn(flat):
        params = dict(zip(pnames, flat[:k]))
        buffers = dict(zip(bnames, flat[k : k + len(bnames)]))
        with stateless._reparametrize_module(
            m, {**params, **buffers}, tie_weights=True
        ):
            out = m(flat[-1])
        return pytree.tree_flatten(out)[0]

    with torch.enable_grad():
        return make_fx(flat_fn)(pb + [x])


def _flat_inputs(m, x):
    return (
        [p for _, p in m.named_parameters()] + [b for _, b in m.named_buffers()] + [x]
    )


def _exec(src):
    ns = {"__name__": "_compiled"}
    exec(compile(src, "<compiled>", "exec"), ns)
    return ns["call"]


def _extract_call(src):
    """Return the dedented source of the module-level ``call`` entry point. The expect
    goldens lock this runtime entry point only: the rest of the emitted module (imports,
    the inert compile-time auto-tuning docstring, the ``# AOT ID`` global-counter
    comment) carries build- and ordering-dependent noise that should not be goldened."""
    mod = ast.parse(src)
    for node in ast.walk(mod):
        if isinstance(node, ast.FunctionDef) and node.name == "call":
            body = "\n".join(src.split("\n")[node.lineno - 1 : node.end_lineno])
            return textwrap.dedent(body)
    raise AssertionError("generated module has no module-level def call")


class TestInductorCompileToPythonCodegen(TestCase):
    # graph_partition is pinned off so the entry point is the stable top-level
    # ``def call(args)`` rather than the default ``Runner.call`` method wrapper. The
    # extern-kernel / CPU codegen the goldens lock is identical either way and does not
    # depend on whether torch was built with CUDA (these are CPU tensors).
    def _inner_call(self, m, x):
        gm = _capture(m, x)
        src, _cache = compile_to_python(
            gm, _flat_inputs(m, x), options={"graph_partition": False}
        )
        return src, _extract_call(src)

    def test_addmm_extern_kernel_codegen(self):
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        src, call_src = self._inner_call(m, x)
        self.assertExpectedInline(
            call_src,
            """\
def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    assert_size_stride(arg1_1, (3, ), (1, ), 'input')
    assert_size_stride(arg2_1, (5, 4), (4, 1), 'input')
    assert_size_stride(arg0_1, (3, 4), (4, 1), 'input')
    buf0 = empty_strided_cpu((5, 3), (3, 1), torch.float32)
    # Topologically Sorted Source Nodes: [t, addmm], Original ATen: [aten.t, aten.addmm]
    extern_kernels.addmm(arg1_1, arg2_1, reinterpret_tensor(arg0_1, (4, 3), (1, 4), 0), alpha=1, beta=1, out=buf0)
    del arg0_1
    del arg1_1
    del arg2_1
    return (buf0, )""",
        )
        with torch.no_grad():
            self.assertEqual(_exec(src)(_flat_inputs(m, x))[0], m(x))

    def test_matmul_extern_kernel_codegen(self):
        m = torch.nn.Linear(4, 3, bias=False).eval()
        x = torch.randn(5, 4)
        src, call_src = self._inner_call(m, x)
        self.assertExpectedInline(
            call_src,
            """\
def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg1_1, (5, 4), (4, 1), 'input')
    assert_size_stride(arg0_1, (3, 4), (4, 1), 'input')
    buf0 = empty_strided_cpu((5, 3), (3, 1), torch.float32)
    # Topologically Sorted Source Nodes: [t, mm], Original ATen: [aten.t, aten.mm]
    extern_kernels.mm(arg1_1, reinterpret_tensor(arg0_1, (4, 3), (1, 4), 0), out=buf0)
    del arg0_1
    del arg1_1
    return (buf0, )""",
        )
        with torch.no_grad():
            self.assertEqual(_exec(src)(_flat_inputs(m, x))[0], m(x))

    def test_addmm_relu_fused_pointwise_codegen(self):
        m = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU()).eval()
        x = torch.randn(5, 4)
        src, call_src = self._inner_call(m, x)
        self.assertExpectedInline(
            call_src,
            """\
def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    assert_size_stride(arg1_1, (3, ), (1, ), 'input')
    assert_size_stride(arg2_1, (5, 4), (4, 1), 'input')
    assert_size_stride(arg0_1, (3, 4), (4, 1), 'input')
    buf0 = empty_strided_cpu((5, 3), (3, 1), torch.float32)
    # Topologically Sorted Source Nodes: [t, addmm], Original ATen: [aten.t, aten.addmm]
    extern_kernels.addmm(arg1_1, arg2_1, reinterpret_tensor(arg0_1, (4, 3), (1, 4), 0), alpha=1, beta=1, out=buf0)
    del arg0_1
    del arg1_1
    del arg2_1
    buf1 = buf0; del buf0  # reuse
    cpp_fused_relu_0(buf1)
    return (buf1, )""",
        )
        with torch.no_grad():
            self.assertEqual(_exec(src)(_flat_inputs(m, x))[0], m(x))

    def test_benchmark_harness_suppressed(self):
        # #187858 pins benchmark_harness=False, so the emitted module is runnable rather
        # than an Inductor profiling harness: none of these debug entry points appear.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        src, _call_src = self._inner_call(m, x)
        for marker in (
            "benchmark_compiled_module",
            "def get_args(",
            "compiled_module_main",
            "print_performance",
        ):
            self.assertNotIn(marker, src)

    def test_no_runnable_module_for_no_compute(self):
        # A graph with no compute (returns its input unchanged) lowers to no module-level
        # ``call``, so there is nothing runnable to inline.
        m = torch.nn.Identity().eval()
        x = torch.randn(5, 4)
        gm = _capture(m, x)
        with self.assertRaises(NoRunnableInductorModuleError):
            compile_to_python(gm, _flat_inputs(m, x))


if __name__ == "__main__":
    run_tests()
