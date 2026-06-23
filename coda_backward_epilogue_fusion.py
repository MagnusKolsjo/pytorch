"""
[not for land] Prototype: backward epilogue fusion across autograd.Function nodes.

A UX prototype for CODA-style backward epilogue fusion ("CODA: Rewriting
Transformer Blocks as GEMM-Epilogue Programs", arXiv:2605.19269). The problem:
the backward of an autograd.Function is not self-contained -- for a chain
mm -> epilogue -> mm, the epilogue's backward wants to fuse as the epilogue of
the *next* matmul's backward (grad_a = (grad_c @ W^T) * f'(a) is a matmul with a
pointwise epilogue). A single Function can't express that, since its backward
runs inside one node.

The approach: the user writes a "1-to-many" op that decomposes into two autograd
nodes (a matmul node and an epilogue marker node), plus a fusion rule. Before
backward, apply_epilogue_fusion() walks the graph and arms the matmul node to
defer its activation gradient into the previous epilogue's backward (no graph
rewriting; deferral rides a placeholder grad along the existing edge).

The user's forward receives TWO ctxs and saves into each explicitly:

    def forward(main_ctx, epilogue_ctx, x, w):
        a = x @ w
        main_ctx.save_for_backward(x, w)      # tensors the matmul backward needs
        epilogue_ctx.save_for_backward(a)     # tensors the pointwise epilogue needs
        return relu(a)

The framework saves each set on the corresponding autograd node, so:
  * `main_backward(main_ctx, grad)` reads `main_ctx.saved_tensors`, honors
    `main_ctx.needs_input_grad`, and returns a grad per forward input. When this
    node defers, the framework sets needs_input_grad[activation] = False, so the
    user simply returns None for that slot (as for any not-needed grad) and the
    fused kernel produces it instead;
  * `epilogue_backward(epilogue_ctx, ...)` reads `epilogue_ctx.saved_tensors`;
  * lifetimes are scoped per node (the epilogue tensor `a` no longer lives on the
    main node), and the deferred producer's placeholder snapshot carries only the
    main set -- never the epilogue set.

The fused backward kernel receives BOTH ctxs, each exposing only its own subset:

    fused_impl(grad_producer_out, producer_ctx, consumer_ctx) -> grad_consumer_main_out
        producer_ctx.saved_tensors  # the producer op's main set (e.g. x, w)
        consumer_ctx.saved_tensors  # the consumer op's epilogue set (e.g. a)

Fusion rules are passed explicitly (no global registry) as a list of
(producer.main_backward, consumer.epilogue_backward, fused_impl) and applied in
a single step:

    plan = apply_epilogue_fusion(loss.grad_fn, rules, expect_num_fusions=2)
    loss.backward()

Each op below is self-contained with its own backwards.

Run:
    python coda_backward_epilogue_fusion.py
"""

from collections import deque
from dataclasses import dataclass

import torch
from torch.autograd import Function


# ---------------------------------------------------------------------------
# Instrumentation.
# ---------------------------------------------------------------------------
class _Log:
    def __init__(self):
        self.reset()

    def reset(self):
        self.c = dict(main_full=0, main_params_only=0, fused_impl=0, epilogue_unfused=0)

    def hit(self, k):
        self.c[k] += 1

    def __repr__(self):
        return repr(self.c)


LOG = _Log()


# Rules are passed explicitly to apply_epilogue_fusion (no global registry). A
# rule is (producer_main_backward, consumer_epilogue_backward, fused_impl), where
#   fused_impl(grad_producer_out, producer_ctx, consumer_ctx) -> grad_consumer_main_out
def _rule_key(producer_cls, consumer_cls):
    return (producer_cls.main_backward, consumer_cls.epilogue_backward)


# ---------------------------------------------------------------------------
# Placeholder grad that rides the existing backward edge (framework-internal).
# ---------------------------------------------------------------------------
class DeferredGradTensor(torch.Tensor):
    """Carries ONLY the producer's incoming grad across the backward edge to the
    consumer epilogue node. The fused impl and both saved ctxs are stamped on the
    consumer at plan time (forward has already run, so they are available), so the
    runtime grad is the only thing that has to thread through here."""

    @staticmethod
    def __new__(cls, shape, dtype, device, real_grad):
        r = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
            cls, shape, dtype=dtype, device=device, requires_grad=False,
        )
        r._real_grad = real_grad
        return r

    __torch_function__ = torch._C._disabled_torch_function_impl  # type: ignore[attr-defined]

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        # The engine routes this placeholder along the backward edge by reading its
        # metadata in C++ -- it never dispatches a real op on it. Reaching here
        # means it leaked into computation, where using it would be silently wrong
        # (its shape is the deferred grad_input, not _real_grad's grad_output).
        raise RuntimeError(
            f"DeferredGradTensor is a metadata-only placeholder for a deferred "
            f"grad_input and must not be used in a real op (got {func}). The "
            f"consumer epilogue node should detect it and unwrap `._real_grad`; "
            f"reaching __torch_dispatch__ means the placeholder leaked into "
            f"computation."
        )


class _WrappedCtx:
    """Forwards every attribute to an inner ctx except the overridden ones, so a
    backward can be handed a ctx that differs in just one field. Mirrors the
    WrappedCtx pattern in torch/_functorch/autograd_function.py."""

    _reserved = ("_inner_ctx",)

    def __init__(self, ctx):
        self._inner_ctx = ctx

    def __getattr__(self, name):
        return getattr(self._inner_ctx, name)

    def __setattr__(self, name, value):
        if name in type(self)._reserved:
            self.__dict__[name] = value
        else:
            setattr(self._inner_ctx, name, value)


class _SavedSnapshot:
    """A standalone holder exposing `saved_tensors`: the producer's saved set,
    captured at plan time because its real saved_tensors are freed before the
    consumer's fused kernel runs. No pass-through wrapper is needed -- the fused
    kernel reads only `saved_tensors` from the producer ctx."""

    def __init__(self, saved):
        self.saved_tensors = saved


class _CtxWithNeedsInputGrad(_WrappedCtx):
    """Override needs_input_grad -- to mark the deferred activation grad as not
    needed (the fused kernel produces it) without mutating the real ctx."""

    _reserved = ("_needs", *_WrappedCtx._reserved)

    def __init__(self, ctx, needs_input_grad):
        super().__init__(ctx)
        self._needs = needs_input_grad

    @property
    def needs_input_grad(self):
        return self._needs


class _StagingCtx:
    """Forward-time stand-in for a node ctx. The user's forward runs against two of
    these (main + epilogue); each collects its saved tensors (and, for the main
    ctx, the intermediate metadata) which the per-call nodes capture by closure."""

    def __init__(self):
        self.saved = ()
        self.output_meta = None

    def save_for_backward(self, *tensors):
        self.saved = tensors

    def set_output_meta(self, like):
        # Declare the metadata of this node's output -- for the main ctx this is the
        # intermediate that flows to the epilogue (the GEMM output before the
        # epilogue). Only shape/dtype/device are used; the value is irrelevant.
        self.output_meta = (like.shape, like.dtype, like.device)


# The role of an autograd node is encoded in its type name: the backward node for
# a custom Function `X` is named `XBackward`. The per-call nodes below always use
# these inner class names, so name matching identifies the role uniformly.
_MAIN_BACKWARD = "_MainNodeBackward"
_EPILOGUE_BACKWARD = "_EpilogueNodeBackward"


def _is_main(node):
    return type(node).__name__ == _MAIN_BACKWARD


def _is_epilogue(node):
    return type(node).__name__ == _EPILOGUE_BACKWARD


# ---------------------------------------------------------------------------
# User-facing base class.
# ---------------------------------------------------------------------------
class FusibleFunction:
    @classmethod
    def apply(cls, *inputs):
        # Run the user's forward once, outside both nodes, against two symmetric
        # staging ctxs.
        main_staging = _StagingCtx()
        epilogue_staging = _StagingCtx()
        with torch.no_grad():
            out = cls.forward(main_staging, epilogue_staging, *inputs)
        # The intermediate (GEMM output) between the two nodes may differ in shape
        # from `out` (e.g. SwiGLU is dim-reducing, norm involves a reduction), so the
        # user must declare its metadata; we cannot infer it from `out`.
        if main_staging.output_meta is None:
            raise RuntimeError(
                f"{cls.__name__}.forward must call main_ctx.set_output_meta(...) to "
                f"declare the intermediate (main output) metadata"
            )
        # The per-call node classes form a Function<->Backward reference cycle, so
        # anything their forwards close over is only reclaimed by the gc, not by
        # refcounting. Hold the forward tensors in a box the forwards read, then
        # clear it right after the forwards run (below) so save_for_backward becomes
        # their sole owner and they are released promptly during backward.
        box = {
            "out": out,
            "main_saved": main_staging.saved,
            "epilogue_saved": epilogue_staging.saved,
            "meta": main_staging.output_meta,
        }

        class _MainNode(Function):
            @staticmethod
            def forward(ctx, *inps):
                ctx.cls = cls
                ctx.defer_input_idx = None  # set by the plan; None means do not defer
                ctx.in_metas = tuple((t.shape, t.dtype, t.device) for t in inps)
                ctx.save_for_backward(*box["main_saved"])
                # Emit a phantom carrier of the intermediate's metadata: a fresh
                # tensor (so it gets this node's grad_fn, with next edges = inputs).
                # Its value is unused -- the epilogue node returns the real output.
                shape, dtype, device = box["meta"]
                return torch.empty(shape, dtype=dtype, device=device)

            @staticmethod
            def backward(ctx, grad_main_out):
                # ctx already exposes saved_tensors and the real needs_input_grad;
                # we only override the latter when deferring (to skip the dx GEMM).
                k = ctx.defer_input_idx
                if k is not None:
                    LOG.hit("main_params_only")
                    needs = list(ctx.needs_input_grad)
                    needs[k] = False  # the activation grad is produced by the fused kernel
                    bw_ctx = _CtxWithNeedsInputGrad(ctx, tuple(needs))
                    grads = list(cls.main_backward(bw_ctx, grad_main_out))
                    shape, dtype, device = ctx.in_metas[k]
                    grads[k] = DeferredGradTensor(shape, dtype, device, grad_main_out)
                    return tuple(grads)
                LOG.hit("main_full")
                return tuple(cls.main_backward(ctx, grad_main_out))

        class _EpilogueNode(Function):
            @staticmethod
            def forward(ctx, main_out):
                ctx.cls = cls
                ctx.save_for_backward(*box["epilogue_saved"])
                ctx.fused_impl = None         # set by the plan when this node fuses
                ctx.producer_main_ctx = None  # producer's saved set, set by the plan
                # Attach the real (possibly differently-shaped) final output to the
                # graph here: main_out (phantom) requires grad, so this fresh view of
                # the no_grad `out` gets this node's grad_fn.
                return box["out"].view_as(box["out"])

            @staticmethod
            def backward(ctx, grad_out):
                # ctx is the consumer ctx (its own saved_tensors); the producer's
                # saved set comes from the snapshot stamped by the plan.
                if isinstance(grad_out, DeferredGradTensor):
                    LOG.hit("fused_impl")
                    grad_main_out = ctx.fused_impl(
                        grad_out._real_grad, ctx.producer_main_ctx, ctx
                    )
                    ctx.producer_main_ctx = None  # release the producer snapshot
                else:
                    LOG.hit("epilogue_unfused")
                    grad_main_out = cls.epilogue_backward(ctx, grad_out)
                return (grad_main_out,)

        main_out = _MainNode.apply(*inputs)
        result = _EpilogueNode.apply(main_out)
        box.clear()  # forwards have run; drop refs so only save_for_backward holds them
        return result

    @staticmethod
    def forward(main_ctx, epilogue_ctx, *inputs):
        raise NotImplementedError

    @staticmethod
    def main_backward(main_ctx, grad_main_out):
        raise NotImplementedError

    @staticmethod
    def epilogue_backward(epilogue_ctx, grad_out):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Fusion plan via autograd-graph traversal, gated by the registry.
# ---------------------------------------------------------------------------
NO_RULE = "no rule registered"  # structural candidate with no registered rule


@dataclass
class _PlannedPair:
    producer: object
    consumer: object
    unfused_reason: str | None  # None when fusion is armed; else why it bailed

    @property
    def fused(self):
        return self.unfused_reason is None

    def _label(self):
        return (f"{self.producer.cls.__name__}.main_backward -> "
                f"{self.consumer.cls.__name__}.epilogue_backward")


class FusionPlan:
    """Result of apply_epilogue_fusion; the categorized pairs are internal.

    Public API is assert_num_fusions() (and repr for display).
    """

    def __init__(self, fused, missing_rules, bailed):
        self._fused = fused                  # pairs armed for fusion
        self._missing_rules = missing_rules  # candidates with no rule given
        self._bailed = bailed                # candidates skipped (e.g. branching)

    def assert_num_fusions(self, expected):
        got = len(self._fused)
        if got != expected:
            lines = [f"expected {expected} backward fusions, planned {got}"]
            if self._missing_rules:
                lines.append(
                    f"{len(self._missing_rules)} fusible main -> epilogue "
                    f"adjacency(ies) have no registered rule:"
                )
                lines += [f"  - {p._label()}" for p in self._missing_rules]
                lines.append(
                    "Did you forget to pass a rule for these to "
                    "apply_epilogue_fusion(rules=...)?"
                )
            other = [p.unfused_reason for p in self._bailed]
            if other:
                lines.append(f"other bailed pairs: {other}")
            raise AssertionError("\n".join(lines))
        return self

    def __repr__(self):
        all_pairs = self._fused + self._missing_rules + self._bailed
        if not all_pairs:
            return "FusionPlan(no candidates)"
        return "FusionPlan(\n  " + "\n  ".join(
            f"{p.producer.cls.__name__}.main -> "
            f"{p.consumer.cls.__name__}.epilogue: "
            f"{'FUSE' if p.fused else 'bail:' + p.unfused_reason}"
            for p in all_pairs
        ) + "\n)"


def apply_epilogue_fusion(root, rules, expect_num_fusions=None):
    r"""Plan backward epilogue fusion over the autograd graph reachable from ``root``.

    Walks the backward graph and, for every adjacency where a producer op's
    ``main_backward`` (a matmul backward) feeds a consumer op's
    ``epilogue_backward`` (a pointwise backward), arms the producer to defer its
    activation gradient so the consumer can run a single fused kernel instead --
    the backward analogue of matmul epilogue fusion. The graph is **not** rewritten;
    deferral is armed in place on the existing nodes, so this must be called after
    forward and before :meth:`backward`, every iteration.

    .. note::
        Two things each op's implementation must get right:

        - ``main_ctx.set_output_meta(a)`` is **required** in ``forward`` -- it declares
          the shape/dtype/device of the intermediate (the GEMM output ``a``) that
          flows between the two nodes, which cannot be inferred from the final output
          (e.g. SwiGLU is dim-reducing). ``apply`` raises if it is missing.
        - ``main_backward`` must guard each gradient with ``main_ctx.needs_input_grad``
          and compute the weight gradient (dW) unconditionally first. When this op's
          activation gradient is deferred into a fused kernel, the framework hands
          ``main_backward`` a ctx whose ``needs_input_grad`` is ``False`` at the
          activation slot, so it computes only dW and returns ``None`` for dx (which
          the fused kernel produces).

    Args:
        root (Tensor or Node): the loss tensor (or its ``grad_fn``) to traverse
            back from.
        rules (list): fusion rules, each a tuple
            ``(producer_cls.main_backward, consumer_cls.epilogue_backward, fused_impl)``.
            ``fused_impl(grad_producer_out, producer_ctx, consumer_ctx)`` returns the
            grad of the consumer's main output, computing the producer's deferred
            ``grad_input`` GEMM with the consumer's epilogue fused on. Only registered
            pairs fuse.
        expect_num_fusions (int, optional): if given, assert exactly this many
            fusions were planned, raising a diagnostic that names any fusible
            adjacency lacking a rule. Default: ``None``.

    Returns:
        FusionPlan: the plan; call ``plan.assert_num_fusions(n)`` to check coverage.

    Example::

        >>> import torch
        >>> class MMRelu(FusibleFunction):
        ...     @staticmethod
        ...     def forward(main_ctx, epilogue_ctx, x, w):
        ...         a = x @ w
        ...         main_ctx.save_for_backward(x, w)
        ...         main_ctx.set_output_meta(a)        # REQUIRED: boundary meta (else apply raises)
        ...         epilogue_ctx.save_for_backward(a)
        ...         return torch.relu(a)               # single fused fwd kernel
        ...     @staticmethod
        ...     def main_backward(main_ctx, grad_a):   # the matmul backward
        ...         x, w = main_ctx.saved_tensors
        ...         # dW first (always local); dx (input 0) is guarded by
        ...         # needs_input_grad and skipped when deferred into the fused kernel.
        ...         gw = x.T @ grad_a if main_ctx.needs_input_grad[1] else None
        ...         gx = grad_a @ w.T if main_ctx.needs_input_grad[0] else None
        ...         return gx, gw
        ...     @staticmethod
        ...     def epilogue_backward(epilogue_ctx, grad_out):   # the pointwise backward
        ...         (a,) = epilogue_ctx.saved_tensors
        ...         return grad_out * (a > 0).to(a.dtype)
        >>>
        >>> def mm_bw_relu_bw_fused(grad_producer_out, producer_ctx, consumer_ctx):
        ...     _x, w = producer_ctx.saved_tensors     # producer's matmul weight
        ...     (a,) = consumer_ctx.saved_tensors      # consumer's preactivation
        ...     return (grad_producer_out @ w.T) * (a > 0).to(a.dtype)
        >>>
        >>> rules = [(MMRelu.main_backward, MMRelu.epilogue_backward, mm_bw_relu_bw_fused)]
        >>> x = torch.randn(4, 6, requires_grad=True)
        >>> w1 = torch.randn(6, 6, requires_grad=True)
        >>> w2 = torch.randn(6, 6, requires_grad=True)
        >>> out = MMRelu.apply(MMRelu.apply(x, w1), w2)      # mm1 -> relu -> mm2 -> relu
        >>> loss = out.sum()
        >>> plan = apply_epilogue_fusion(loss, rules, expect_num_fusions=1)
        >>> loss.backward()   # relu1's backward runs fused into mm2's grad_input GEMM
    """
    if isinstance(root, torch.Tensor):
        root = root.grad_fn
    rule_map = {(p, c): impl for p, c, impl in rules}

    # BFS over every backward node reachable from root (from torch.autograd.graph's
    # iter_graph), plus an in-degree tally for the branching bail-out below.
    def iter_graph(roots):
        seen, q = set(), deque()
        for node in roots:
            if node is not None:
                seen.add(node)
                q.append(node)
        while q:
            node = q.popleft()
            for fn, _ in node.next_functions:
                if fn is None or fn in seen:
                    continue
                seen.add(fn)
                q.append(fn)
            yield node

    nodes = list(iter_graph([root]))
    in_degree = {}
    for n in nodes:
        for child, _ in n.next_functions:
            if child is not None:
                in_degree[child] = in_degree.get(child, 0) + 1

    fused, missing_rules, bailed = [], [], []
    for n in nodes:
        if not _is_main(n):
            continue
        # The fusible edge is the input whose producer is an epilogue node; it may
        # sit at any position (the rest are parameters). Exactly one is supported:
        # deferral emits a single placeholder, so multiple fusible inputs cannot be
        # represented by the single-branch model.
        candidates = [(i, c) for i, (c, _) in enumerate(n.next_functions)
                      if c is not None and _is_epilogue(c)]
        if len(candidates) > 1:
            raise RuntimeError(
                f"{type(n).__name__}: {len(candidates)} inputs feed epilogue nodes; "
                f"deferring more than one grad_input is not supported"
            )
        if not candidates:
            continue
        idx, consumer = candidates[0]
        impl = rule_map.get(_rule_key(n.cls, consumer.cls))
        if impl is None:
            # Structural candidate (a main feeds an epilogue) but no rule given.
            missing_rules.append(_PlannedPair(n, consumer, NO_RULE))
            continue
        if in_degree.get(consumer, 0) > 1:
            bailed.append(_PlannedPair(n, consumer, "branching consumer"))
            continue
        n.defer_input_idx = idx  # deferring: placeholder goes on this edge in the backward
        # Stamp everything the consumer's fused backward needs now, while the
        # producer's saved tensors are still alive (forward has run, backward has
        # not). Only the producer's runtime grad threads through the placeholder.
        consumer.fused_impl = impl
        consumer.producer_main_ctx = _SavedSnapshot(tuple(n.saved_tensors))
        fused.append(_PlannedPair(n, consumer, None))  # None == fusion armed

    plan = FusionPlan(fused, missing_rules, bailed)
    if expect_num_fusions is not None:
        plan.assert_num_fusions(expect_num_fusions)
    return plan


# ===========================================================================
# Example user code: two self-contained ops, each with its own backwards.
# ===========================================================================
class MMRelu(FusibleFunction):
    @staticmethod
    def forward(main_ctx, epilogue_ctx, x, w):
        a = x @ w
        main_ctx.save_for_backward(x, w)
        main_ctx.set_output_meta(a)  # REQUIRED (else apply raises): boundary metadata
        epilogue_ctx.save_for_backward(a)
        return torch.relu(a)

    @staticmethod
    def main_backward(main_ctx, grad_main_out):
        x, w = main_ctx.saved_tensors
        # dW is always local; compute it first. dx (input 0) is guarded by
        # needs_input_grad and skipped when this op's activation grad is deferred
        # into the fused kernel (the framework sets needs_input_grad[0] = False).
        grad_w = x.transpose(-1, -2) @ grad_main_out if main_ctx.needs_input_grad[1] else None
        grad_x = grad_main_out @ w.transpose(-1, -2) if main_ctx.needs_input_grad[0] else None
        return grad_x, grad_w

    @staticmethod
    def epilogue_backward(epilogue_ctx, grad_out):
        (a,) = epilogue_ctx.saved_tensors
        return grad_out * (a > 0).to(a.dtype)


class MMTanh(FusibleFunction):
    @staticmethod
    def forward(main_ctx, epilogue_ctx, x, w):
        a = x @ w
        main_ctx.save_for_backward(x, w)
        main_ctx.set_output_meta(a)  # REQUIRED (else apply raises): boundary metadata
        epilogue_ctx.save_for_backward(a)
        return torch.tanh(a)

    @staticmethod
    def main_backward(main_ctx, grad_main_out):
        x, w = main_ctx.saved_tensors
        # dW first (always local); dx is guarded and skipped when deferred.
        grad_w = x.transpose(-1, -2) @ grad_main_out if main_ctx.needs_input_grad[1] else None
        grad_x = grad_main_out @ w.transpose(-1, -2) if main_ctx.needs_input_grad[0] else None
        return grad_x, grad_w

    @staticmethod
    def epilogue_backward(epilogue_ctx, grad_out):
        (a,) = epilogue_ctx.saved_tensors
        return grad_out * (1 - torch.tanh(a) ** 2)


def mm_relu_fused_backward(grad_producer_out, producer_main_ctx, consumer_epilogue_ctx):
    _x_p, w_p = producer_main_ctx.saved_tensors
    (a_c,) = consumer_epilogue_ctx.saved_tensors
    grad_main_input = grad_producer_out @ w_p.transpose(-1, -2)
    return grad_main_input * (a_c > 0).to(a_c.dtype)


def mm_tanh_fused_backward(grad_producer_out, producer_main_ctx, consumer_epilogue_ctx):
    _x_p, w_p = producer_main_ctx.saved_tensors
    (a_c,) = consumer_epilogue_ctx.saved_tensors
    grad_main_input = grad_producer_out @ w_p.transpose(-1, -2)
    return grad_main_input * (1 - torch.tanh(a_c) ** 2)


# One rule per (producer.main_backward, consumer.epilogue_backward) pair, passed
# explicitly to apply_epilogue_fusion. The fused impl depends only on the
# consumer's epilogue, so both producers reuse the same impl.
RULES = [
    (MMRelu.main_backward, MMRelu.epilogue_backward, mm_relu_fused_backward),
    (MMTanh.main_backward, MMRelu.epilogue_backward, mm_relu_fused_backward),
    (MMRelu.main_backward, MMTanh.epilogue_backward, mm_tanh_fused_backward),
    (MMTanh.main_backward, MMTanh.epilogue_backward, mm_tanh_fused_backward),
]


# ===========================================================================
# Verification.
# ===========================================================================
def _check(name, ins, refs, atol=1e-9):
    ok = True
    print(f"=== gradient check: {name} ===")
    labels = ["x"] + [f"w{i}" for i in range(1, len(ins))]
    for nm, t, r in zip(labels, ins, refs):
        err = (t.grad - r).abs().max().item()
        good = torch.allclose(t.grad, r, atol=atol)
        ok &= good
        print(f"  grad_{nm}: max_abs_err={err:.2e} ok={good}")
    assert ok, f"{name}: gradients do not match reference"


def scenario_mixed_chain():
    """x -> MMTanh -> MMRelu -> MMTanh -> sum.

    Fusions:
      ep1 (MMTanh) <- main2 (MMRelu): key (MMRelu.main, MMTanh.epilogue) -> tanh rule
      ep2 (MMRelu) <- main3 (MMTanh): key (MMTanh.main, MMRelu.epilogue) -> relu rule
    """
    print("\n########## scenario: mixed chain ##########")
    torch.manual_seed(0)
    B, K = 4, 6

    def make():
        return [torch.randn(B if i == 0 else K, K, dtype=torch.double,
                            requires_grad=True) for i in range(4)]

    ref = make()
    x, w1, w2, w3 = ref
    torch.tanh(torch.relu(torch.tanh(x @ w1) @ w2) @ w3).sum().backward()
    refs = [t.grad.clone() for t in ref]

    ins = make()
    for t, r in zip(ins, ref):
        t.data.copy_(r.data)
    x, w1, w2, w3 = ins

    LOG.reset()
    loss = MMTanh.apply(MMRelu.apply(MMTanh.apply(x, w1), w2), w3).sum()
    plan = apply_epilogue_fusion(loss.grad_fn, RULES, expect_num_fusions=2)
    print(plan)
    loss.backward()

    _check("mixed", ins, refs)
    print("kernel paths:", LOG)
    assert LOG.c["fused_impl"] == 2
    assert LOG.c["epilogue_unfused"] == 1
    assert LOG.c["main_params_only"] == 2
    assert LOG.c["main_full"] == 1
    print("PASS: both epilogues fused across the mixed chain.")


if __name__ == "__main__":
    scenario_mixed_chain()
    print("\nALL SCENARIOS PASSED")
