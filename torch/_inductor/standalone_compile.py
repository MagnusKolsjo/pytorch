from __future__ import annotations

import ast
import contextlib
import copy
import logging
import os
import pickle
import shutil
import threading
from abc import ABC, abstractmethod
from contextlib import AbstractContextManager, nullcontext
from typing import Any, Literal, TYPE_CHECKING


DynamicShapesType = Literal["from_example_inputs", "from_tracing_context", "from_graph"]

import torch.fx
from torch._dynamo.aot_compile_types import BundledAOTAutogradSerializableCallable
from torch._dynamo.utils import dynamo_timed
from torch._inductor.cpp_builder import normalize_path_separator
from torch._inductor.cudagraph_utils import BoxedDeviceIndex
from torch._inductor.runtime.cache_dir_utils import temporary_cache_dir
from torch._inductor.utils import BoxedBool, InputType
from torch._subclasses import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.fx.graph_module import _share_torchbind_and_process_group_on_deepcopy

from . import config
from ._functionalize_collectives import (
    _functionalize_inplace_collectives,
    _unbox_process_group_torchbinds,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from torch.compiler._cache import CacheInfo
    from torch.fx import GraphModule


log = logging.getLogger(__name__)


# Serializes ``compile_to_python``. That function mutates the process-global
# ``GraphLowering.save_output_code`` hook and enters
# ``CacheArtifactManager.with_fresh_cache`` (process-global cache state), so two
# threads running it concurrently would clobber each other's hook and cache.
# ``compile_to_python`` is in ``torch._inductor.__all__``, so direct callers must
# be protected here rather than relying on a higher-layer lock. The AOT-level
# ``compile_to_python`` (torch._functorch.aot_autograd) holds its own distinct
# ``_COMPILE_LOCK`` and only ever calls into this layer while holding it, so the
# acquisition order is always AOT-then-inductor and these two locks never deadlock
# (they are different locks and there is no reverse nesting).
#
# This is an RLock, not a plain Lock, because the inductor compile that runs while
# the lock is held can itself re-enter ``compile_to_python`` on the SAME thread: an
# inductor pass or custom backend may compile a subgraph to python during lowering.
# Same-thread re-entry is safe here because the inner call saves and restores the
# (now ``capture_hook``) ambient hook around its own region and uses its own fresh
# cache scope, so the nested capture is self-contained and the outer hook is intact
# on return; a plain Lock would self-deadlock on that re-entry instead.
_COMPILE_TO_PYTHON_LOCK = threading.RLock()


class CompiledArtifact(ABC):
    """
    CompiledArtifact class represents the inductor cache artifacts that
    can be invoked in order to avoid repeated compilation.

    CompiledArtifact can be obtained by calling standalone_compile(gm, example_inputs)
    to create a fresh CompiledArtifact from a GraphModule and example inputs.

    Later this CompiledArtifact can be saved to disk, either as a binary or unpacked
    into the provided folder via the CompiledArtifact.save function.

    CompiledArtifact.load provides a way to create a CompiledArtifact from the
    binary or unpacked data.

    Finally, the CompiledArtifact can be invoked via the __call__ method
    to execute the cached artifact.
    """

    def __init__(
        self,
        compiled_fn: Callable[..., Any],
        artifacts: tuple[bytes, CacheInfo] | None,
    ):
        self._compiled_fn = compiled_fn
        self._artifacts = artifacts

    @abstractmethod
    def __call__(self, *args: Any) -> Any: ...

    @abstractmethod
    def save(
        self, *, path: str, format: Literal["binary", "unpacked"] = "binary"
    ) -> None: ...

    @staticmethod
    def load(
        *, path: str, format: Literal["binary", "unpacked"] = "binary"
    ) -> CompiledArtifact:
        if format == "unpacked":
            # If format is unpacked, it must be a CacheCompiledArtifact
            return CacheCompiledArtifact.load(path=path, format=format)

        if format != "binary":
            raise AssertionError(f"expected format == 'binary', got {format}")
        with open(path, "rb") as file:
            from torch.utils._appending_byte_serializer import BytesReader

            from .codecache import torch_key

            result_bytes = file.read()
            reader = BytesReader(result_bytes)
            header = reader.read_bytes()
            if header == AOTCompiledArtifact.AOT_HEADER:
                if reader.read_bytes() != torch_key():
                    raise AssertionError("torch_key mismatch in serialized artifact")
                artifact = reader.read_bytes()
                if not reader.is_finished():
                    raise AssertionError("expected reader to be finished")
                return AOTCompiledArtifact.deserialize(artifact)
            # Otherwise, it's in the CacheCompiledArtifact format
            elif header == CacheCompiledArtifact.CACHE_HEADER:
                if reader.read_bytes() != torch_key():
                    raise AssertionError("torch_key mismatch in serialized artifact")
                key = reader.read_str()
                artifact_bytes = reader.read_bytes()
                if not reader.is_finished():
                    raise AssertionError("expected reader to be finished")
                torch.compiler.load_cache_artifacts(artifact_bytes)
                return CacheCompiledArtifact._load_impl(nullcontext(), key)
            else:
                raise RuntimeError(
                    "Invalid header, expected CacheCompiledArtifact or AOTCompiledArtifact, got: "
                    + header.decode("utf-8")
                )


class CacheCompiledArtifact(CompiledArtifact):
    """
    CompiledArtifact that depends on torch.compiler.save_cache_artifacts
    """

    CACHE_HEADER = bytes("CacheCompiledArtifact", "utf-8")

    def __init__(
        self,
        compiled_fn: Callable[..., Any],
        artifacts: tuple[bytes, CacheInfo] | None,
    ):
        self._compiled_fn = compiled_fn
        self._artifacts = artifacts

    def __call__(self, *args: Any) -> Any:
        return self._compiled_fn(*args)

    def is_saveable(self) -> bool:
        if self._artifacts is None:
            return False
        _, cache_info = self._artifacts
        # 0 means nothing was saved
        # >1 means multiple artifacts were saved, which is concerning
        # (we only expect one)
        return len(cache_info.aot_autograd_artifacts) == 1

    def _to_binary_bytes(self) -> bytes:
        """Serialize this artifact to the in-memory ``binary`` byte format.

        Produces exactly the bytes that ``save(format="binary")`` writes to disk, so
        callers that only need the bytes (rather than a file on disk) can reuse this
        without going through ``save``. The byte layout is the one
        ``load(format="binary")`` reads back: header, ``torch_key``, the autograd-cache
        ``key`` string, then the opaque ``artifact_bytes``.
        """
        if self._artifacts is None:
            raise RuntimeError(
                "CompiledArtifact.save failed to save since there's no artifact to save"
            )
        artifact_bytes, cache_info = self._artifacts
        if len(cache_info.aot_autograd_artifacts) == 0:
            raise RuntimeError(
                f"CompiledArtifact.save failed to save due to no aot_autograd artifacts. "
                f"This likely means there was something that was not serializable in the "
                f"graph passed to standalone_compile. This can generally be fixed by "
                f"ensuring that your model only uses constructs that are serializable. "
                f"{cache_info}"
            )
        if len(cache_info.aot_autograd_artifacts) > 1:
            raise AssertionError(
                f"CompiledArtifact.save failed to save because there was more than one "
                f"artifact but we only expected one. {cache_info}"
            )
        key = cache_info.aot_autograd_artifacts[0]

        from torch.utils._appending_byte_serializer import BytesWriter

        from .codecache import torch_key

        writer = BytesWriter()
        writer.write_bytes(CacheCompiledArtifact.CACHE_HEADER)
        writer.write_bytes(torch_key())
        writer.write_str(key)
        writer.write_bytes(artifact_bytes)
        return writer.to_bytes()

    def save(
        self, *, path: str, format: Literal["binary", "unpacked"] = "binary"
    ) -> None:
        with dynamo_timed("CompiledArtifact.save"):
            if format == "binary":
                # can't assert that it is a file since it might not exist yet
                if os.path.isdir(path):
                    raise AssertionError(f"expected path to not be a dir: {path}")

                from torch._inductor.codecache import write_atomic

                # ``_to_binary_bytes`` is the single source of the None / empty /
                # multiple aot_autograd_artifacts validation, so the binary branch
                # does not repeat those checks here.
                write_atomic(path, self._to_binary_bytes())
            else:
                if format != "unpacked":
                    raise AssertionError(f"expected format == 'unpacked', got {format}")
                if self._artifacts is None:
                    raise RuntimeError(
                        "CompiledArtifact.save failed to save since there's no artifact to save"
                    )
                artifact_bytes, cache_info = self._artifacts
                if len(cache_info.aot_autograd_artifacts) == 0:
                    raise RuntimeError(
                        f"CompiledArtifact.save failed to save due to no aot_autograd artifacts. "
                        f"This likely means there was something that was not serializable in the "
                        f"graph passed to standalone_compile. This can generally be fixed by "
                        f"ensuring that your model only uses constructs that are serializable. "
                        f"{cache_info}"
                    )
                if len(cache_info.aot_autograd_artifacts) > 1:
                    raise AssertionError(
                        f"CompiledArtifact.save failed to save because there was more than one "
                        f"artifact but we only expected one. {cache_info}"
                    )
                if os.path.exists(path):
                    if not os.path.isdir(path):
                        raise AssertionError(f"expected path to be a dir: {path}")
                    shutil.rmtree(path, ignore_errors=True)

                from .codecache import FxGraphCache

                with temporary_cache_dir(path):
                    # This function unpacks the cache artifacts to disk
                    loaded_cache_info = torch.compiler.load_cache_artifacts(
                        artifact_bytes
                    )
                    if loaded_cache_info is None:
                        raise AssertionError(
                            "expected loaded_cache_info to not be None"
                        )
                    # Now write all the output_code artifacts to disk so that
                    # they can be inspected and modified
                    for key in loaded_cache_info.inductor_artifacts:
                        subdir = FxGraphCache._get_tmp_dir_for_key(key)
                        if not os.path.exists(subdir):
                            raise AssertionError(f"expected subdir to exist: {subdir}")
                        for path in sorted(os.listdir(subdir)):
                            with open(os.path.join(subdir, path), "rb") as f:
                                graph = pickle.load(f)
                            output_file = graph.write_to_disk()
                            log.info("Output code written to: %s", output_file)

    @staticmethod
    def _load_impl(
        cache_dir_ctx: AbstractContextManager[Any], key: str
    ) -> CompiledArtifact:
        with (
            cache_dir_ctx,
            config.patch(unsafe_skip_cache_dynamic_shape_guards=True),
        ):
            with torch._functorch.config.patch(strict_autograd_cache=True):
                from torch._functorch._aot_autograd.autograd_cache import (
                    AOTAutogradCache,
                )

                result = AOTAutogradCache._lookup(
                    key,
                    local=True,
                    remote=False,
                    args=[],
                    cache_info={},
                    aot_config=None,
                )

            if result is None:
                raise AssertionError(
                    "expected AOTAutogradCache lookup result to not be None"
                )
            (entry, _) = result

            from .compile_fx import _CompileFxKwargs

            fx_config = _CompileFxKwargs(
                cudagraphs=BoxedBool(False),
                boxed_forward_device_index=BoxedDeviceIndex(0),
            )

            context = torch._guards.TracingContext(FakeTensorMode(shape_env=ShapeEnv()))
            with torch._guards.tracing(context):
                compiled_fn = entry.wrap_post_compile(
                    [], entry.sanitized_aot_config, fx_config
                )
        return CacheCompiledArtifact(lambda *args: compiled_fn(list(args)), None)

    @staticmethod
    def _prepare_load(
        *, path: str, format: Literal["binary", "unpacked"] = "binary"
    ) -> tuple[str, AbstractContextManager[Any]]:
        """
        Do format specific prep and loads, return a context manager and key
        """
        path = normalize_path_separator(path)
        with dynamo_timed("CompiledArtifact.load"):
            if format == "binary":
                # can't assert that it is a file since it might not exist yet
                if os.path.isdir(path):
                    raise AssertionError(f"expected path to not be a dir: {path}")
                with open(path, "rb") as file:
                    artifacts = file.read()
                from torch.utils._appending_byte_serializer import BytesReader

                from .codecache import torch_key

                reader = BytesReader(artifacts)
                if reader.read_bytes() != torch_key():
                    raise AssertionError("torch_key mismatch in serialized artifact")
                key = reader.read_str()
                artifact_bytes = reader.read_bytes()
                if not reader.is_finished():
                    raise AssertionError("expected reader to be finished")

                torch.compiler.load_cache_artifacts(artifact_bytes)
                return key, nullcontext()
            else:
                if format != "unpacked":
                    raise AssertionError(f"expected format == 'unpacked', got {format}")
                if not os.path.isdir(path):
                    raise AssertionError(f"expected path to be a dir: {path}")
                autograd_cache_dir = os.path.join(path, "aotautograd")
                if not os.path.isdir(autograd_cache_dir):
                    raise AssertionError(
                        f"expected autograd_cache_dir to be a dir: {autograd_cache_dir}"
                    )
                files = list(os.listdir(autograd_cache_dir))
                if len(files) != 1:
                    raise AssertionError(f"expected exactly 1 file, got {len(files)}")
                key = files[0]
                cache_dir_ctx = temporary_cache_dir(path)
                return key, cache_dir_ctx

    @staticmethod
    def load(
        *, path: str, format: Literal["binary", "unpacked"] = "binary"
    ) -> CompiledArtifact:
        key, cache_dir_ctx = CacheCompiledArtifact._prepare_load(
            path=path, format=format
        )
        return CacheCompiledArtifact._load_impl(cache_dir_ctx, key)


class AOTCompiledArtifact(CompiledArtifact):
    """
    Similar to CompiledArtifact, but the object is a single, bundled precompiled function.
    This object is always a serializable callable function.

    This object is essentially a wrapper for BundledAOTAutogradSerializableCallable, which
    is used by torch._dynamo.aot_compile for AOT Precompilation.
    """

    AOT_HEADER = bytes("AOTCompiledArtifact", "utf-8")

    def __init__(
        self,
        compiled_fn: Callable[..., Any],
    ):
        self.inner_fn = BundledAOTAutogradSerializableCallable(compiled_fn)
        self._artifacts = (
            None  # We don't need artifacts, the inner object handles everything
        )

    @staticmethod
    def from_bundled_callable(
        bundled_fn: BundledAOTAutogradSerializableCallable,
    ) -> AOTCompiledArtifact:
        return AOTCompiledArtifact(bundled_fn.compiled_fn)

    def __call__(self, *args: Any) -> Any:
        return self.inner_fn(*args)

    def save(
        self, *, path: str, format: Literal["binary", "unpacked"] = "binary"
    ) -> None:
        if format == "unpacked":
            raise RuntimeError(
                "AOTCompiledArtifact does not support unpacked format yet"
            )
        result_bytes = self.serialize()
        from torch.utils._appending_byte_serializer import BytesWriter

        from .codecache import torch_key

        writer = BytesWriter()
        writer.write_bytes(AOTCompiledArtifact.AOT_HEADER)
        writer.write_bytes(torch_key())
        writer.write_bytes(result_bytes)

        from torch._inductor.codecache import write_atomic

        # Save a sentinel file to indicate that this is AOT
        write_atomic(path, writer.to_bytes())

    def serialize(self) -> bytes:
        return BundledAOTAutogradSerializableCallable.serialize_compile_artifacts(
            self.inner_fn
        )

    @staticmethod
    def deserialize(result_bytes: bytes) -> AOTCompiledArtifact:
        deserialized = (
            BundledAOTAutogradSerializableCallable.deserialize_compile_artifacts(
                result_bytes
            )
        )
        if not isinstance(deserialized, BundledAOTAutogradSerializableCallable):
            raise AssertionError(
                f"expected BundledAOTAutogradSerializableCallable, got {type(deserialized)}"
            )
        return AOTCompiledArtifact.from_bundled_callable(deserialized)

    @staticmethod
    def load(
        *, path: str, format: Literal["binary", "unpacked"] = "binary"
    ) -> CompiledArtifact:
        if format == "unpacked":
            raise RuntimeError(
                "AOTCompiledArtifact does not support unpacked format yet"
            )
        with open(path, "rb") as file:
            from torch.utils._appending_byte_serializer import BytesReader

            from .codecache import torch_key

            result_bytes = file.read()
            reader = BytesReader(result_bytes)
            header = reader.read_bytes()
            if header != AOTCompiledArtifact.AOT_HEADER:
                raise AssertionError("expected AOTCompiledArtifact header")
            if reader.read_bytes() != torch_key():
                raise AssertionError("torch_key mismatch in serialized artifact")
            artifact = reader.read_bytes()
            if not reader.is_finished():
                raise AssertionError("expected reader to be finished")
            return AOTCompiledArtifact.deserialize(artifact)


def _resolve_ignore_shape_env(dynamic_shapes: DynamicShapesType):
    # tells compile_fx to ignore the shape_envs on the ambient context
    # and the graph_module.
    return dynamic_shapes == "from_example_inputs"


def _resolve_fake_mode(
    gm: GraphModule,
    dynamic_shapes: DynamicShapesType,
    fake_mode: FakeTensorMode | None = None,
) -> FakeTensorMode:
    if dynamic_shapes == "from_example_inputs":
        if fake_mode is not None:
            if fake_mode.shape_env is None:
                raise ValueError(
                    "standalone_compile requires `fake_mode` to have a ShapeEnv "
                    'when `dynamic_shapes="from_example_inputs"`.'
                )
            return fake_mode
        return FakeTensorMode(shape_env=ShapeEnv())
    elif fake_mode is not None:
        raise ValueError(
            "standalone_compile only supports passing `fake_mode` when "
            '`dynamic_shapes="from_example_inputs"`.'
        )
    elif dynamic_shapes == "from_tracing_context":
        # Reuse fake_mode from the TracingContext.
        # NB: The TracingContext only exists if we're currently in a torch.compile backend.
        context = torch._guards.TracingContext.get()
        if context.fake_mode is None:
            raise AssertionError("expected TracingContext.fake_mode to not be None")
        return context.fake_mode
    elif dynamic_shapes == "from_graph":
        # Strategy: find a FakeTensor in the graph output, grab its FakeTensorMode.
        # The graph passed to standalone_compile must be an Inductor-approved graph,
        # which means that there is at least one Tensor output and the output node
        # contains a flat list of Tensors.
        last_node = next(iter(reversed(gm.graph.nodes)))
        if last_node.op != "output":
            raise AssertionError(
                f"expected last node op == 'output', got {last_node.op}"
            )
        if len(last_node.args) != 1:
            raise AssertionError(
                f"expected last node to have 1 arg, got {len(last_node.args)}"
            )

        # If gm came from Dynamo, then last_node.args[0] is always a list,
        # even in single-Tensor returns.
        #
        # It's possible to get into a situation where last_node.args[0]
        # is a Node (and not a list!). This happens if you call split_module
        # on the graph. We allow for this case since it is common.
        nodes = (
            [last_node.args[0]]
            if isinstance(last_node.args[0], torch.fx.Node)
            else last_node.args[0]
        )
        for node in nodes:
            if "example_value" in node.meta:
                maybe_tensor = node.meta["example_value"]
                if isinstance(maybe_tensor, torch._subclasses.fake_tensor.FakeTensor):
                    return maybe_tensor.fake_mode

        return FakeTensorMode(shape_env=ShapeEnv())
    else:
        raise ValueError(
            f"standalone_compile got unsupported `dynamic_shapes` value: dynamic_shapes={dynamic_shapes}."
        )


@contextlib.contextmanager
def _standalone_context(
    gm: GraphModule,
    dynamic_shapes: DynamicShapesType,
    aot: bool,
    fake_mode: FakeTensorMode | None = None,
):
    from torch.compiler._cache import CacheArtifactManager

    resolved_fake_mode = _resolve_fake_mode(gm, dynamic_shapes, fake_mode)
    tracing_context = torch._guards.TracingContext(resolved_fake_mode)
    with (
        torch._guards.tracing(tracing_context),
        CacheArtifactManager.with_fresh_cache(),
        config.patch("triton.autotune_at_compile_time", True),
        torch._functorch.config.patch(
            {
                "bundled_autograd_cache": aot,
                # Standalone artifacts are saved immediately after compile_fx
                # returns. Training graphs normally lower the backward lazily on
                # first backward(), so force it while the artifact recorder is
                # still active.
                "force_non_lazy_backward_lowering": True,
            }
        ),
    ):
        yield


def standalone_compile(
    gm: GraphModule,
    example_inputs: Sequence[InputType],
    *,
    dynamic_shapes: DynamicShapesType,
    options: Any,
    aot: bool = False,  # AOT mode, which uses BundledAOTAutogradCache
    donate_graph_module: bool = False,
    fake_mode: FakeTensorMode | None = None,
) -> CompiledArtifact:
    """
    Implementation of torch.inductor.standalone_compile
    """
    from .compile_fx import compile_fx

    ignore_shape_env = _resolve_ignore_shape_env(dynamic_shapes)
    with _standalone_context(gm, dynamic_shapes, aot, fake_mode):
        # compile_fx takes ownership of gm and may mutate it on cache miss.
        # Deepcopy first so the rewrites below land on the owned copy rather
        # than the caller's gm. The gm may carry a non-pickleable torchbind
        # ProcessGroup (or, after a previous unbox, a Python
        # ``dist.ProcessGroup``); smuggle it through deepcopy as a shared
        # reference instead of crashing.
        if not donate_graph_module:
            with _share_torchbind_and_process_group_on_deepcopy():
                gm = copy.deepcopy(gm)
        # ``make_fx`` traces ``dist.*`` collectives as opaque ``c10d.{op}_``
        # calls. Inductor's collective machinery only recognizes the
        # ``_c10d_functional.{op}`` + ``wait_tensor`` form, so rewrite here
        # before compile_fx runs. Also unbox any torchbind ProcessGroup
        # attrs into Python ``dist.ProcessGroup`` so the runtime collective
        # op accepts them (raw torchbind is rejected).
        gm = _functionalize_inplace_collectives(gm)
        gm = _unbox_process_group_torchbinds(gm)
        compiled_fn = compile_fx(
            gm, example_inputs, ignore_shape_env=ignore_shape_env, **options
        )
        if not callable(compiled_fn):
            raise AssertionError("expected compiled_fn to be callable")
        if aot:
            if not hasattr(compiled_fn, "serialize"):
                raise RuntimeError(
                    "Compiled function should have serialize method when aot=True"
                )
            return AOTCompiledArtifact(compiled_fn)
        artifacts = torch.compiler.save_cache_artifacts()
        if artifacts is None:
            log.warning(
                "standalone_compile artifact generation failed, cannot save. "
                "Run with TORCH_LOGS=+torch._inductor.codecache to identify the problem"
            )

    return CacheCompiledArtifact(compiled_fn, artifacts)


def _defines_module_level_call(src: str) -> bool:
    """Whether ``src`` is the runnable Inductor output module (vs a kernel-only one).

    The module-level ``call`` entry point is codegen'd in two forms: when
    ``config.graph_partition`` is on it is ``call = runner.call`` (the ``def call`` is
    an indented ``Runner`` method); otherwise it is a top-level ``def call(args):``.
    graph_partition defaults off in fbcode, so both forms must be recognized.

    Detection is structural: we AST-parse the module and look for a top-level
    ``def call`` or a top-level ``call = runner.call`` assignment. This avoids the
    false positives a raw substring scan hits, where an embedded triton-kernel
    string, comment, or docstring (even one whose ``def call(`` text sits at column 0
    inside a triple-quoted literal) would match. The partition form's actual ``def
    call`` method is nested inside a class and is intentionally NOT matched directly;
    it is recognized only via the top-level ``call = runner.call`` binding, exactly as
    before. If ``src`` does not parse (it always should, being codegen'd python), we
    conservatively report no module-level ``call``.
    """
    try:
        module = ast.parse(src)
    except SyntaxError:
        return False
    for stmt in module.body:
        if isinstance(stmt, ast.FunctionDef) and stmt.name == "call":
            return True
        if isinstance(stmt, ast.Assign):
            targets_call = any(
                isinstance(t, ast.Name) and t.id == "call" for t in stmt.targets
            )
            value = stmt.value
            is_runner_call = (
                isinstance(value, ast.Attribute)
                and value.attr == "call"
                and isinstance(value.value, ast.Name)
                and value.value.id == "runner"
            )
            if targets_call and is_runner_call:
                return True
    return False


class NoRunnableInductorModuleError(RuntimeError):
    """Raised by ``compile_to_python`` when the graph yields no runnable Inductor
    output module -- it has no compute to lower (returns inputs/constants unchanged),
    so there is no module-level ``call`` to inline. Distinct from an unexpected
    multiple-module case; callers (e.g. torch.compiler.precompile) convert this to a clear
    user-facing error suggesting an alternative.
    """


def _extract_runnable_module(captured: list[str]) -> str:
    """Return the single Inductor output-code module that defines the runnable
    module-level ``call`` entry point, from the sources captured via the
    ``GraphLowering.save_output_code`` hook during codegen.

    No post-hoc stripping is needed: ``compile_to_python`` disables the benchmark
    harness at codegen time, so each captured module is the runnable kernels plus
    ``call``. The compile-time auto-tuning block (a harmless inert raw-string
    docstring; the real auto-tuning already ran during codegen) is retained at the top
    of the captured module and does not affect ``call`` detection. A standalone compile
    of an Inductor-approved graph yields exactly one such module, whether the source
    comes from fresh codegen or a cache-restore path (both fire the hook).

    Nested benchmark lowerings (max_autotune SubgraphChoiceCaller) also define a
    ``call`` but are already filtered out by ``compile_to_python``'s capture hook
    (they are named ``benchmark_*``), so ``len(runnable) > 1`` here means a genuine
    unexpected multi-module case, not a benchmark lowering leaking through.
    """
    runnable = [s for s in captured if _defines_module_level_call(s)]
    if len(runnable) == 0:
        # No module defines ``call`` -- the graph has no compute to lower (it returns
        # inputs/constants unchanged, e.g. ``lambda x: x``, ``x.detach()``, ``return 7``).
        raise NoRunnableInductorModuleError(
            "the compiled graph produced no runnable Inductor output module: it has no "
            "compute to lower (returns inputs/constants unchanged)."
        )
    if len(runnable) > 1:
        raise RuntimeError(
            f"expected exactly one runnable Inductor output module, found "
            f"{len(runnable)}; compile_to_python cannot inline this artifact."
        )
    return runnable[0]


def _binary_cache_bytes(artifact: CompiledArtifact) -> bytes | None:
    """Serialize the artifact to opaque cache bytes, or None if it is not
    serializable (e.g. graphs with input mutations currently do not produce a
    saveable aot_autograd artifact). The source still runs standalone without it."""
    # The only EXPECTED no-cache case is an artifact with no saveable aot_autograd
    # entry (e.g. certain input-mutating graphs); detect it up front and skip the
    # save so a genuine ``save`` regression surfaces below at warning instead of
    # being silently downgraded to an "uncacheable" fallback (which would only show
    # up later as a missing FxGraphCache hit on reload).
    if not isinstance(artifact, CacheCompiledArtifact) or not artifact.is_saveable():
        log.debug("standalone artifact has no saveable cache entry; no cache")
        return None
    try:
        # Serialize fully in memory: these are exactly the bytes ``save(format=
        # "binary")`` would write, and ``CompiledArtifact.load`` reads them back via
        # ``load_cache_artifacts``, so the cache round-trips. Avoiding a temp file
        # here means a SIGKILL mid-compile leaves no stray ``.bin`` in the cache dir.
        return artifact._to_binary_bytes()
    except Exception:
        log.warning(
            "standalone artifact failed to serialize unexpectedly; no cache",
            exc_info=True,
        )
        return None


def compile_to_python(
    gm: GraphModule,
    example_inputs: Sequence[InputType],
    *,
    dynamic_shapes: DynamicShapesType = "from_example_inputs",
    options: dict[str, Any] | None = None,
) -> tuple[str, bytes | None]:
    """Compile ``gm`` and return ``(inner_python, cache)`` -- the INNER half of the
    backend contract behind ``torch.compiler.precompile``.

    This is an INTERNAL layered-contract entry point, not an end-user API. End users
    should call ``torch.compiler.precompile``; this function only emits the inductor
    piece of the artifact and assumes its caller (the AOT layer) wraps it. It lives in
    ``torch._inductor`` (a private, leading-underscore module), so it is exposed for
    the AOT layer to import, not as a stable public surface.

    ``inner_python`` is the Inductor output module exposing ``call(args) -> outs``
    for the post-AOTAutograd inner graph (dense, functionalized). It is the inductor
    piece only: it carries NO prelude/epilogue (subclass flatten/unflatten, input-
    mutation copy-back, output-alias regen, grad disabling). Those belong to the AOT
    layer -- see ``torch._functorch.aot_autograd.compile_to_python``, which calls
    this and composes AOTAutograd's codegen'd runtime wrappers around the result.
    Callers must run ``call`` under ``torch.no_grad()`` (the kernels use out= ops).

    Caller preconditions (this layer does not re-derive them; passing a graph that
    violates them yields wrong/unrunnable output rather than a clean error):

    - ``gm`` is a post-AOTAutograd dense, functionalized inner graph (the dense
      forward/backward AOTAutograd hands to its inductor backend), NOT a raw Dynamo
      or pre-dispatch graph still carrying subclasses or autograd.
    - ``gm`` lowers to exactly ONE runnable inductor output module (one module-level
      ``call``). A graph with no compute raises ``NoRunnableInductorModuleError``; an
      unexpected multi-module lowering raises ``RuntimeError``.

    ``dynamic_shapes`` defaults to ``"from_example_inputs"`` here, which DIFFERS from
    the sibling ``torch._inductor.standalone_compile`` default of ``"from_graph"``.
    The reason: this entry point is fed the post-AOTAutograd inner graph, whose
    example-value metadata is the source of truth AOTAutograd already resolved, so
    specializing on ``example_inputs`` matches what the AOT layer expects; the public
    ``standalone_compile`` instead trusts the dynamic shapes baked into the passed-in
    graph. See ``DynamicShapesType`` for the full set of modes.

    ``options`` is an optional inductor config-override dict (``None`` means no
    overrides), the same shape as ``torch._inductor.compile``'s ``options``. The
    keys are inductor config names: they are merged into the ``config.patch``
    block this function already wraps the compile in, so they take effect as
    config rather than being forwarded as ``compile_fx`` kwargs. The two
    capture-critical pins below (``benchmark_harness``, ``cpp_wrapper``) always win
    over a conflicting user option, since the source-capture contract depends on
    them and a caller cannot be allowed to break the emitted module.

    The kernels JIT-compile from the inlined source on first call, so ``inner_python``
    needs no cache. ``cache`` is an opaque acceleration (or ``None`` when the graph
    is not serializable, e.g. some input-mutating graphs, or when caches are disabled
    via ``force_disable_caches`` / ``fx_graph_cache=False``).

    The source is captured directly off codegen via the process-global
    ``GraphLowering.save_output_code`` hook, decoupled from the cache: it produces
    valid ``inner_python`` even when no cacheable artifact exists (the ``cache`` is
    then ``None``). Because the hook is process-global it also fires for an ordinary
    ``torch.compile`` running concurrently on ANOTHER thread, so the hook only captures
    codegen on the installing (owner) thread -- this is what keeps a foreign compile's
    source out of the capture. The module-level ``_COMPILE_TO_PYTHON_LOCK`` serializes
    two ``compile_to_python`` calls against each other (so their hook install/restore
    and the ``with_fresh_cache`` process-global cache state do not interleave); it does
    NOT, by itself, block a concurrent ``torch.compile`` (the owner-thread check does).
    The AOT-level caller holds its own distinct ``_COMPILE_LOCK`` (see
    ``torch._functorch.aot_autograd.compile_to_python``) and always acquires it before
    reaching here, so the order is AOT-then-inductor and the two locks never deadlock.
    The hook is restored in a ``finally`` so it does not leak to other
    ``save_output_code`` users.
    """
    if not isinstance(gm, torch.fx.GraphModule):
        raise TypeError(
            f"compile_to_python expects a post-AOTAutograd torch.fx.GraphModule, "
            f"got {type(gm)}. This is an internal entry point; call "
            f"torch.compiler.precompile instead."
        )
    from .graph import GraphLowering
    from .virtualized import V

    # Suppress the benchmark harness at codegen time rather than stripping it out of
    # the emitted source afterward (the export artifact is meant to run, not be
    # profiled): benchmark_harness emits get_args()/benchmark_compiled_module()/
    # __main__.
    captured: list[str] = []
    # The thread that installed the hook (set inside the lock below). The hook lives on
    # the PROCESS-GLOBAL GraphLowering.save_output_code, so it also fires for an ordinary
    # torch.compile running concurrently on ANOTHER thread; we only capture codegen on
    # the owning thread (mirroring the AOT capture sink, which is thread-local).
    owner_thread: int | None = None

    def capture_hook(src: str) -> None:
        # Ignore codegen from any other thread: a concurrent normal torch.compile (or
        # another foreign lowering) on a different thread would otherwise append its
        # module here and trip the single-module check in _extract_runnable_module.
        if threading.get_ident() != owner_thread:
            return
        # Under max_autotune, nested benchmark GraphLowerings (e.g.
        # SubgraphChoiceCaller._compile_for_benchmarking) fire this same
        # process-global hook with their own ``def call(`` while compiling the
        # outer graph. Those benchmark lowerings are named ``benchmark_*`` (see
        # codegen/subgraph.py); skip them so only the OUTERMOST runnable module is
        # captured. ``V.graph`` is the GraphLowering currently being codegen'd when
        # the hook fires (set via ``V.set_graph_handler``), so its ``name`` tells us
        # whether this is a nested benchmark lowering.
        graph = V.graph
        name = getattr(graph, "name", None)
        if isinstance(name, str) and name.startswith("benchmark_"):
            return
        captured.append(src)

    with _COMPILE_TO_PYTHON_LOCK:
        # Read the ambient hook INSIDE the lock: on the precompile path compile_to_python
        # is the only writer and is serialized by this lock, so a concurrent caller has
        # already restored its own hook before we read here. Reading outside the lock is a
        # TOCTOU that would leak another caller's dead capture_hook into the process-global.
        #
        # ``_COMPILE_TO_PYTHON_LOCK`` only serializes ``compile_to_python`` against itself;
        # it does NOT block an ordinary ``torch.compile`` (or the test-only
        # ``run_and_get_code`` / ``get_code`` helpers, which patch the same process-global
        # hook without this lock) running on another thread. The owner-thread check in
        # ``capture_hook`` is what keeps a concurrent foreign compile's source out of our
        # capture; the lock just keeps two ``compile_to_python`` calls from interleaving
        # their hook install/restore. Read the ambient hook INSIDE the lock so a concurrent
        # compile_to_python has already restored its own hook (reading outside is a TOCTOU
        # that would leak a dead capture_hook into the process-global).
        owner_thread = threading.get_ident()
        prev_hook = GraphLowering.save_output_code
        GraphLowering.save_output_code = staticmethod(capture_hook)
        # Treat ``options`` as inductor config overrides and fold them into the same
        # ``config.patch`` we already enter for the capture-critical pins. The pins are
        # applied AFTER the user options so they override any conflicting key: the
        # source-capture contract depends on them and must not be user-overridable.
        config_patches: dict[str, Any] = dict(options or {})
        config_patches.update(
            {
                "benchmark_harness": False,
                # The C++ wrapper backend emits a C++ ``call``, not the python
                # ``def call(args)`` this lowering extracts and inlines, so a python
                # artifact cannot come from it. Pin it off regardless of the ambient
                # config so the captured module is always the python one.
                "cpp_wrapper": False,
            }
        )
        try:
            with torch.no_grad(), config.patch(config_patches):
                # ``options`` are already applied above as config, so pass none down
                # to ``standalone_compile`` (it forwards options as ``compile_fx``
                # kwargs, which a config key like ``max_autotune`` is not).
                artifact = standalone_compile(
                    gm,
                    example_inputs,
                    dynamic_shapes=dynamic_shapes,
                    options={},
                )
        finally:
            GraphLowering.save_output_code = prev_hook
    inner_python = _extract_runnable_module(captured)
    cache = _binary_cache_bytes(artifact)
    return inner_python, cache


def autograd_cache_key(
    graph,
    example_inputs,
    dynamic_shapes: DynamicShapesType,
    aot: bool = False,  # AOT mode, which uses BundledAOTAutogradCache
    fake_mode: FakeTensorMode | None = None,
):
    from . import compile_fx

    ignore_shape_env = _resolve_ignore_shape_env(dynamic_shapes)
    with _standalone_context(graph, dynamic_shapes, aot, fake_mode):
        return compile_fx.autograd_cache_key(
            graph,
            example_inputs,
            ignore_shape_env=ignore_shape_env,
        )
