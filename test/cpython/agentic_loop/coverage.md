# CPython Dynamo Agentic Coverage Plan

Status: Cycle 2 top-10 exhausted (G11-G20). G13/G14/G18/G19/G20 landed
(G15/G16/G17 were G14 collateral); G11/G12 were triaged and DEFERRED (CPython
C-implementation-detail / object-lifetime-GC internals, sentinels left in
place). G19 (`test_deque-TestBasic.test_basics`) landed via deque `__init__`
support. G20 (`test_range-RangeTest.test_range_iterators`) landed via `object()`
support on top of the native itertools iterator-variable stack (chain /
zip_longest / islice), which also makes that otherwise-pathological test
practical under Dynamo. The relevance CSVs were regenerated per the README (32
landed/collateral rows dropped, deferred rows kept and tagged). The new
actionable top-10 is Cycle 3, G21-G30, drawn from the re-ranked CSV. G21
(`test_range-RangeTest.test_user_index_method`) landed via `__index__`
coercion in `range()` and slice subscript. G22
(`test_sort-TestDecorateSortUndecorate.test_reverse_stability`) was triaged and
DEFERRED (data-dependent sort over dynamic random values; needs random -> SymInt
routing or data-dependent sort support; sentinel left in place). G23
(`test_list-ListTest.test_init`, relevance 78.4) is the next unworked actionable
gate.

Goal: improve `PYTORCH_TEST_WITH_DYNAMO=1` coverage for CPython tests by
working the highest-value expected failures first. The actionable gates come
from the relevance-ranked study in:

```
test/cpython/agentic_loop/cpython_dynamo_expected_failure_relevance.csv
```

Each gate targets exactly one CPython expected-failure sentinel. Each gate
should produce one focused implementation commit after review and validation.

Operational loop instructions live in:

```
test/cpython/agentic_loop/agent_manager.md
```

CPython protocol orientation lives in:

```
test/cpython/agentic_loop/CPYTHON_MIRRORING.md
```

## Ground Rules

- Do not relax gate exit criteria during an implementation cycle.
- Do not mark a gate complete without measured evidence from the current tree.
- Do not edit vendored CPython tests under `test/cpython/v3_13` unless the
  human explicitly asks for a CPython import/update.
- Do not add new expected-failure or skip sentinels without human approval.
- Use `agent_space/` for scratch files, temporary reports, JUnit XML, and
  sentinel backups.
- Prefer source fixes in `torch/_dynamo` and focused regression tests in the
  normal Dynamo test suite.
- Remove only the CPython sentinel proven fixed by the gate.
- Do not batch multiple gates into one commit.

Sentinel directories:

```
test/dynamo_expected_failures/
test/dynamo_skips/
```

Example key:

```
CPython313-test_list-ListTest.test_constructors
```

## Baseline

Fast CPU validation baseline:

```
test/cpython/agentic_loop/cpu_fast_ci_baseline.md
```

Baseline command:

```bash
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu \
  python agent_space/run_cpython_and_dynamo_timing.py --shards 32
```

Validation harness note: the fast CPU loop script
`agent_space/run_cpython_and_dynamo_timing.py` does not exist in this repo.
Every prior gate used affected-CPython-file runs plus targeted Dynamo suites
for validation instead; new gates should do the same.

## Validation Commands

Run one target test with Dynamo:

```bash
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu \
PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short \
  test/cpython/v3_13/test_iter.py::TestCase::test_reduce_mutating_builtins_iter
```

Run the affected CPython file:

```bash
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu \
PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short \
  test/cpython/v3_13/test_iter.py
```

Use Dynamo logs only for opaque single-test repros:

```bash
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu \
TORCH_LOGS="+dynamo,graph_breaks" TORCHDYNAMO_VERBOSE=1 \
PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short \
  test/cpython/v3_13/test_iter.py::TestCase::test_reduce_mutating_builtins_iter
```

Before claiming a gate complete, the implementation subagent must report:

- target sentinel removed;
- focused regression test added or updated when the fix is semantic;
- target CPython test passes with the sentinel removed;
- affected CPython file has no new real failures;
- fast CPU validation loop ran and only baseline failures remain, except for
  the expected pass/skip improvement from this gate (or, if the harness script
  is absent, the affected-file run substitute);
- `lintrunner -a` passed before the gate commit.

## Completed Gates (Cycle 1 ledger)

These gates landed; their sentinels were removed and committed. CSV rows for
the landed tests have been dropped from both relevance CSVs.

| Gate | Test | Commit | Resolution |
|---|---|---|---|
| G1 | `CPython313-test_dict-DictTest.test_eq` | 3bdd341f4e6 | `ConstantVariable.is_python_equal` routed non-constant `other` through `generic_richcompare_bool` so a user `__eq__` (and its raised exception) runs, mirroring `PyObject_RichCompareBool`. Regression: `test_cmp_eq_key_raises` in `test/dynamo/test_dicts.py`. |
| G4 | `CPython313-test_defaultdict-TestDefaultDict.test_shallow_copy` | 82c76e895ad | Extended `DefaultDictVariable.call_method` to handle the `__copy__` slot and added a `UserDefinedClassVariable.call_method` branch so the unbound `defaultdict.__copy__(instance)` form from `copy.copy` dispatches to the instance, preserving type, `default_factory`, and contents. Regression: `test_defaultdict_shallow_copy_preserves_factory`. |
| G5 | `CPython313-test_set-TestFrozenSet.test_do_not_rehash_dict_keys` | 3a0da519450 | tp_hash slot dispatch (`type.__hash__(instance)` for int/float/str + subclasses) plus CPython `set_update_internal` do-not-rehash semantics: set/frozenset/dict built from an existing set/dict reuse stored `HashableTracker` keys instead of re-hashing. Touched `builtin.py`, `sets.py`, `dicts.py`, `functions.py`. Regression: `MiscTests.test_do_not_rehash_dict_keys`. |
| G6 | `CPython313-test_set-TestFrozenSetSubclass.test_do_not_rehash_dict_keys` | 3a0da519450 | G5 collateral (same root-cause fix, sentinel removed in the G5 commit). |
| G7 | `CPython313-test_set-TestSet.test_do_not_rehash_dict_keys` | 3a0da519450 | G5 collateral (same root-cause fix, sentinel removed in the G5 commit). |
| G8 | `CPython313-test_set-TestSetSubclass.test_do_not_rehash_dict_keys` | 3a0da519450 | G5 collateral (same root-cause fix, sentinel removed in the G5 commit). |

## Deferred Gates (Cycle 1 ledger)

These tests were triaged and intentionally deferred. Their sentinels are LEFT
IN PLACE (still in `test/dynamo_expected_failures/`). Their CSV rows are kept
but tagged in the new `deferred` column so they do not re-surface as active
gates.

| Gate | Test | Reason |
|---|---|---|
| G2 | `CPython313-test_dict-DictTest.test_fromkeys` | Local dict-subclass construction machinery (out of scope). Test body builds `fromkeys` against locally-defined `dict` subclasses overriding `__new__`/`__init__`/`__setitem__`; passing requires source-backed local subclass construction. The instance-method `fromkeys` routing improvement is a valid standalone change but does not make the gate pass. |
| G3 | `CPython313-test_dict-DictTest.test_getitem` | Class-body closure cell / local class construction (out of scope). `class BadEq` closes over the later-defined `Exc` cell; tracing the class body reads the still-uninitialized `Exc` cell -> "Read uninitialized cell" graph break. |
| G9 | `CPython313-test_list-ListTest.test_deopt_from_append_list` | Vendored `@unittest.skip("Fails on python <=3.13.2 ...")`; the skip dominates the expected-failure wrapper so the test is skipped with or without the sentinel. The sentinel is a dead but harmless artifact; passing would require editing the vendored test (forbidden). Human chose to leave it deferred. |
| G10 | `CPython313-test_dict-DictTest.test_copy_maintains_tracking` | `gc.is_tracked` is a CPython cyclic-GC container-tracking introspection builtin (in Dynamo skipfiles) with no analog in the `VariableTracker` model; `CPYTHON_MIRRORING.md` lists GC traversal under "What not to mirror". Out of scope. |

## Gates (Cycle 2: actionable top-10, G11-G20)

These are the ten highest-ranked rows whose `deferred` column is empty, taken
from the regenerated `cpython_dynamo_expected_failure_relevance.csv`. Gate
numbers continue from G11 to avoid collisions with the Cycle 1 ledger.

### G11: Iterator Reduce With Mutating Builtins (`bytes`)

Status: DEFERRED (Cycle 2). Triaged as a CPython C-implementation-detail test
with no analog in Dynamo's `VariableTracker` model. Sentinel LEFT IN PLACE;
CSV row tagged `deferred`. No source change made.

Root-cause classification: out of scope (CPython interpreter internals /
"what not to mirror"). The visible `bytes(8)` graph break (relevance score) is
only the first of several blockers and is not the subject of the test. The test
is the reproducer for CPython issue #101765: it verifies the C-level argument
evaluation ordering inside `listiter_reduce_general`
(`Objects/listobject.c`), where `_PyEval_GetBuiltin(&_Py_ID(iter))` must run
BEFORE the iterator's internal `it_seq`/`it_index` pointers are read. The test
mutates `builtins.__dict__` (deletes `iter`/`reversed`, re-inserts under a
custom-`__hash__`/`__eq__` key whose `__eq__` exhausts the iterator), then
calls `it.__reduce__()` on builtin iterators and asserts the returned pickle
state reflects the now-exhausted iterator, e.g.
`run_iter("xyz") == (orig["iter"], ("",))`.

Making this gate pass would require Dynamo to symbolically execute CPython's C
`__reduce__` implementations for every builtin iterator (list/str/tuple/
callable/`reversed`) AND reproduce CPython's undefined-in-C argument evaluation
ordering and the `_PyEval_GetBuiltin`-reads-mutated-`builtins` interplay. Per
`CPYTHON_MIRRORING.md` "What not to mirror" (CPython implementation details
that do not affect tracing semantics), this is explicitly out of scope. The
isolated `bytes(int)`/`bytearray(int)` constructor gap is a legitimate, separate
fixable improvement but does NOT make this gate pass on its own.

Repro evidence (current tree, sentinel temporarily removed):

```bash
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu PYTORCH_TEST_WITH_DYNAMO=1 \
  pixi run -w pytorch -e pytorch313 python -m pytest \
  "test/cpython/v3_13/test_iter.py::TestCase::test_reduce_mutating_builtins_iter" -q -rs
# -> FAILED: torch._dynamo.exc.Unsupported: Failed to trace builtin operator
#    "Dynamo does not know how to trace builtin operator `bytes` with argument
#    types ['int']"; from user code at test_iter.py:341 on `(bytes(8),)`.
```

Isolated probe confirming the deeper blocker (after the `bytes` break):

```python
@torch.compile(backend="eager", fullgraph=True)
def f():
    return iter([1, 2, 3]).__reduce__()
# -> Unsupported: Dynamo does not know how to trace method `__reduce__`
#    of class `list_iterator`
```

(Mutating `builtins.__dict__` itself - del + setitem - does trace fine; the
unsupported pieces are iterator `__reduce__` and the C-ordering semantics.)

The original gate scaffolding (target sentinel, test, relevance, baseline
failure kind, source areas) is preserved below for the record.

Sanity-checked (pre-defer) still an expected failure (SKIPPED, not XPASS) on
the current tree.

Target sentinel:

```
CPython313-test_iter-TestCase.test_reduce_mutating_builtins_iter
```

Target test:

```bash
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu \
PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short \
  test/cpython/v3_13/test_iter.py::TestCase::test_reduce_mutating_builtins_iter
```

Relevance score: 80.2.

Baseline failure kind:

```
Failed to trace builtin operator (Dynamo does not know how to trace builtin
operator `bytes` with argument types ['int']); graph break at
test/cpython/v3_13/test_iter.py:341 on `(bytes(8),)`.
```

Likely source areas:

```
torch/_dynamo/variables/builtin.py
torch/_dynamo/variables/iter.py
torch/_dynamo/variables/constant.py
```

Exit criteria:

- Remove `test/dynamo_expected_failures/CPython313-test_iter-TestCase.test_reduce_mutating_builtins_iter`.
- Add focused Dynamo regression coverage when the fix is semantic.
- The target test passes with `PYTORCH_TEST_WITH_DYNAMO=1`.
- The full `test_iter.py` CPython file has no new real failures.
- Fast CPU validation (affected-file substitute) passes modulo documented
  baseline failures.
- Commit exactly this gate.

### G12: List Free After Iterating

Status: DEFERRED (Cycle 2). Triaged as a CPython object-lifetime / GC-internals
test with no analog in Dynamo's `VariableTracker` model. Sentinel LEFT IN PLACE;
CSV `deferred` column tagged. No source change made. G13 promoted to active.

Root-cause classification: out of scope (refcounting / object lifetime / GC,
plus local class construction). The visible "Attempted to call function marked
as skipped" graph break is `importlib.import_module`, reached only because the
test is wrapped in `@support.suppress_immortalization()`. That contextmanager
calls `_testinternalcapi.suppress_immortalization(True)` -- a CPython internal
C-API that toggles object *immortalization* (a refcount/lifetime mechanism) so
the test can observe deterministic deallocation. Even past that blocker, the
test body `test.support.check_free_after_iterating` defines a local subclass
`class A(cls)` with a `__del__` that runs `next(it)`, then asserts (after
`gc_collect()`) that `__del__` fired exactly when the sequence was deallocated
at end-of-iteration. This is object lifetime, `__del__`-on-deallocation, and GC
collection behavior -- all explicitly listed under `CPYTHON_MIRRORING.md`
"What not to mirror" (refcounting and object lifetime, deallocation slots, GC
traversal) -- combined with local class construction machinery (the
low-relevance early-exit category). Dynamo does not model `__del__` finalizer
timing or deallocation, so making this gate pass would require mirroring CPython
object lifetime, which is out of scope.

Repro evidence (current tree, sentinel temporarily removed):

```bash
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu PYTORCH_TEST_WITH_DYNAMO=1 \
  pixi run -w pytorch -e pytorch313 python -m pytest \
  "test/cpython/v3_13/test_list.py::ListTest::test_free_after_iterating" -q -rs
# -> FAILED: torch._dynamo.exc.Unsupported: Attempted to call function marked
#    as skipped; module: importlib, qualname: import_module. Reached from
#    @support.suppress_immortalization() ->
#    _testinternalcapi = import_module("_testinternalcapi").
```

(The inherited definition lives in `test/cpython/v3_13/seq_tests.py:483`
`test_free_after_iterating` -> `support.check_free_after_iterating(self, iter,
self.type2test)`.)

The original gate scaffolding is preserved below for the record.

Target sentinel:

```
CPython313-test_list-ListTest.test_free_after_iterating
```

Target test:

```bash
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu \
PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short \
  test/cpython/v3_13/test_list.py::ListTest::test_free_after_iterating
```

Relevance score: 80.0.

Baseline failure kind:

```
Attempted to call function marked as skipped
```

Likely source areas:

```
torch/_dynamo/variables/lists.py
torch/_dynamo/variables/iter.py
torch/_dynamo/trace_rules.py
```

Exit criteria:

- Remove `test/dynamo_expected_failures/CPython313-test_list-ListTest.test_free_after_iterating`.
- Add focused Dynamo regression coverage when the fix is semantic.
- The target test passes with `PYTORCH_TEST_WITH_DYNAMO=1`.
- The full `test_list.py` CPython file has no new real failures.
- Fast CPU validation (affected-file substitute) passes modulo documented
  baseline failures.
- Commit exactly this gate.

### G13: List Contains Ordering

Status: FIXED (Cycle 2). Classification (a): genuine in-scope object-protocol
bug in `iter_contains`. Source fix landed in the working tree (uncommitted);
target sentinel and 7 collateral sentinels removed in the working tree.

Root cause: `iter_contains` (`torch/_dynamo/utils.py`) had a constant fast path
that, when `search` was a python constant, scanned items checking only
`x.is_python_constant()` -- silently skipping any non-constant element. CPython
`list_contains` (Objects/listobject.c) instead calls
`PyObject_RichCompareBool(item, search, Py_EQ)` on every element in order and
short-circuits on the first match, so a non-constant element's custom `__eq__`
(and any exception it raises) must run in order. For `[StopCompares(), 1]`,
`1 in ...` returned `True` instead of propagating `StopCompares().__eq__`'s
`DoNotTestEq`.

Fix: only constant-fold when `search` and every element are python constants;
otherwise iterate in order comparing each element via
`generic_richcompare_bool(tx, x, search, "__eq__")` (Dynamo's
`PyObject_RichCompareBool` analog, element first), short-circuiting on a
constant-True result and OR-accumulating symbolic results. This is the list
analog of the G1 dict/set `__eq__`-exception-propagation fix and routes through
the shared object-protocol richcompare path.

Files changed:
- `torch/_dynamo/utils.py` (`iter_contains`)
- `test/dynamo/test_contains_protocol.py` (new `ContainsOrderTest`:
  short-circuit-before-raising-eq, list raising-eq propagation, tuple
  raising-eq propagation)

Sentinels removed (working tree, uncommitted): target + collateral, all
verified failing-before / passing-after with the fix:
- `CPython313-test_list-ListTest.test_contains_order` (target)
- `CPython313-test_tuple-TupleTest.test_contains_order`
- `CPython313-test_userlist-UserListTest.test_contains_order`
- `CPython313-test_deque-TestSequence.test_contains_order`
- `CPython313-test_list-ListTest.test_contains_fake`
- `CPython313-test_tuple-TupleTest.test_contains_fake`
- `CPython313-test_userlist-UserListTest.test_contains_fake`
- `CPython313-test_deque-TestSequence.test_contains_fake`

The 4 `test_contains_fake` tests (rich comparison against `ALWAYS_EQ`/
`NEVER_EQ`) are collateral from the same fix. Validation: all 4 affected
CPython files (`test_list`, `test_tuple`, `test_userlist`, `test_deque`) pass
under Dynamo with the 8 sentinels removed; `test/dynamo/test_contains_protocol`,
`test_list`, `test_dicts`, `test_sets` all green.

Note: `test_set.py::TestFrozenSet::test_hash` and `TestFrozenSetSubclass.test_hash`
XPASS both WITH and WITHOUT this change -- they are pre-existing collateral
(likely from the landed G5 commit), NOT caused by G13, so their sentinels are
left in place for a separate gate to handle.

Next gate: G14.

Original gate scaffolding preserved below.

Status: current active gate (promoted after G12 deferred).

Target sentinel:

```
CPython313-test_list-ListTest.test_contains_order
```

Target test:

```bash
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu \
PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short \
  test/cpython/v3_13/test_list.py::ListTest::test_contains_order
```

Relevance score: 79.8.

Baseline failure kind:

```
Data-dependent branching
```

Likely source areas:

```
torch/_dynamo/variables/lists.py
torch/_dynamo/variables/builtin.py
torch/_dynamo/variables/user_defined.py
```

Exit criteria:

- Remove `test/dynamo_expected_failures/CPython313-test_list-ListTest.test_contains_order`.
- Add focused Dynamo regression coverage when the fix is semantic.
- The target test passes with `PYTORCH_TEST_WITH_DYNAMO=1`.
- The full `test_list.py` CPython file has no new real failures.
- Fast CPU validation (affected-file substitute) passes modulo documented
  baseline failures.
- Commit exactly this gate.

### G14: Set discard

Status: FIXED (Cycle 2). Classification (a): genuine in-scope object-protocol
bug in `set.remove`/`set.discard` key handling. Source fix landed in the
working tree (uncommitted); target sentinel plus 3 collateral sibling sentinels
(G15/G16/G17) removed in the working tree.

Root cause: `SetVariable.call_method` for `remove`/`discard` checked membership
with `args[0] in self` (`__contains__`), which returns `False` for an
unhashable element (`is_hashable` pre-check). So `s.remove([])` raised
`KeyError([])` instead of `TypeError: unhashable type: 'list'`, and
`s.discard([])` silently succeeded instead of raising `TypeError`. CPython
`set_remove_impl`/`set_discard_impl` call `set_discard_key`, which hashes the
key first (raising `TypeError` for an unhashable key), with a set-key fallback
that coerces a set to a frozenset for the lookup (membership-test-with-set
semantics). `KeyError` is only raised, on the original key, after a successful
hash finds the key absent.

Fix: added `SetVariable.lookup_key`, factoring out the CPython
`set_contains_key`/`set_discard_key` key-normalization already inlined in
`sq_contains` (unhashable -> `TypeError`, except a set key is coerced to
`FrozensetVariable`). `sq_contains`, `remove`, and `discard` now all route
through it. `remove`/`discard` normalize the key first, then do the membership
check / pop on the normalized key; `remove` raises `KeyError` on the original
key. This mirrors the G1/G13 object-protocol direction (route through the
shared CPython algorithm rather than a local spot fix).

Files changed:
- `torch/_dynamo/variables/sets.py` (`lookup_key`, `sq_contains`, `remove`,
  `discard`)
- `test/dynamo/test_sets.py` (`_SetBase.test_remove_discard_unhashable`;
  `_SetKeyCoercionMixin.test_remove_set_key` / `test_discard_set_key` on
  `SetTests` and `UserDefinedSetTests`)

Sentinels removed (working tree, uncommitted), each verified failing-before
(initial repro: 4 failed) / passing-after (final repro: 4 passed):
- `CPython313-test_set-TestSet.test_discard` (target, G14)
- `CPython313-test_set-TestSet.test_remove` (G15)
- `CPython313-test_set-TestSetSubclass.test_discard` (G16)
- `CPython313-test_set-TestSetSubclass.test_remove` (G17)

The 3 sibling gates G15/G16/G17 are collateral of the same root-cause fix
(remove is the KeyError-raising twin of discard; the Subclass variants exercise
the same `call_method` path on a `set` subclass).

Validation:
- 4 targets pass under Dynamo with sentinels removed.
- Full `test_set.py` under Dynamo: only the 2 known/unrelated
  `TestFrozenSet::test_hash` / `TestFrozenSetSubclass::test_hash` XPASS remain
  (documented in G13, sentinels left in place).
- `test/dynamo/test_sets.py` (176 passed, 1 skipped) and
  `test/dynamo/test_dicts.py` (297 passed, 1 xfailed) green.

Note: the fast CPU loop script `agent_space/run_cpython_and_dynamo_timing.py`
does not exist; used the affected-CPython-file run plus targeted Dynamo suites
as the validation substitute (consistent with prior gates).

Next gate: G18.

Original gate scaffolding preserved below.

Target sentinel:

```
CPython313-test_set-TestSet.test_discard
```

Target test:

```bash
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu \
PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short \
  test/cpython/v3_13/test_set.py::TestSet::test_discard
```

Relevance score: 79.7.

Baseline failure kind:

```
Observed exception
```

Likely source areas:

```
torch/_dynamo/variables/sets.py
torch/_dynamo/variables/builtin.py
torch/_dynamo/variables/user_defined.py
```

Exit criteria:

- Remove `test/dynamo_expected_failures/CPython313-test_set-TestSet.test_discard`.
- Add focused Dynamo regression coverage when the fix is semantic.
- The target test passes with `PYTORCH_TEST_WITH_DYNAMO=1`.
- The full `test_set.py` CPython file has no new real failures.
- Fast CPU validation (affected-file substitute) passes modulo documented
  baseline failures.
- Commit exactly this gate.

### G15: Set remove

Target sentinel:

```
CPython313-test_set-TestSet.test_remove
```

Target test:

```bash
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu \
PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short \
  test/cpython/v3_13/test_set.py::TestSet::test_remove
```

Relevance score: 79.7.

Baseline failure kind:

```
Observed exception
```

Likely source areas:

```
torch/_dynamo/variables/sets.py
torch/_dynamo/variables/builtin.py
torch/_dynamo/variables/user_defined.py
```

Exit criteria:

- Remove `test/dynamo_expected_failures/CPython313-test_set-TestSet.test_remove`.
- Add focused Dynamo regression coverage when the fix is semantic.
- The target test passes with `PYTORCH_TEST_WITH_DYNAMO=1`.
- The full `test_set.py` CPython file has no new real failures.
- Fast CPU validation (affected-file substitute) passes modulo documented
  baseline failures.
- Commit exactly this gate.

### G16: Set Subclass discard

Target sentinel:

```
CPython313-test_set-TestSetSubclass.test_discard
```

Target test:

```bash
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu \
PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short \
  test/cpython/v3_13/test_set.py::TestSetSubclass::test_discard
```

Relevance score: 79.7.

Baseline failure kind:

```
Observed exception
```

Likely source areas:

```
torch/_dynamo/variables/sets.py
torch/_dynamo/variables/user_defined.py
torch/_dynamo/variables/builtin.py
```

Exit criteria:

- Remove `test/dynamo_expected_failures/CPython313-test_set-TestSetSubclass.test_discard`.
- Add focused Dynamo regression coverage when the fix is semantic.
- The target test passes with `PYTORCH_TEST_WITH_DYNAMO=1`.
- The full `test_set.py` CPython file has no new real failures.
- Fast CPU validation (affected-file substitute) passes modulo documented
  baseline failures.
- Commit exactly this gate.

### G17: Set Subclass remove

Target sentinel:

```
CPython313-test_set-TestSetSubclass.test_remove
```

Target test:

```bash
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu \
PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short \
  test/cpython/v3_13/test_set.py::TestSetSubclass::test_remove
```

Relevance score: 79.7.

Baseline failure kind:

```
Observed exception
```

Likely source areas:

```
torch/_dynamo/variables/sets.py
torch/_dynamo/variables/user_defined.py
torch/_dynamo/variables/builtin.py
```

Exit criteria:

- Remove `test/dynamo_expected_failures/CPython313-test_set-TestSetSubclass.test_remove`.
- Add focused Dynamo regression coverage when the fix is semantic.
- The target test passes with `PYTORCH_TEST_WITH_DYNAMO=1`.
- The full `test_set.py` CPython file has no new real failures.
- Fast CPU validation (affected-file substitute) passes modulo documented
  baseline failures.
- Commit exactly this gate.

### G18: Dict View Containment Check Errors

Status: FIXED (Cycle 2). Classification (a): genuine in-scope object-protocol
bug in dict items-view containment. Source fix landed in the working tree
(uncommitted); target sentinel removed.

Root cause: `DictItemsVariable.sq_contains` compared the stored value via
`is_python_equal` (identity-only), so a stored value's custom `__eq__` never
ran. CPython `dictitems_contains` does
`PyObject_RichCompareBool(found, value, Py_EQ)` on the stored value, so a value
whose `__eq__` raises must propagate that exception; the identity-only check
swallowed it.

Fix: route the stored-value compare through
`generic_richcompare_bool(tx, stored, val, "__eq__")` (stored as the left
operand, matching `found`), mirroring CPython's
`PyObject_RichCompareBool(found, value, Py_EQ)`, so the value `__eq__` runs and
any exception propagates. This also fixes `dict_items` rich comparisons
(`==`,`!=`,`<`,`<=`,`>`,`>=`) that reach the value compare. Sentinel removed;
regression tests `test_dict_items_cmp_value_eq_raises` /
`test_dict_items_cmp_value_present_absent` added (with a `_BadCmpValue` helper
whose `__eq__` raises). This mirrors the G1/G13/G14 object-protocol direction
(route through the shared CPython algorithm rather than a local identity check).

Files changed:
- `torch/_dynamo/variables/dicts.py` (`DictItemsVariable.sq_contains`)
- `test/dynamo/test_dicts.py` (`_BadCmpValue`;
  `DictMethodsTests.test_dict_items_cmp_value_eq_raises` /
  `test_dict_items_cmp_value_present_absent`)

Sentinel removed (working tree, uncommitted):
- `CPython313-test_dict-DictTest.test_errors_in_view_containment_check` (target)

Validation (current tree):

```bash
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu \
PYTORCH_TEST_WITH_DYNAMO=1 python -m pytest \
  "test/cpython/v3_13/test_dict.py::DictTest::test_errors_in_view_containment_check" -q
# 1 passed

CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu \
python -m pytest test/dynamo/test_dicts.py -q
# 303 passed, 1 xfailed
```

Next gate: G19 (G13/G14/G18 landed; G15/G16/G17 were G14 collateral; G11/G12
deferred).

Original gate scaffolding preserved below.

Target sentinel:

```
CPython313-test_dict-DictTest.test_errors_in_view_containment_check
```

Target test:

```bash
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu \
PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short \
  test/cpython/v3_13/test_dict.py::DictTest::test_errors_in_view_containment_check
```

Relevance score: 79.4.

Baseline failure kind:

```
Observed exception
```

Likely source areas:

```
torch/_dynamo/variables/dicts.py
torch/_dynamo/variables/builtin.py
torch/_dynamo/variables/user_defined.py
```

Exit criteria:

- Remove `test/dynamo_expected_failures/CPython313-test_dict-DictTest.test_errors_in_view_containment_check`.
- Add focused Dynamo regression coverage when the fix is semantic.
- The target test passes with `PYTORCH_TEST_WITH_DYNAMO=1`.
- The full `test_dict.py` CPython file has no new real failures.
- Fast CPU validation (affected-file substitute) passes modulo documented
  baseline failures.
- Commit exactly this gate.

### G19: Deque Basics

Target sentinel:

```
CPython313-test_deque-TestBasic.test_basics
```

Target test:

```bash
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu \
PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short \
  test/cpython/v3_13/test_deque.py::TestBasic::test_basics
```

Relevance score: 78.9.

Baseline failure kind:

```
Unsupported method call
```

Likely source areas:

```
torch/_dynamo/variables/lists.py
torch/_dynamo/variables/user_defined.py
torch/_dynamo/variables/builtin.py
```

Exit criteria:

- Remove `test/dynamo_expected_failures/CPython313-test_deque-TestBasic.test_basics`.
- Add focused Dynamo regression coverage when the fix is semantic.
- The target test passes with `PYTORCH_TEST_WITH_DYNAMO=1`.
- The full `test_deque.py` CPython file has no new real failures.
- Fast CPU validation (affected-file substitute) passes modulo documented
  baseline failures.
- Commit exactly this gate.

### G20: Range Iterators

Target sentinel:

```
CPython313-test_range-RangeTest.test_range_iterators
```

Target test:

```bash
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu \
PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short \
  test/cpython/v3_13/test_range.py::RangeTest::test_range_iterators
```

Relevance score: 78.5.

Baseline failure kind:

```
Failed to trace builtin operator
```

Likely source areas:

```
torch/_dynamo/variables/iter.py
torch/_dynamo/variables/lists.py
torch/_dynamo/variables/builtin.py
```

Exit criteria:

- Remove `test/dynamo_expected_failures/CPython313-test_range-RangeTest.test_range_iterators`.
- Add focused Dynamo regression coverage when the fix is semantic.
- The target test passes with `PYTORCH_TEST_WITH_DYNAMO=1`.
- The full `test_range.py` CPython file has no new real failures.
- Fast CPU validation (affected-file substitute) passes modulo documented
  baseline failures.
- Commit exactly this gate.

## Gates (Cycle 3: actionable top-10, G21-G30)

These are the ten highest-ranked rows whose `deferred` column is empty in the
regenerated `cpython_dynamo_expected_failure_relevance.csv`. Gate numbers
continue from G21. Work them in order; triage each for in-scope vs deferred per
`CPYTHON_MIRRORING.md` before implementing. Each gate is one focused commit:
remove only the proven sentinel, add focused Dynamo regression coverage when the
fix is semantic, target test passes under `PYTORCH_TEST_WITH_DYNAMO=1`, affected
CPython file has no new real failures, CPU fast validation (affected-file
substitute) passes modulo baseline, exactly one gate commit.

### G21: Range With User __index__

Status: FIXED (Cycle 3). Classification (a): genuine in-scope object-protocol
gap. CPython applies `PyNumber_Index` (`__index__`) to `range()` arguments and
to slice members in range subscript; Dynamo did neither. `call_range`
(`builtin.py`) returned None for `UserDefinedObjectVariable` args (graph break
"Failed to trace builtin operator range"); after fixing that, `range(10)[:I(5)]`
crashed because `validate_sequence_index` (`object_protocol.py`) coerced a
non-slice index via `__index__` but never the members of a slice key.

Fix: `call_range` coerces `UserDefinedObjectVariable` args via `nb_index_impl`
and retries the constant/symint path; `validate_sequence_index` applies
`__index__` to each non-None slice member (CPython `PySlice_Unpack`), a shared
fix for list/tuple/str/bytes slicing too. `nb_index_impl` already propagates a
raising `__index__` and the non-int `TypeError`.

Files changed:
- `torch/_dynamo/variables/builtin.py` (`call_range`)
- `torch/_dynamo/variables/object_protocol.py` (`validate_sequence_index`)
- `test/dynamo/test_sequence_ops.py` (`TestRangeUserIndex`: args, slice,
  raising `__index__`, non-int `__index__`)

Sentinel removed: `CPython313-test_range-RangeTest.test_user_index_method`.
Validation: target passes; full `test_range.py` 14 passed / 14 skipped;
`test_getitem` + `test_sequence_ops` + `test_misc` 965 passed (no regressions).

Original gate scaffolding preserved below.

Target sentinel:

```
CPython313-test_range-RangeTest.test_user_index_method
```

Target test:

```bash
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu \
PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short \
  test/cpython/v3_13/test_range.py::RangeTest::test_user_index_method
```

Relevance score: 78.5. Baseline failure kind: Failed to trace builtin operator.

Likely source areas:

```
torch/_dynamo/variables/builtin.py
torch/_dynamo/variables/iter.py
torch/_dynamo/variables/lists.py
```

### G22: Sort Reverse Stability

Status: DEFERRED (Cycle 3). Triaged as requiring broad changes (random ->
SymInt routing and/or data-dependent sort), out of scope for a focused gate.
Sentinel LEFT IN PLACE; CSV row tagged `deferred`. No source change made.

Root-cause classification: out of scope (data-dependent control flow on dynamic
values). The CSV failure kind ("Attempted to call function marked as skipped")
is stale. The test builds `data = [(random.randrange(100), i) for i in
range(200)]` and sorts it with `cmp_to_key(my_cmp)` where `my_cmp` does
`(x0 > y0) - (x0 < y0)` on the random ints. Two compounding blockers, both
rooted in how Dynamo models random values:

1. `random.randrange` routes through `RandomVariable` -> `call_random_fn`, which
   deliberately makes the value DYNAMIC (a `RandomValueSource` graph input,
   re-run at runtime) rather than a baked constant. So the ints are
   tensor-backed `UnspecializedPythonVariable`s with no static constant.
   `_handle_insert_op_in_graph`'s unspec branch calls
   `unwrap_unspec_args_kwargs` -> `as_python_constant`, which raises
   `AsPythonConstantNotImplementedError` -> graph break "unimplemented builtin
   op on tensor arguments". A local fallback (keep the op symbolic when no
   constant is available) fixes plain arithmetic (`add`/`mul` work) but then
   `(x0 > y0) - (x0 < y0)` becomes bool-tensor minus bool-tensor ->
   "Subtraction with two bool tensors is not supported": the tensor model
   diverges from Python int/bool semantics.

2. Even with perfect scalar handling, `list.sort(key=...)` over 200 elements
   keyed on dynamic random values requires data-dependent ordering decisions
   Dynamo cannot make at trace time without baking a specific order, which
   contradicts the dynamic-random model (the graph re-runs the random calls).

Making this gate pass would require routing random scalars through
`wrap_symint`/`wrap_symfloat` (the `call_random_fn` TODO) and/or supporting
data-dependent sort -- both broad, cross-cutting changes. Deferred.

Target sentinel:

```
CPython313-test_sort-TestDecorateSortUndecorate.test_reverse_stability
```

Target test:

```bash
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu \
PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short \
  test/cpython/v3_13/test_sort.py::TestDecorateSortUndecorate::test_reverse_stability
```

Relevance score: 78.5.

### G23: List Init

Status: FIXED (Cycle 3). Classification (a): genuine in-scope object-protocol
bug in `ListVariable.__init__`. Source fix landed in the working tree
(uncommitted); target sentinel removed. No collateral sentinels.

Root cause: `ListVariable.call_method` already had a `__init__` branch, but it
did NOT mirror CPython `list___init___impl` (Objects/listobject.c). For
`len(args) == 0` it returned `None` WITHOUT clearing the list, so
`a.__init__()` on `[1, 2, 3]` left the list unchanged and
`assertEqual(a, [])` failed (the `AssertionError` surfaced as an
`Unsupported`/graph-break inside the polyfilled `assertEqual`). The
`len(args) == 1` branch replaced contents via slice assignment (fine for
overwrite) but `len(args) > 1` fell through to `super().call_method` with no
arg-count validation.

Fix: rewrote the branch to mirror CPython `list___init___impl` -- clear the
list, then extend with the optional iterable arg. Routes the extend through
the existing `extend` `call_method` (reusing CPython `list.extend`
fast-path logic) rather than duplicating unpack logic. Validates 0-or-1 args
and 0 kwargs via `raise_args_mismatch`, matching the sibling list-method
branches and the G19 deque `__init__` pattern.

Files changed:
- `torch/_dynamo/variables/lists.py` (`ListVariable.call_method` `__init__`
  branch)
- `test/dynamo/test_sequence_ops.py` (`TestSqConcat`:
  `test_list_reinit_clears`, `test_list_reinit_overwrites`,
  `test_list_reinit_from_iterable`, `test_list_reinit_too_many_args`;
  `TestRangeUserIndex.test_list_reinit_fullgraph`)

Sentinel removed (working tree, uncommitted):
- `CPython313-test_list-ListTest.test_init` (target)

No collateral sentinels: there is no `test_userlist-...test_init` sentinel
(UserList shares the same `list_tests.CommonTest.test_init` but already
passed), tuples are immutable, and `test_deque-TestBasic.test_init` is a
separate deque path covered by G19.

Exact commands run and results (current tree):

```bash
# Target test, sentinel removed -> PASS
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu PYTORCH_TEST_WITH_DYNAMO=1 \
  python test/cpython/v3_13/test_list.py ListTest.test_init
# OK (1 test)

# Whole affected CPython file -> PASS
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu PYTORCH_TEST_WITH_DYNAMO=1 \
  python test/cpython/v3_13/test_list.py
# Ran 65 tests, OK (skipped=14); no failures, no XPASS

# New regression tests -> PASS
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu \
  python test/dynamo/test_sequence_ops.py -k list_reinit
# Ran 5 tests, OK

# Full sequence_ops suite -> PASS
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu \
  python test/dynamo/test_sequence_ops.py
# Ran 135 tests, OK (expected failures=2, pre-existing)

# Nearby Dynamo suite sanity -> PASS
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu \
  python test/dynamo/test_functions.py
# Ran 523 tests, OK (skipped=8, expected failures=2)
```

Risks: low. The extend route reuses the established CPython-mirroring
`list.extend` path; behavior on a single iterable arg is unchanged for the
common case and now additionally clears-first (matching CPython, which is a
no-op for the prior slice-assignment when extend appends to an emptied list).

Next gate: G24.

Original gate scaffolding preserved below.

Target sentinel:

```
CPython313-test_list-ListTest.test_init
```

Target test:

```bash
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu \
PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short \
  test/cpython/v3_13/test_list.py::ListTest::test_init
```

Relevance score: 78.4. Baseline failure kind: Observed exception. (list.__init__
re-init; the list analog of the G19 deque.__init__ fix.)

Likely source areas:

```
torch/_dynamo/variables/lists.py
torch/_dynamo/variables/builtin.py
```

### G24: Dict Splittable Pop

Status: FIXED (Cycle 3). Classification (a): genuine in-scope observable dict
ordering bug in `obj.__dict__` mutation tracking. NOT a split-table internals
problem -- the "splittable" framing is a CPython implementation detail; the
observable thing the test checks is dict insertion-order semantics after a
pop + re-add. Source fix landed in the working tree (uncommitted); target
sentinel removed. No collateral sentinels.

Root cause: `obj.__dict__` key order is observable via `list(obj.__dict__)`.
Dynamo tracks instance-dict mutations in `SideEffects.store_attr_mutations`
(an insertion-ordered dict) and `DunderDictVariable`/`SideEffectsProxyDict`
iterate over it to produce `__dict__` order. When a key was popped
(`store_instance_dict_attr` stores a `DeletedVariable`) and then re-added
(`d[k] = v`), the re-store reused the key's ORIGINAL slot in
`store_attr_mutations` (re-assigning an existing dict key preserves its
position). So for a `C.__dict__` built in-graph from `o.x,o.y,o.z=1,2,3`, then
`o.__dict__.pop('y')` then `o.__dict__['y']=42`, Dynamo produced
`['x','y','z']` while CPython gives `['x','z','y']` (CPython appends a
re-inserted key at the end). The whole test method is compiled with
`error_on_graph_break=True` and `enable_trace_load_build_class=True` (set by
`CPythonTestCase.setUpClass`), so the local `class C` builds in-graph and all
attributes live in the side effects table (`item_dict` empty); the wrong order
surfaced at `self.assertEqual(list(a), ['x', 'z', 'y'])` (test_dict.py:1141) as
`ObservedAssertionErrorError -> Unsupported: Observed exception`.

Fix: in `SideEffects.store_instance_dict_attr`, when storing a non-delete value
for a key whose current side-effects entry is a `DeletedVariable`, drop that
stale (deleted) entry from `store_attr_mutations[item]` (and its
`attr_mutation_kinds`) before re-storing, so the new value re-inserts at the
end of the insertion-ordered dict. This mirrors CPython dict re-insertion
ordering and is scoped to INSTANCE_DICT mutations (the only attribute mutations
whose order is user-observable); generic object setattr is unaffected.

Files changed:
- `torch/_dynamo/side_effects.py` (`store_instance_dict_attr`)
- `test/dynamo/test_dicts.py` (`DunderDictVariableTests`:
  `test_dunder_dict_pop_reinsert_order`,
  `test_dunder_dict_pop_missing_raises_keyerror`,
  `test_dunder_dict_pop_default`; all `fullgraph=True`)

Sentinel removed (working tree, uncommitted):
- `CPython313-test_dict-DictTest.test_splittable_pop` (target)

No collateral sentinels. The sibling split-table tests still fail on other,
out-of-scope blockers and their sentinels are LEFT IN PLACE (verified failing
with sentinel removed):
- `test_splittable_del` (FAILED -- separate del/sizeof path)
- `test_splittable_popitem` (FAILED -- popitem + sizeof)
- `test_splittable_setdefault` (FAILED -- setdefault ordering + sizeof)
- `test_splittable_to_generic_combinedtable` (FAILED -- combined-table internals)
- `test_splittable_update` has no sentinel and already passes (not collateral).

Scope note: a separate, PRE-EXISTING (present on the clean tree, NOT introduced
by this fix) ordering bug remains for the sourced-object case -- popping then
re-adding a key that originally lived in the real `obj.__dict__` (so it is in
`SideEffectsProxyDict.item_dict`, not the side-effects table). There
`SideEffectsProxyDict.__iter__` yields side-effects-table keys before
`item_dict` keys, so a re-added original key sorts to the front instead of the
end. That path is not exercised by this target test (the test builds dicts
fresh in-graph, `item_dict` empty) and fixing it cleanly needs explicit
instance-dict order state on `SideEffects`; deferred as out of scope for this
gate.

Exact commands run and results (current tree):

```bash
# Target test, sentinel removed -> PASS
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu PYTORCH_TEST_WITH_DYNAMO=1 \
  python test/cpython/v3_13/test_dict.py DictTest.test_splittable_pop
# OK (1 test)

# Whole affected CPython file -> PASS
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu PYTORCH_TEST_WITH_DYNAMO=1 \
  python test/cpython/v3_13/test_dict.py
# Ran 112 tests, OK (skipped=42); no failures, no XPASS
# (benign "Exception ignored in __del__" noise from test_store_evilattr,
#  unrelated to this change)

# New regression tests -> PASS
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu \
  python test/dynamo/test_dicts.py DunderDictVariableTests
# Ran 11 tests, OK

# Nearby Dynamo suites sanity -> PASS
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu \
  python test/dynamo/test_dicts.py
# Ran 304 tests, OK (expected failures=1)
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu \
  python test/dynamo/test_modules.py
# Ran 139 tests, OK
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu \
  python test/dynamo/test_functions.py
# Ran 523 tests, OK (skipped=8, expected failures=2)
```

Risks: low. The fix only adjusts ordering for instance-dict keys that were
deleted and re-added (a delete must have stored a `DeletedVariable` first), and
is gated on the existing entry being a `DeletedVariable`, so it never touches
plain overwrite-in-place ordering or generic (non-`__dict__`) attribute
mutation. `lintrunner` clean on the two changed source files.

Next gate: G25.

Original gate scaffolding preserved below.

Target sentinel:

```
CPython313-test_dict-DictTest.test_splittable_pop
```

Target test:

```bash
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu \
PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short \
  test/cpython/v3_13/test_dict.py::DictTest::test_splittable_pop
```

Relevance score: 77.9. Baseline failure kind: Observed exception.

Likely source areas:

```
torch/_dynamo/variables/dicts.py
torch/_dynamo/variables/builtin.py
```

### G25: Sort Bad Decorator

Status: FIXED (Cycle 3). Classification (a): two genuine in-scope
object-protocol gaps. Source fix landed in the working tree (uncommitted);
target sentinel removed (already deleted in the tree at session start) plus one
verified collateral sentinel removed. No commit.

Root cause: the test does
`self.assertRaises(TypeError, data.sort, key=lambda x,y: 0)` where `data =
'...'.split()`. CPython `list.sort` calls `key(item)` with a single argument;
a two-arg key raises `TypeError`, and the test asserts that. Two separate
Dynamo gaps blocked it:

1. Argument-binding mismatches were graph breaks, not observed exceptions.
   Calling a traced Python function with the wrong arity raises `TypeError` in
   `bind_args`/`bind_args_cached` (CPython would raise `TypeError` at call
   time), but the inline path (`symbolic_convert.py:_inline_call`) converted
   that `TypeError` into an `Unsupported` "failed to bind arguments" graph
   break (gb7312, `USER_ERROR`). So `list.sort`'s internal `key(x)` call (and
   any wrong-arity call) could not propagate a catchable `TypeError`.
2. `str.split()` returned an immutable list. `ConstantVariable.call_method`
   wrapped str-method results via `ConstantVariable.create(...)`, which builds
   a `ListVariable` with no `mutation_type` (immutable). So `data.sort()` on a
   `.split()` result fell through `ListVariable.call_method`'s
   `name == "sort" and self.is_mutable()` guard to `super().call_method` ->
   "Unsupported method call `sort`", before the key was ever called. CPython
   `str.split` returns a fresh, caller-owned mutable list.

Fix:
- `torch/_dynamo/variables/functions.py`: new `ArgumentBindingError(TypeError)`
  marker raised by `bind_args_cached` (the four signature-mismatch sites) and
  by `NestedUserFunctionVariable.bind_args` (wrapping the
  `inspect.signature(...).bind` `TypeError`). A plain `TypeError` raised
  elsewhere in binding (e.g. the internal "Only supports regular Python
  functions" limitation) is intentionally NOT marked, so it still graph-breaks.
- `torch/_dynamo/symbolic_convert.py`: the inline path catches
  `ArgumentBindingError` before the generic `TypeError`/graph-break and routes
  it through `exc.raise_observed_exception(TypeError, ...)` with a CPython-like
  `"<name>() <detail>"` message, so user code catching `TypeError` behaves like
  eager. The non-marked `TypeError` path is unchanged.
- `torch/_dynamo/variables/constant.py`: when a `str` method returns a `list`
  (split/rsplit/splitlines), build a new mutable `ListVariable`
  (`ValueMutationNew()`) instead of an immutable one, mirroring CPython's
  fresh caller-owned list.

Files changed:
- `torch/_dynamo/variables/functions.py`
- `torch/_dynamo/symbolic_convert.py`
- `torch/_dynamo/variables/constant.py`
- `test/dynamo/test_functions.py` (new `ArgumentBindingTests` with 6 tests:
  missing/extra positional, unexpected keyword, list.sort bad key,
  str.split-returns-mutable-list, str.split + cmp_to_key; plus rewrote
  `DefaultsTests.test_unsupported_msg_in_bind_args_error` into
  `test_bind_args_mismatch_raises_typeerror` because that test asserted the OLD
  graph-break behavior that is now an observed `TypeError`).

Sentinels removed (working tree, uncommitted):
- `CPython313-test_sort-TestDecorateSortUndecorate.test_baddecorator` (target;
  was already deleted in the tree at session start, kept removed).
- `CPython313-test_sort-TestDecorateSortUndecorate.test_decorated` (collateral;
  verified XPASS only because of the `str.split()` mutable-list fix -- it does
  `data = '...'.split(); data.sort(key=str.lower)` and
  `copy.sort(key=cmp_to_key(my_cmp))`). Verified passing with sentinel removed.

Repro evidence before fix (target test under Dynamo):
`torch._dynamo.exc.Unsupported: Unsupported method call ... method 'sort' of
class 'list' ... call_method ListVariable(length=9) sort [] {'key':
NestedUserFunctionVariable()}`. Isolated probes confirmed both blockers: a
2-arg lambda called with 1 arg gave gb7312 "failed to bind arguments"; a
`"...".split()` list reached `sort` as a non-mutable `ListVariable`.

Exact commands run and results (current tree):

```bash
# Target test under Dynamo, sentinel removed -> PASS
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu PYTORCH_TEST_WITH_DYNAMO=1 \
  python test/cpython/v3_13/test_sort.py TestDecorateSortUndecorate.test_baddecorator
# OK (1 test)

# Whole affected CPython file under Dynamo -> PASS, no XPASS
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu PYTORCH_TEST_WITH_DYNAMO=1 \
  python test/cpython/v3_13/test_sort.py
# Ran 21 tests, OK (skipped=14)

# New + updated regression tests -> PASS
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu \
  python test/dynamo/test_functions.py ArgumentBindingTests \
  DefaultsTests.test_bind_args_mismatch_raises_typeerror
# Ran 7 tests, OK

# Nearby Dynamo suites sanity -> PASS
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu \
  python test/dynamo/test_functions.py
# Ran 529 tests, OK (skipped=8, expected failures=2)
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu \
  python test/dynamo/test_sequence_ops.py
# Ran 137 tests, OK (expected failures=2)
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu \
  python test/dynamo/test_misc.py
# Ran 765 tests, OK (skipped=12, expected failures=4)
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu \
  python test/dynamo/test_exceptions.py
# Ran 76 tests, OK
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu \
  python test/dynamo/test_decorators.py
# Ran 83 tests, OK (skipped=2)
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu PYTORCH_TEST_WITH_DYNAMO=1 \
  python test/cpython/v3_13/test_list.py
# Ran 65 tests, OK (skipped=14)
```

`test/dynamo/test_repros.py` has 2 ERRORS
(`test_linalg_inv_singular_aot_eager_raises`,
`test_linalg_inv_check_errors_preserved_in_aot_graph`) but both are
PRE-EXISTING environment failures ("requires compiling PyTorch with LAPACK";
this is the CPU-only build), unrelated to this change.

Risks: the argument-binding change is the broad one -- it converts wrong-arity
inline calls from a graph break to a propagating observed `TypeError`. This is
strictly more CPython-faithful (eager raises `TypeError` in exactly those
cases) and is scoped to genuine signature mismatches via the
`ArgumentBindingError` marker, leaving internal-limitation `TypeError`s as
graph breaks. The one fallout was `test_unsupported_msg_in_bind_args_error`,
which asserted the old behavior and was rewritten accordingly. The
`str.split` change is strictly more permissive (immutable -> mutable fresh
list). `lintrunner` clean on all four changed files.

Next gate: G26.

Original gate scaffolding preserved below.

Target sentinel:

```
CPython313-test_sort-TestDecorateSortUndecorate.test_baddecorator
```

Target test:

```bash
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu \
PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short \
  test/cpython/v3_13/test_sort.py::TestDecorateSortUndecorate::test_baddecorator
```

Relevance score: 77.9. Baseline failure kind: Unsupported method call.

Likely source areas:

```
torch/_dynamo/variables/lists.py
torch/_dynamo/variables/builtin.py
torch/_dynamo/variables/user_defined.py
```

### G26: Dict Views Mapping

Status: FIXED (Cycle 3). Classification (a): genuine in-scope object-protocol
gap. CPython exposes a read-only `mapping` attribute on dict_keys/dict_values/
dict_items via the `dictview_mapping` getset descriptor
(Objects/dictobject.c), returning a `mappingproxy` of the underlying dict.
Dynamo had no `var_getattr` for `.mapping` on `DictViewVariable`, so
`d.keys().mapping` produced a generic `GetAttrVariable(DictKeysVariable(),
mapping)`; `isinstance(that, mappingproxy)` then graph-broke with
"builtin isinstance() cannot determine type of argument" (gb0175). Source fix
landed in the working tree (uncommitted); target sentinel removed. No
collateral sentinels.

Confirmed failure before change (current tree, sentinel removed):

```
torch._dynamo.exc.Unsupported: builtin isinstance() cannot determine type of
argument. Dynamo doesn't have a rule to determine the type of argument
GetAttrVariable(DictKeysVariable(), mapping); isinstance(
GetAttrVariable(DictKeysVariable(), mapping),
UserDefinedClassVariable(<class 'mappingproxy'>)). From user code at
test_dict.py:178 (self.assertIsInstance(m, mappingproxy)).
```

Root cause: `DictViewVariable` had no `var_getattr` override; `.mapping` fell
through to the base hook, which builds a `GetAttrVariable`. `isinstance` over
a `GetAttrVariable` has no type rule.

Fix: added `DictViewVariable.var_getattr` returning
`MappingProxyVariable(self.dv_dict)` for `name == "mapping"` (delegating other
names to `super().var_getattr`). `MappingProxyVariable` already models
`types.MappingProxyType` correctly (python_type, isinstance, richcompare,
mutation-reflection through the shared `dv_dict` VT), so `isinstance(m,
mappingproxy)` and `m == d` work, and a later `d["foo"]="bar"` is reflected
through the live proxy. This routes through the existing mappingproxy model
rather than a local spot fix.

Files changed:
- `torch/_dynamo/variables/dicts.py` (`DictViewVariable.var_getattr`)
- `test/dynamo/test_dicts.py` (`DictTests.test_dict_view_mapping`,
  `DictTests.test_dict_view_mapping_reflects_mutation`; both `fullgraph=True`)

Sentinel removed (working tree, uncommitted):
- `CPython313-test_dict-DictTest.test_views_mapping` (target)

Exact commands run and results (current tree):

```bash
# Target test, sentinel removed -> PASS
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu PYTORCH_TEST_WITH_DYNAMO=1 \
  python test/cpython/v3_13/test_dict.py DictTest.test_views_mapping
# OK (1 test)

# Whole affected CPython file -> PASS
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu PYTORCH_TEST_WITH_DYNAMO=1 \
  python test/cpython/v3_13/test_dict.py
# Ran 112 tests, OK (skipped=41); no failures, no XPASS
# (benign "Exception ignored in __del__" noise from test_store_evilattr,
#  unrelated to this change)

# New regression tests -> PASS
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu \
  python test/dynamo/test_dicts.py DictTests.test_dict_view_mapping \
  DictTests.test_dict_view_mapping_reflects_mutation
# Ran 2 tests, OK

# Nearby Dynamo suite sanity -> PASS
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu \
  python test/dynamo/test_dicts.py
# Ran 306 tests, OK (expected failures=1, pre-existing)
```

Risks: low. The change only adds a single attribute name (`mapping`) on dict
views, returning the already-modeled `MappingProxyVariable`; all other getattr
names are unchanged (delegated to super). `lintrunner` clean on both changed
files.

Next gate: G27.

Original gate scaffolding preserved below.

Target sentinel:

```
CPython313-test_dict-DictTest.test_views_mapping
```

Target test:

```bash
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu \
PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short \
  test/cpython/v3_13/test_dict.py::DictTest::test_views_mapping
```

Relevance score: 77.8. Baseline failure kind: builtin isinstance() cannot
determine type of argument.

Likely source areas:

```
torch/_dynamo/variables/dicts.py
torch/_dynamo/variables/builtin.py
torch/_dynamo/variables/user_defined.py
```

### G27: Dict Non-Str Single-Instance Setitem

Status: FIXED (Cycle 3). Classification (a): genuine in-scope object-protocol
gap. CPython's instance `__dict__` accepts arbitrary hashable keys when set via
the mapping API (only attribute access via setattr/getattribute requires str);
this is NOT the split/combined-table internals (those are an implementation
detail per CPYTHON_MIRRORING.md "what not to mirror"). The observable behavior
the test checks -- `f.__dict__[1] = 1` succeeds and `list(f.__dict__)` reflects
it -- was wrongly rejected by Dynamo. Source fix landed in the working tree
(uncommitted); target sentinel removed plus one verified collateral sentinel.

Confirmed failure before change (current tree, sentinel removed):

```
AssertionError: Expected str key, got <class 'int'>
  File ".../torch/_dynamo/variables/dicts.py", line 1535, in __setitem__
    raise AssertionError(f"Expected str key, got {type(name)}")
from user code at test_dict.py:1324 (f.__dict__[1] = 1).
```

Root cause: `SideEffectsProxyDict.__setitem__` (`dicts.py`) asserted the
instance-dict key was a `str`. Instance-`__dict__` mutations are tracked in
`SideEffects.store_attr_mutations[item]` (an insertion-ordered dict keyed by the
attribute name) and, crucially, are REPLAYED via
`object_setattr_ignore_descriptor(obj, name, value)` (`utils.py`), which does a
plain `obj.__dict__[name] = value` -- NOT `STORE_ATTR`. So a non-str key is fine
end-to-end: the side-effects table, `has_pending_mutation_of_attr`, `load_attr`,
`__iter__` (wraps each key in `ConstantVariable.create`), and the codegen replay
are all dict-key-generic; only the `__setitem__` str assertion (a holdover from
the str-only attribute model) blocked it. The `str` type annotations on the
side-effects attr APIs are inaccurate but harmless at runtime.

Fix: dropped the str-only assertion in `SideEffectsProxyDict.__setitem__`. The
key is still unwrapped via `_maybe_unwrap_key` (which calls
`HashableTracker.vt.as_python_constant()`, so a non-constant key already raises
and graph-breaks upstream), so only python-constant keys reach
`store_instance_dict_attr`. This is scoped to instance-`__dict__` item
assignment; the separate wholesale `obj.__dict__ = {...}` replacement path
(`get_value___dict__`) still enforces str keys (it materializes via a different
path) and is unchanged.

Note: this is distinct from the wholesale-replacement non-str rejection in
`get_value___dict__` (GENERIC_SETATTR of `__dict__`), which is a different,
out-of-scope path not exercised by this test.

Files changed:
- `torch/_dynamo/variables/dicts.py` (`SideEffectsProxyDict.__setitem__`)
- `test/dynamo/test_dicts.py` (`DunderDictVariableTests`:
  `test_dunder_dict_non_str_key_setitem`,
  `test_dunder_dict_non_str_key_roundtrip`; both `fullgraph=True`)

Sentinels removed (working tree, uncommitted), each verified failing-before
(`Expected str key, got <class 'int'>`) / passing-after:
- `CPython313-test_dict-DictTest.test_object_set_item_single_instance_non_str_key`
  (target)
- `CPython313-test_dict-DictTest.test_splittable_to_generic_combinedtable`
  (collateral -- `d[2] = 2` on an instance dict hits the identical
  `__setitem__` str assertion; the "combined-table" framing is the same
  observable non-str-instance-dict-key behavior, NOT table internals).

Collateral NOT removed (verified still FAILING on other, out-of-scope blockers
with their sentinels removed -- left in place):
- `test_splittable_del` (sys.getsizeof / del path)
- `test_splittable_popitem` (sys.getsizeof + popitem)
- `test_splittable_setdefault` (sys.getsizeof + setdefault ordering)

G29/G30 (`test_reentrant_insertion`, `test_str_nonstr`) sentinels left untouched
(unrelated closure-cell / `__del__` reentrancy blockers; not made to XPASS by
this fix).

Exact commands run and results (current tree):

```bash
# Target test under Dynamo, sentinel removed -> PASS
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu PYTORCH_TEST_WITH_DYNAMO=1 \
  python test/cpython/v3_13/test_dict.py DictTest.test_object_set_item_single_instance_non_str_key
# OK (1 test)

# Collateral test, sentinel removed -> PASS
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu PYTORCH_TEST_WITH_DYNAMO=1 \
  python test/cpython/v3_13/test_dict.py DictTest.test_splittable_to_generic_combinedtable
# OK (1 test)

# Whole affected CPython file -> PASS, no failures/XPASS
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu PYTORCH_TEST_WITH_DYNAMO=1 \
  python test/cpython/v3_13/test_dict.py
# Ran 112 tests, OK (skipped=39)
# (benign "Exception ignored in __del__" noise from test_store_evilattr,
#  unrelated to this change)

# New regression tests -> PASS
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu \
  python test/dynamo/test_dicts.py \
  DunderDictVariableTests.test_dunder_dict_non_str_key_setitem \
  DunderDictVariableTests.test_dunder_dict_non_str_key_roundtrip
# Ran 2 tests, OK

# Nearby Dynamo suite sanity -> PASS
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu \
  python test/dynamo/test_dicts.py
# Ran 308 tests, OK (expected failures=1, pre-existing)
```

Risks: low. The change only removes an over-strict assertion on a path whose
replay (`object_setattr_ignore_descriptor` -> plain `__dict__[name] = value`)
already supports arbitrary hashable keys. Non-constant keys still graph-break
upstream in `_maybe_unwrap_key`. `lintrunner` clean on both changed files.

Next gate: G28.

Original gate scaffolding preserved below.

Target sentinel:

```
CPython313-test_dict-DictTest.test_object_set_item_single_instance_non_str_key
```

Target test:

```bash
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu \
PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short \
  test/cpython/v3_13/test_dict.py::DictTest::test_object_set_item_single_instance_non_str_key
```

Relevance score: 77.7. Baseline failure kind: Expected str key, got
<class 'int'> (splitdict / non-str key store specialization).

Likely source areas:

```
torch/_dynamo/variables/dicts.py
torch/_dynamo/variables/builtin.py
```

### G28: Deque Subclass Basics

Status: FIXED (Cycle 3). Classification (a): genuine in-scope object-protocol
gap in user-defined `collections.deque` subclass construction. NOT a
build-class / closure-cell problem -- the local `class Deque(deque)` traces
fine (the failure is `UserDefinedClassVariable(Deque)` already exists), the gap
is purely modeling construction + the inherited deque C-slot ops on the
subclass instance. Source fix landed in the working tree (uncommitted); target
sentinel removed plus 3 verified collateral sentinels. No commit.

Root cause: three places in Dynamo's user-defined-object construction path knew
about `list`/`dict`/`set`/`tuple` subclasses but not `collections.deque`
subclasses:

1. `SideEffects.cls_supports_mutation_side_effects` (`side_effects.py`)
   allowlists each builtin container's `__getattribute__` slot wrapper.
   `deque.__getattribute__` is its own C slot wrapper (NOT
   `object.__getattribute__`), so a deque subclass failed the check and
   `UserDefinedClassVariable.call_function` fell through to
   `super().call_function` -> "Unsupported function call" on
   `Deque(range(25))`.
2. `UserDefinedClassVariable.supported_c_new_functions` (`user_defined.py`) did
   not list `deque.__new__`, so the `instantiate_user_defined_class_object`
   polyfill's `cls.__new__(cls, ...)` (which resolves to `deque.__new__`) was
   not a supported tp_new for side-effect-tracked construction.
3. `SideEffects.get_variable_cls` had no deque-subclass branch, so a constructed
   deque subclass would have been a plain `UserDefinedObjectVariable` with no
   deque `_base_vt` to delegate the inherited C-slot methods to.

Fix (mirrors the existing `UserDefinedListVariable` pattern):
- Added `deque_methods` (callables in `collections.deque.__dict__`) in
  `utils.py`, the deque analog of `list_methods`/`tuple_methods`.
- Added `UserDefinedDequeVariable(UserDefinedObjectVariable)` in
  `user_defined.py`, backed by a `DequeVariable` `_base_vt` with
  `_base_methods = deque_methods`. The existing `UserDefinedObjectVariable`
  `_base_vt` delegation machinery (call_method, tp_iter, sq_contains, len, etc.)
  then routes the inherited deque ops to `DequeVariable`, which already has the
  CPython-faithful append/appendleft/pop/popleft/extend/__init__/iter behavior
  from G19. Exported it in `variables/__init__.py`.
- Added `collections.deque.__getattribute__` to
  `cls_supports_mutation_side_effects`, `collections.deque.__new__` to
  `supported_c_new_functions` (and to the init_args-ignored set alongside
  dict/set in the `__new__` branch, since deque tp_new takes no construction
  args), and a `issubclass(user_cls, collections.deque)` branch to
  `get_variable_cls` returning `UserDefinedDequeVariable`.

The polyfill does `cls.__new__(cls)` (empty subclass instance via
`deque.__new__`) then `obj.__init__(iterable)`, which delegates to
`DequeVariable.__init__` (clear-then-extend, the G19 path).

Files changed:
- `torch/_dynamo/utils.py` (`deque_methods`)
- `torch/_dynamo/variables/user_defined.py` (`UserDefinedDequeVariable`,
  `supported_c_new_functions`, `__new__` init-args branch, `deque_methods`
  import)
- `torch/_dynamo/variables/__init__.py` (export `UserDefinedDequeVariable`)
- `torch/_dynamo/side_effects.py` (`cls_supports_mutation_side_effects`,
  `get_variable_cls`)
- `test/dynamo/test_sequence_ops.py` (new `TestDequeSubclass`:
  `test_construct_and_basic_ops`, `test_reinit_clears_and_extends`,
  `test_pop_popleft_clear`, `test_construct_empty`, `test_iterate`, all
  `fullgraph=True`; also un-`expectedFailure`-d
  `TestSqConcat.test_user_defined_deque_concat`, which the same fix makes pass)

Sentinels removed (working tree, uncommitted):
- `CPython313-test_deque-TestSubclass.test_basics` (target)
- `CPython313-test_deque-TestSubclass.test_strange_subclass` (collateral:
  `class X(deque)` construction)
- `CPython313-test_deque-TestSequence.test_addmul` (collateral: builds
  `class subclass(deque)`)
- `CPython313-test_deque-TestSequence.test_getitemoverwriteiter` (collateral:
  builds `class T(deque)` with a `__getitem__` override)

Collateral XPASS (test/dynamo, NOT a CPython sentinel): the
`@unittest.expectedFailure` on `TestSqConcat.test_user_defined_deque_concat`
was removed (it now passes -- deque subclass `+` works). Its sibling
`test_user_defined_deque_inplace_concat` still expected-fails on a separate
deque `sq_inplace_concat` "Observed exception" path (out of scope; left
`expectedFailure`).

Repro evidence (current tree, sentinel temporarily removed before fix):

```
torch._dynamo.exc.Unsupported: Unsupported function call
  Explanation: Dynamo does not know how to trace the function
  `<class '__main__.Deque'>`
  Developer debug context: call_function
  UserDefinedClassVariable(<class '__main__.Deque'>) [RangeVariable()] {}
from user code: test/cpython/v3_13/test_deque.py:848 in test_basics
  d = Deque(range(25))
```

Exact commands run and results (current tree):

```bash
# Target test, sentinel removed -> PASS
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu PYTORCH_TEST_WITH_DYNAMO=1 \
  python test/cpython/v3_13/test_deque.py TestSubclass.test_basics
# OK (1 test)

# Whole affected CPython file, 4 sentinels removed -> PASS
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu PYTORCH_TEST_WITH_DYNAMO=1 \
  python test/cpython/v3_13/test_deque.py
# Ran 78 tests, OK (skipped=41); no failures, no XPASS

# New regression tests -> PASS
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu \
  python test/dynamo/test_sequence_ops.py TestDequeSubclass
# Ran 5 tests, OK

# Full sequence_ops suite -> PASS
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu \
  python test/dynamo/test_sequence_ops.py
# Ran 142 tests, OK (expected failures=1, the inplace-concat path)

# Nearby Dynamo suites sanity -> PASS
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu \
  python test/dynamo/test_functions.py   # Ran 529, OK (skipped=8, xfail=2)
  python test/dynamo/test_modules.py     # Ran 139, OK
  python test/dynamo/test_dicts.py       # Ran 308, OK (xfail=1)
  python test/dynamo/test_sets.py        # Ran 176, OK (skipped=1)

# List/tuple subclass paths unaffected by the cls_supports_mutation change
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu PYTORCH_TEST_WITH_DYNAMO=1 \
  python test/cpython/v3_13/test_list.py     # Ran 65, OK (skipped=14)
```

Risk note: `test/cpython/v3_13/test_userlist.py` shows 16 "Unexpected success"
XPASS errors, but these are PRE-EXISTING (verified by `git stash`-ing the G28
source changes and re-running -- still 16 errors on the clean tree). `UserList`
is a `MutableSequence`, not a deque subclass, so it is unrelated to this gate;
those sentinels are LEFT IN PLACE for a separate gate.

Risks: low. The four construction-path edits are each a one-line addition that
extends an existing builtin-container allowlist/branch to include
`collections.deque`; the new `UserDefinedDequeVariable` reuses the established
`_base_vt`/`_base_methods` delegation already proven for list/tuple/dict/set
subclasses, and the deque behavior itself is the already-landed G19
`DequeVariable`. `lintrunner` clean on all changed source files.

Next gate: G29.

Original gate scaffolding preserved below.

Target sentinel:

```
CPython313-test_deque-TestSubclass.test_basics
```

Target test:

```bash
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu \
PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short \
  test/cpython/v3_13/test_deque.py::TestSubclass::test_basics
```

Relevance score: 77.4. Baseline failure kind: Unsupported function call.
(Deque-subclass construction; sibling of the G19 deque work.)

Likely source areas:

```
torch/_dynamo/variables/lists.py
torch/_dynamo/variables/user_defined.py
torch/_dynamo/variables/builtin.py
```

### G29: Dict Reentrant Insertion

Target sentinel:

```
CPython313-test_dict-DictTest.test_reentrant_insertion
```

Target test:

```bash
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu \
PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short \
  test/cpython/v3_13/test_dict.py::DictTest::test_reentrant_insertion
```

Relevance score: 77.3. Baseline failure kind: Read uninitialized cell. (Likely
local class-body closure-cell construction, the deferred-G3 pattern; triage for
out-of-scope before implementing.)

Likely source areas:

```
torch/_dynamo/variables/dicts.py
torch/_dynamo/symbolic_convert.py
torch/_dynamo/variables/user_defined.py
```

#### G29 verdict: DEFERRED (low-relevance build-class / source-backed closure-cell)

Triaged and deferred under the agent_manager.md "Low-Relevance Early Exit" rule.
No source change. Sentinel left in place. Only this Markdown file is modified.

Exact repro command:

```bash
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu PYTORCH_TEST_WITH_DYNAMO=1 \
TORCHDYNAMO_VERBOSE=1 python test/cpython/v3_13/test_dict.py \
  DictTest.test_reentrant_insertion
```

Exact failure (abbreviated traceback):

```
torch._dynamo.exc.Unsupported: Read uninitialized cell
  Explanation: Attempted to read a cell variable that has not been populated yet.
  Developer debug context: CellVariable()
  gb0091

from user code:
   File ".../test/cpython/v3_13/test_dict.py", line 1344, in test_reentrant_insertion
    self.check_reentrant_insertion(mutate)
   File ".../test/cpython/v3_13/test_dict.py", line 1332, in check_reentrant_insertion
    class Mutating:

# Dynamo internal frames:
  builtin.py:1855 call___build_class__
    fn = args[0].get_function(allow_sourced_cells=True)
  functions.py:1973 _get_function_impl
    cell_contents = tx.output.side_effects.load_cell(cell_var)
  side_effects.py:541 load_cell -> unimplemented("Read uninitialized cell")
```

Root-cause classification: OUT-OF-SCOPE build-class / source-backed closure-cell,
NOT an in-scope reentrant-dict-insertion semantic gap.

The test helper `check_reentrant_insertion` (lines 1328-1338) defines a *local*
class inside a method:

```python
def check_reentrant_insertion(self, mutate):
    class Mutating:
        def __del__(self):
            mutate(d)          # free vars: mutate, d
    d = {k: Mutating() for k in 'abcdefghijklmnopqr'}   # d assigned AFTER class def
    for k in list(d):
        d[k] = k
```

The `class Mutating` body closes over the free variables `mutate` and `d`. Dynamo
routes `__build_class__` through `call___build_class__`, which calls
`get_function(allow_sourced_cells=True)`. Building the class function requires
loading its closure cells via `side_effects.load_cell`. The cell for `d` has not
been populated at the point the class is constructed (`d` is bound on the line
*after* the class definition, while the `__del__` closure captures it), so
`load_cell` raises "Read uninitialized cell". This is the deferred-G3
class-body closure-cell construction pattern: making the gate pass would require
source-backed `__build_class__` closure / local class construction machinery,
which the early-exit rule places out of scope absent explicit human approval.

This is corroborated by the same test file: the sibling tests
`DictTest.test_merge_and_mutate` and `DictTest.test_equal_operator_modifying_operand`
are already guarded with
`@unittest.skipIf(TEST_WITH_TORCHDYNAMO, "__build_class__ with closed over objects not supported")`,
confirming closed-over local classes are a known unsupported class here, not a
dict-insertion semantics gap. The reentrant-insertion behavior itself
(`__del__` mutating the dict during item replacement) is never reached because
tracing fails at class construction.

DEFERRED. No `torch/_dynamo` source change attempted. Sentinel
`CPython313-test_dict-DictTest.test_reentrant_insertion` left in place. Working
tree ends with only `coverage.md` modified.

### G30: Dict Str/Non-Str Key

Target sentinel:

```
CPython313-test_dict-DictTest.test_str_nonstr
```

Target test:

```bash
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu \
PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short \
  test/cpython/v3_13/test_dict.py::DictTest::test_str_nonstr
```

Relevance score: 77.3. Baseline failure kind: Read uninitialized cell. (Same
closure-cell pattern as G29; triage for out-of-scope before implementing.)

Likely source areas:

```
torch/_dynamo/variables/dicts.py
torch/_dynamo/symbolic_convert.py
torch/_dynamo/variables/user_defined.py
```

#### G30 verdict: DEFERRED (low-relevance build-class / source-backed closure-cell)

Triaged and deferred under the agent_manager.md "Low-Relevance Early Exit" rule,
same root cause as deferred G29. No source change. Sentinel left in place. Only
this Markdown file is modified.

Exact repro command (current G27-inclusive tree, sentinel temporarily moved):

```bash
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu PYTORCH_TEST_WITH_DYNAMO=1 \
TORCHDYNAMO_VERBOSE=1 python test/cpython/v3_13/test_dict.py \
  DictTest.test_str_nonstr
```

Exact failure (abbreviated traceback):

```
torch._dynamo.exc.Unsupported: Read uninitialized cell
  Explanation: Attempted to read a cell variable that has not been populated yet.
  Developer debug context: CellVariable()
  gb0091

from user code:
   File ".../test/cpython/v3_13/test_dict.py", line 1626, in test_str_nonstr
    class Key3:

# Dynamo internal frames:
  builtin.py:1855 call___build_class__
    fn = args[0].get_function(allow_sourced_cells=True)
  functions.py:1973 _get_function_impl
    cell_contents = tx.output.side_effects.load_cell(cell_var)
  side_effects.py:541 load_cell -> unimplemented("Read uninitialized cell")
```

Root-cause classification: OUT-OF-SCOPE build-class / source-backed closure-cell,
NOT an in-scope str/non-str-key dict semantic gap. Tracing fails at *class
construction*, before any dict lookup/insertion behavior is exercised. The test
defines a local class `Key3` (test_dict.py lines 1626-1635) whose `__eq__`
method closes over the `nonlocal eq_count`:

```python
eq_count = 0
class Key3:
    def __hash__(self):
        return hash('key3')
    def __eq__(self, other):
        nonlocal eq_count
        ...
        eq_count += 1
```

Dynamo routes `__build_class__` through `call___build_class__`
(`builtin.py:1855`), which calls `get_function(allow_sourced_cells=True)`.
Building the class function requires loading its closure cells via
`side_effects.load_cell` (`functions.py:1973` -> `side_effects.py:541`). The
closure cell is not populated through Dynamo's source-backed `__build_class__`
closure path, so `load_cell` raises "Read uninitialized cell". This is the
identical deferred-G29 / deferred-G3 class-body closure-cell construction
pattern; making the gate pass would require source-backed `__build_class__`
closure / local-class-construction machinery, which the early-exit rule places
out of scope absent explicit human approval.

G27 fix did NOT affect this test: the G27 change dropped the str-only assertion
in `SideEffectsProxyDict.__setitem__`, but tracing here fails at class
construction long before any instance/`__dict__` `__setitem__` is reached. The
"str/non-str key" framing in the gate title refers to the CPython dict lookup
optimization the *test body* exercises, but Dynamo never reaches that body.

DEFERRED. No `torch/_dynamo` source change attempted. Sentinel
`CPython313-test_dict-DictTest.test_str_nonstr` left in place. Working tree ends
with only `coverage.md` modified.

## Proposed Gate Changes Awaiting Human Approval

Use this section only when an implementation subagent believes a gate is too
broad, too narrow, stale, or blocked by unrelated infrastructure.

No proposed changes.
