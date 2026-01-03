# NA

Thanks @sebffischer and @dfalbel for the thoughtful review. Replies
inline:

1.  Package boundary

- I agree it makes sense to keep the transformation in anvil for now
  because it depends on Graph internals and will likely evolve with
  them. As the quickr lowering stabilizes and grows, we can revisit
  splitting out just the transformation API later.

2.  Custom quickr kernels inside PJRT/XLA

- quickr currently only supports base R types, and there’s no efficient
  way (via the public API today) to consume a pointer to an f32 buffer;
  we’d have to materialize an f64 SEXP just to pass data into quickr.
- Also, quickr has no GPU support yet. This is something I would like to
  add in the future, and it’s necessary before it can be a “custom
  kernel” approach.
- I do think quickr could become a good language for writing custom
  kernels with GPU support and full XLA type coverage, but that will
  require changes on the quickr side to accommodate non-atomic SEXP
  types and external-buffer views.
- I’m aligned with the caution here: the execution models, layouts, and
  type systems differ, so I don’t plan to bridge quickr kernels into XLA
  in this PR. If we ever explore that, it should be a separate design
  discussion.
- I like the idea of letting users register custom primitives in anvil.
  For now the existing pattern (new primitive + rules) seems sufficient,
  but I’m open to a follow-up if there’s appetite.

3.  Primitive-level tests vs XLA

- Great suggestion. I can add a small, mostly automated test matrix that
  runs supported primitives through both backends (skipping if quickr is
  not installed). This should also help with the current coverage gap in
  the quickr files.

4.  Literals / GraphLiteral support

- Good catch. The quickr lowering does handle GraphLiteral inputs by
  inlining them as scalars, but this PR was cut before the recent
  GraphLiteral changes landed. I’ll rebase and add explicit tests (e.g.,
  x + 1) to make sure literals work as expected.

5.  jit() backend selection

- I agree this would be a nice UX improvement. I’d prefer to keep it out
  of this PR and instead add a follow-up proposal for a multi-backend
  jit, possibly with an option to set the default.

6.  Assumptions about input types/shapes at runtime

- The compiled quickr function is specialized to the shapes and dtypes
  recorded in the Graph. We emit fixed-size allocations/loops (and use
  quickr::declare) based on those shapes, so runtime inputs are expected
  to match rank and shape. At the moment there is no dynamic-shape
  support or runtime validation; if inputs differ, behavior is undefined
  (likely an error or incorrect result). I can document this more
  explicitly.

Other TODOs you listed - trace_fn with AbstractTensor inputs: agreed,
but out of scope here. - compile_graph_pjrt helper in anvil: also
agreed; happy to tackle separately.

Let me know if you prefer any of these follow-ups to be split into
separate issues before I proceed.
