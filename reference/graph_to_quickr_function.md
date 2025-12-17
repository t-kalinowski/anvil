# Convert a Graph to a quickr-compiled function

Lowers a supported subset of [`anvil::Graph`](Graph.md) objects to a
plain R function and compiles it with
[`quickr::quick()`](https://rdrr.io/pkg/quickr/man/quick.html).

## Usage

``` r
graph_to_quickr_function(graph)
```

## Arguments

- graph:

  ([`Graph`](Graph.md))  
  Graph to convert.

## Value

(`function`)

## Details

The returned function expects plain R scalars/vectors/arrays (not
[`AnvilTensor`](nv_tensor.md)) and returns plain R values/arrays.

If the graph returns multiple outputs (e.g. a nested list), the compiled
function returns the same structure by packing/unpacking values for
quickr.

At the moment this only supports graphs with a flat (non-nested)
argument list.

Currently supported primitives are: `constant`, `add`, `sub`, `mul`,
`divide`, `negate`, `broadcast_in_dim`, `dot_general`, `transpose`,
`reshape`, `sum`. The code generator currently supports tensors up to
rank 5. Some primitives are more restricted (e.g. `transpose` currently
only handles rank-2 tensors).
