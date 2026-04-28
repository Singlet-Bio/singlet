# API Reference

## Core I/O

```{eval-rst}
.. autofunction:: singlepress.write_1pz
.. autofunction:: singlepress.read_1pz
.. autofunction:: singlepress.read_1pz_int
.. autofunction:: singlepress.read_1pz_columns
```

## Metadata & Validation

```{eval-rst}
.. autofunction:: singlepress.info_1pz
.. autofunction:: singlepress.validate_1pz
.. autofunction:: singlepress.colsums_1pz
```

## Lazy Access

```{eval-rst}
.. autofunction:: singlepress.open_1pz
.. autoclass:: singlepress.OnePZFile
   :members:
   :undoc-members:
```

## Dataset Operations

```{eval-rst}
.. autofunction:: singlepress.cbind_1pz
.. autofunction:: singlepress.rbind_1pz
.. autofunction:: singlepress.subset_1pz
.. autofunction:: singlepress.sample_1pz
```

## Normalization

```{eval-rst}
.. autofunction:: singlepress.lognorm
```

## PyTorch Dataloaders

```{eval-rst}
.. automodule:: singlepress.torch
   :members:
   :undoc-members:
   :show-inheritance:
```

## Ecosystem Interoperability

```{eval-rst}
.. automodule:: singlepress.interop
   :members:
   :undoc-members:
   :show-inheritance:
```
