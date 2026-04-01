# PyTorch Integration

```{eval-rst}
.. module:: singlet.torch
```

## Sparse Tensor Creation

```{eval-rst}
.. autofunction:: singlet.torch.to_sparse_csr
.. autofunction:: singlet.torch.to_sparse_coo
.. autofunction:: singlet.torch.from_anndata
```

## Dataset & DataLoader

```{eval-rst}
.. autoclass:: singlet.torch.SpzDataset
   :members:
   :special-members: __len__, __getitem__

.. autoclass:: singlet.torch.DataLoader
   :members:
   :special-members: __iter__, __len__
```
