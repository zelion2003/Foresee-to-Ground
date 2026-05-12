

"""Repository-level compatibility helpers for older Torch builds."""

try:
    import torch
    try:
        from torch.utils import _pytree as _torch_pytree  # type: ignore

        if not hasattr(torch.utils, "register_pytree_node"):
            def register_pytree_node(node_type, flatten_fn, unflatten_fn, serialized_type_name=None, *args, **kwargs):
                return _torch_pytree._register_pytree_node(node_type, flatten_fn, unflatten_fn)
            torch.utils.register_pytree_node = register_pytree_node  # type: ignore[attr-defined]
            _torch_pytree.register_pytree_node = register_pytree_node  # type: ignore[attr-defined]
    except Exception:
        pass

    if not hasattr(torch, "compiler"):
        class _DummyCompiler:
            @staticmethod
            def is_compiling():
                return False
        torch.compiler = _DummyCompiler()  # type: ignore[attr-defined]
    elif not hasattr(torch.compiler, "is_compiling"):
        torch.compiler.is_compiling = lambda: False  # type: ignore[attr-defined]
except Exception:
    pass
