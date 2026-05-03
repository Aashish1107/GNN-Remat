import sys, types, os

class _Auto(types.ModuleType):
    """Triton stub: attribute access returns _Auto; str/repr returns module name."""
    def __init__(self, n):
        super().__init__(n)
        self.__path__     = []
        self.__file__     = f"<triton_stub {n}>"
        self.__spec__     = None
        self.__package__  = n.rsplit(".", 1)[0] if "." in n else n

    def __getattr__(self, k):
        if k.startswith("__"):
            raise AttributeError(k)
        c = _Auto(f"{self.__name__}.{k}")
        object.__setattr__(self, k, c)
        return c

    def __call__(self, *a, **kw):
        return _Auto(f"{self.__name__}()")

    def __str__(self):  return self.__name__
    def __repr__(self): return f"<triton_stub {self.__name__!r}>"

    # Make it behave like a string where torch.library.utils.get_source
    # calls .endswith() on __file__
    def endswith(self, s):  return str(self.__file__).endswith(s)

_t = _Auto("triton")

# Concrete attributes that torch inspects
_t.language.dtype                      = type("dtype", (), {})
_t.compiler.compiler.CompiledKernel    = type("CompiledKernel", (), {})
_t.backends.compiler.AttrsDescriptor   = type("AttrsDescriptor", (), {})

for _n in [
    "triton","triton.language","triton.language.core","triton.language.extra",
    "triton.compiler","triton.compiler.compiler","triton.runtime",
    "triton.runtime.autotuner","triton.runtime.jit","triton.runtime.cache",
    "triton.runtime.driver","triton.runtime.errors","triton.knobs",
    "triton.backends","triton.backends.compiler","triton.testing","triton.ops",
]:
    sys.modules.setdefault(_n, _Auto(_n))
sys.modules["triton"] = _t

sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages")
sys.path.insert(0, os.getcwd() + "/gnn_remat")

import pytest
args = sys.argv[1:] or [os.getcwd() + "/gnn_remat/tests/", "-v", "--tb=short", "--no-header"]
sys.exit(pytest.main(args))
