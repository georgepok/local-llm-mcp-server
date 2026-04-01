"""pxr stub module for running Isaac Lab Newton branch without Omniverse.

Import this BEFORE any isaaclab imports to mock the pxr (USD) module
that Newton standalone mode doesn't need but the import chain references.
"""
import sys
import types


class PxrStubModule(types.ModuleType):
    """A stub module that returns stubs for any attribute access."""
    def __getattr__(self, name):
        return PxrStubModule(f"{self.__name__}.{name}")
    def __call__(self, *args, **kwargs):
        return None
    def __bool__(self):
        return False
    def __iter__(self):
        return iter([])


def install_pxr_stub():
    """Install a comprehensive pxr stub into sys.modules."""
    pxr = PxrStubModule("pxr")

    submodules = [
        "Usd", "UsdPhysics", "UsdGeom", "UsdUtils", "UsdShade", "UsdLux",
        "Gf", "Sdf", "Vt", "Tf", "Kind",
        "PhysxSchema", "PhysicsSchemaTools",
        "Ar", "Pcp", "Plug",
    ]

    for name in submodules:
        mod = PxrStubModule(f"pxr.{name}")
        setattr(pxr, name, mod)
        sys.modules[f"pxr.{name}"] = mod

    sys.modules["pxr"] = pxr


def install_omni_client_stub():
    """Minimal omni.client stub — just enough for assets.py to not crash."""
    import importlib

    # Check if omni is already a real package
    try:
        importlib.import_module("omni.client")
        return  # already available
    except ImportError:
        pass

    # Create omni as a proper package
    omni_pkg = types.ModuleType("omni")
    omni_pkg.__path__ = []
    omni_pkg.__package__ = "omni"

    # omni.client with Result enum and stat/copy/read_file functions
    client = types.ModuleType("omni.client")

    class Result:
        OK = 0
        ERROR = 1

    class CopyBehavior:
        OVERWRITE = 0

    client.Result = Result
    client.CopyBehavior = CopyBehavior
    client.stat = lambda path: (Result.ERROR, None)  # file not found
    client.copy = lambda src, dst, behavior=None: Result.ERROR
    client.read_file = lambda path: (Result.ERROR, None, b"")

    omni_pkg.client = client
    sys.modules["omni"] = omni_pkg
    sys.modules["omni.client"] = client


if __name__ != "__main__":
    install_pxr_stub()
    install_omni_client_stub()
