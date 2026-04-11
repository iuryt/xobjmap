import pytest


def _backends():
    backends = ["numpy"]
    try:
        import jax  # noqa: F401
        backends.append("jax")
    except ImportError:
        pass
    return backends


@pytest.fixture(params=_backends())
def backend(request):
    return request.param
