from .typechecker import patch_typeguard


def pytest_addoption(parser):
    group = parser.getgroup("jaxtyping")
    group.addoption(
        "--jaxtyping-patch-typeguard",
        action="store_true",
        help="Run jaxtyping's typeguard patch.",
    )


def pytest_configure(config):
    if config.getoption("jaxtyping_patch_typeguard"):
        patch_typeguard()
