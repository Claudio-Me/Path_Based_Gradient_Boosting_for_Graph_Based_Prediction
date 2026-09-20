"""Version-tolerant import of the PathBoost estimators.

The algorithm behind this paper is distributed as the ``path_boost`` package
(PyPI: https://pypi.org/project/path_boost/, source:
https://github.com/Claudio-Me/extended_path_boost).

Releases up to 1.6 exposed the very same estimators under the module name
``extended_path_boost``; the package was renamed in 2.0.0 without changing any
of the public class names.  The experiments in this repository were originally
run against the pre-rename module, so both spellings are accepted here and the
scripts work unchanged with either install.
"""

_MODULE_CANDIDATES = ("path_boost", "extended_path_boost")

_INSTALL_HINT = (
    "Could not import the PathBoost package.\n"
    "Install it with:\n"
    "    pip install path_boost==2.1.0\n"
    "or, for the exact version used in the paper:\n"
    "    pip install git+https://github.com/Claudio-Me/extended_path_boost.git"
)


def import_path_boost():
    """Return the installed PathBoost module, whichever name it carries.

    Raises:
        ImportError: if neither ``path_boost`` nor ``extended_path_boost`` is installed.
    """
    import importlib

    errors = []
    for name in _MODULE_CANDIDATES:
        try:
            return importlib.import_module(name)
        except ImportError as exc:  # pragma: no cover - depends on the environment
            errors.append(f"  {name}: {exc}")
    raise ImportError(_INSTALL_HINT + "\n\nTried:\n" + "\n".join(errors))


def get_classifier():
    """Return the ``SequentialPathBoostClassifier`` class (classification experiments)."""
    return import_path_boost().SequentialPathBoostClassifier


def get_regressor():
    """Return the ``SequentialPathBoost`` class (regression experiments)."""
    return import_path_boost().SequentialPathBoost


def get_path_boost():
    """Return the anchor-partitioned ``PathBoost`` class."""
    return import_path_boost().PathBoost


def get_version():
    """Return the installed PathBoost version string, or ``'unknown'``."""
    return getattr(import_path_boost(), "__version__", "unknown")
