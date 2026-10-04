"""Hyper-TSign model construction."""

from .eventsign_model import EventSignModel


def build_model(cfg):
    """Build the main Hyper-TSign model from a configuration mapping or object."""
    return EventSignModel(cfg)


__all__ = ["EventSignModel", "build_model"]
