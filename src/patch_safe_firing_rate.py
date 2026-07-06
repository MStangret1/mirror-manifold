from __future__ import annotations

"""
Drop-in safe function file. Copy or import this instead of the original plotting-prone helper.
"""

from core_utils import neuron_firing_rate_hdf5_safe as Neuron_firing_rate_HDF5

__all__ = ["Neuron_firing_rate_HDF5"]
