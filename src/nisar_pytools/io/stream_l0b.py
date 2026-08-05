"""Stream a subset of a NISAR L0B (RRSD) raw product without downloading it.

Motivation
----------
L0B granules are whole datatakes: a single NISAR raw product is 58-127 GB, and
the echo datasets are ~94 GB each. Most raw-data questions (interference in the
range spectrum, chirp health, receiver behaviour) need a few thousand pulses out
of 871,482, so downloading the granule to read 0.2% of it is wasteful.

The images are chunked and gzipped, so HDF5 can already read an arbitrary
sub-block over HTTP byte-range. What makes a naive attempt collapse is *how* the
file is traversed, not the range reads themselves.

Why the obvious approach fails
------------------------------
An L0B echo dataset is (871482, 26960) with (512, 512) chunks, which is roughly
90,000 chunks per dataset and four such datasets. Calling ``visititems`` (or
anything else that enumerates the file) forces HDF5 to read every object header
and chunk index in the product. Measured against a 112 GB granule that pulled
8.4 GB in 23 minutes without reaching a single echo.

Opening one dataset by path and slicing it does not need the whole index: the
chunk B-tree is searchable, so HDF5 reads only the nodes along the path to the
chunks actually requested. This module therefore never enumerates -- it goes
straight to the dataset path and slices.

Encoding
--------
Echo samples are stored as a compound ``{'r': uint16, 'i': uint16}``. Despite
the unsigned storage type the values are **two's complement**: codes cluster
just above 0 and just below 65536, which are small positive and small negative
numbers respectively. Reading them as unsigned (or subtracting a mid-scale
offset, as one would for offset binary) maps small negative values onto large
positive ones and destroys the signal.

Observed on a NISAR L0B granule: only ~160 distinct codes appear, the most
common being 16, 50, 84 and their complements 65520, 65486, 65452, i.e. a
uniform quantiser with a step of 34 running symmetrically about zero.
"""

from __future__ import annotations

import logging
from typing import Any

import h5py
import numpy as np

log = logging.getLogger(__name__)

#: collections holding L0B raw products, newest maturity first
L0B_SHORT_NAMES = (
    "NISAR_L0B_RRSD_PROVISIONAL_V1",
    "NISAR_L0B_RRSD_BETA_V1",
)

#: modulus of the 16-bit echo codes, used to unwrap them to signed values
UINT16_MODULUS = 65536

_SWATHS = "science/LSAR/RRSD/swaths"


def echo_path(frequency: str = "A", tx: str = "H", rx: str = "H") -> str:
    """HDF5 path of one raw echo dataset.

    Parameters
    ----------
    frequency : str
        ``'A'`` (the wide band) or ``'B'`` (the narrow side band).
    tx, rx : str
        Transmit and receive polarization letters, e.g. ``'H'`` and ``'V'``.

    Returns
    -------
    str
        Path such as ``science/LSAR/RRSD/swaths/frequencyA/txH/rxH/HH``.
    """
    return f"{_SWATHS}/frequency{frequency}/tx{tx}/rx{rx}/{tx}{rx}"


def _ensure_auth(strategy: str = "netrc") -> None:
    """Log in to Earthdata unless the caller already has.

    ``earthaccess.open`` needs an initialised store; searching alone does not
    set one up, so a caller that only ran ``search_data`` would still fail.
    """
    import earthaccess

    if getattr(earthaccess, "__store__", None) is None:
        earthaccess.login(strategy=strategy)


def open_remote_l0b(
    granule: str,
    short_names: tuple[str, ...] = L0B_SHORT_NAMES,
    auth_strategy: str = "netrc",
) -> h5py.File:
    """Open a remote L0B over byte-range as an :class:`h5py.File`.

    No data is transferred beyond what later reads touch. Logs in to Earthdata
    if the caller has not already. The caller is responsible for closing the
    handle.

    Parameters
    ----------
    granule : str
        Granule name, with or without the ``.h5`` suffix.
    short_names : tuple of str
        Collections to search, in order.
    auth_strategy : str
        Passed to ``earthaccess.login`` when no session exists.

    Raises
    ------
    ValueError
        If the granule is not found in any collection.
    """
    import earthaccess

    _ensure_auth(auth_strategy)
    name = granule[:-3] if granule.endswith(".h5") else granule
    for short_name in short_names:
        results = earthaccess.search_data(short_name=short_name, readable_granule_name=name)
        if results:
            break
    else:
        raise ValueError(f"No NISAR L0B granule found for {name} in {short_names}")

    fileobj = earthaccess.open(results[:1])[0]
    return h5py.File(fileobj, "r", driver="fileobj")


def _to_signed(codes: np.ndarray) -> np.ndarray:
    """Reinterpret unsigned 16-bit codes as two's complement, as float32."""
    signed = codes.astype(np.int32)
    signed[signed >= UINT16_MODULUS // 2] -= UINT16_MODULUS
    return signed.astype(np.float32)


def decode_echo(block: np.ndarray) -> np.ndarray:
    """Convert raw compound ``(r, i)`` samples to ``complex64``.

    The stored dtype is unsigned but the values are two's complement, so codes
    at or above 32768 are negative. See the module docstring.

    Parameters
    ----------
    block : numpy.ndarray
        Array with a compound dtype exposing ``'r'`` and ``'i'`` fields.

    Returns
    -------
    numpy.ndarray
        Complex64 array of the same shape.
    """
    if block.dtype.names is None or not {"r", "i"} <= set(block.dtype.names):
        raise ValueError(f"expected a compound (r, i) dtype, got {block.dtype}")
    return (_to_signed(block["r"]) + 1j * _to_signed(block["i"])).astype(np.complex64)


def stream_l0b(
    granule: str,
    pulses: slice | tuple[int, int],
    samples: slice | tuple[int, int] | None = None,
    frequency: str = "A",
    tx: str = "H",
    rx: str = "H",
    decode: bool = True,
    handle: h5py.File | None = None,
) -> np.ndarray:
    """Read a rectangular block of raw echoes from a remote L0B granule.

    Only the chunks overlapping the requested window are transferred. The
    dataset is addressed directly by path, so the product's other datasets and
    their chunk indices are never touched.

    Parameters
    ----------
    granule : str
        Granule name. Ignored when ``handle`` is given.
    pulses : slice or (int, int)
        Azimuth pulse range to read.
    samples : slice or (int, int), optional
        Range sample range. ``None`` reads the full range extent.
    frequency, tx, rx : str
        Which echo dataset to read, see :func:`echo_path`.
    decode : bool
        Convert the compound ``(r, i)`` samples to ``complex64``. Set ``False``
        to get the raw compound array unchanged.
    handle : h5py.File, optional
        An already-open remote handle, to amortise the open across several
        reads. When omitted the granule is opened and closed internally.

    Returns
    -------
    numpy.ndarray
        Complex64 block of shape ``(n_pulses, n_samples)``, or the raw compound
        array when ``decode`` is False.

    Examples
    --------
    >>> block = stream_l0b(granule, pulses=(155645, 157693), samples=(197, 11759))
    >>> block.shape
    (2048, 11562)
    """
    pulse_slice = slice(*pulses) if isinstance(pulses, tuple) else pulses
    sample_slice = slice(*samples) if isinstance(samples, tuple) else samples

    own_handle = handle is None
    fh = open_remote_l0b(granule) if own_handle else handle
    try:
        dataset = fh[echo_path(frequency, tx, rx)]
        block = dataset[pulse_slice, :] if sample_slice is None else \
            dataset[pulse_slice, sample_slice]
    finally:
        if own_handle:
            fh.close()

    log.debug("streamed %s pulses=%s samples=%s", granule, pulse_slice, sample_slice)
    return decode_echo(block) if decode else block


def stream_l0b_metadata(
    granule: str,
    names: tuple[str, ...] = ("UTCtime", "slantRange", "calType",
                              "nominalAcquisitionPRF", "rangeBandwidth",
                              "chirpDuration", "centerFrequency"),
    frequency: str = "A",
    tx: str = "H",
    handle: h5py.File | None = None,
) -> dict[str, Any]:
    """Read the small per-datatake arrays needed to interpret streamed echoes.

    These are the pulse time axis, slant-range axis, calibration flags and
    radar constants. They are small and unchunked, so reading them is cheap.

    Parameters
    ----------
    granule : str
        Granule name. Ignored when ``handle`` is given.
    names : tuple of str
        Dataset names under ``swaths/frequency{F}/tx{T}``.
    frequency, tx : str
        Which transmit group to read from.
    handle : h5py.File, optional
        An already-open remote handle.

    Returns
    -------
    dict
        Name to value; missing datasets are omitted.
    """
    own_handle = handle is None
    fh = open_remote_l0b(granule) if own_handle else handle
    try:
        group = fh[f"{_SWATHS}/frequency{frequency}/tx{tx}"]
        out: dict[str, Any] = {}
        for name in names:
            if name in group:
                out[name] = group[name][()]
    finally:
        if own_handle:
            fh.close()
    return out


def pulses_for_time(utc_time: np.ndarray, start_s: float, stop_s: float) -> tuple[int, int]:
    """Pulse index range covering a seconds-of-day interval.

    L0B ``UTCtime`` is seconds of day, so an L1 product's acquisition window maps
    onto raw pulse indices directly.

    Parameters
    ----------
    utc_time : numpy.ndarray
        The granule's ``UTCtime`` array.
    start_s, stop_s : float
        Interval bounds in seconds of day.

    Returns
    -------
    tuple of int
        ``(first, last)`` pulse indices, suitable for :func:`stream_l0b`.
    """
    inside = np.flatnonzero((utc_time >= start_s) & (utc_time <= stop_s))
    if inside.size == 0:
        raise ValueError(f"no pulses between {start_s} and {stop_s} s")
    return int(inside[0]), int(inside[-1]) + 1
