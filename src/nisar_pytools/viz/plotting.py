"""Visualization helpers for NISAR SAR data products."""

from __future__ import annotations

import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.figure import Figure


def plot_amplitude(
    data: xr.DataArray,
    db: bool = True,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "gray",
    ax: plt.Axes | None = None,
    title: str | None = None,
    **kwargs,
) -> Figure:
    """Plot SLC amplitude.

    Parameters
    ----------
    data : xr.DataArray
        Complex SLC or real amplitude array.
    db : bool
        If ``True`` (default), convert to dB (10 * log10).
    vmin, vmax : float, optional
        Color scale limits.
    cmap : str
        Colormap. Default ``"gray"``.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If ``None``, creates a new figure.
    title : str, optional
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    vals = np.abs(data.values)
    if db:
        with np.errstate(divide="ignore"):
            vals = 10 * np.log10(vals**2)
        label = "Amplitude (dB)"
    else:
        label = "Amplitude"

    fig, ax = _get_fig_ax(ax)
    im = ax.imshow(
        vals,
        extent=_extent(data),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
        **kwargs,
    )
    fig.colorbar(im, ax=ax, label=label)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title or "Amplitude")
    return fig


def plot_phase(
    data: xr.DataArray,
    cmap: str = "twilight",
    ax: plt.Axes | None = None,
    title: str | None = None,
    **kwargs,
) -> Figure:
    """Plot phase (wrapped or unwrapped).

    Parameters
    ----------
    data : xr.DataArray
        Phase in radians (real-valued), or complex (phase is extracted).
    cmap : str
        Colormap. Default ``"twilight"`` (cyclic).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    title : str, optional
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    vals = data.values
    if np.iscomplexobj(vals):
        vals = np.angle(vals)

    is_wrapped = float(np.nanmax(vals) - np.nanmin(vals)) <= 2 * np.pi + 0.1

    fig, ax = _get_fig_ax(ax)
    norm = Normalize(vmin=-np.pi, vmax=np.pi) if is_wrapped else None
    im = ax.imshow(
        vals,
        extent=_extent(data),
        cmap=cmap,
        norm=norm,
        aspect="auto",
        **kwargs,
    )
    fig.colorbar(im, ax=ax, label="Phase (rad)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title or "Phase")
    return fig


def plot_interferogram(
    data: xr.DataArray,
    cmap: str = "twilight",
    ax: plt.Axes | None = None,
    title: str | None = None,
    **kwargs,
) -> Figure:
    """Plot a complex interferogram as wrapped phase.

    Parameters
    ----------
    data : xr.DataArray
        Complex interferogram.
    cmap : str
        Colormap. Default ``"twilight"`` (cyclic).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    title : str, optional
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = _get_fig_ax(ax)
    im = ax.imshow(
        np.angle(data.values),
        extent=_extent(data),
        cmap=cmap,
        vmin=-np.pi,
        vmax=np.pi,
        aspect="auto",
        **kwargs,
    )
    fig.colorbar(im, ax=ax, label="Phase (rad)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title or "Interferogram")
    return fig


def plot_coherence(
    data: xr.DataArray,
    cmap: str = "inferno",
    ax: plt.Axes | None = None,
    title: str | None = None,
    **kwargs,
) -> Figure:
    """Plot coherence magnitude.

    Parameters
    ----------
    data : xr.DataArray
        Coherence in [0, 1].
    cmap : str
        Colormap. Default ``"inferno"``.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    title : str, optional
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = _get_fig_ax(ax)
    im = ax.imshow(
        data.values,
        extent=_extent(data),
        cmap=cmap,
        vmin=0,
        vmax=1,
        aspect="auto",
        **kwargs,
    )
    fig.colorbar(im, ax=ax, label="Coherence")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title or "Coherence")
    return fig


def _get_fig_ax(ax: plt.Axes | None) -> tuple[Figure, plt.Axes]:
    """Return (fig, ax), creating a new figure if needed."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    else:
        fig = ax.get_figure()
    return fig, ax


def _extent(da: xr.DataArray) -> list[float] | None:
    """Extract imshow extent [xmin, xmax, ymin, ymax] from coordinates."""
    if "x" not in da.coords or "y" not in da.coords:
        return None
    x = da.x.values
    y = da.y.values
    return [float(x[0]), float(x[-1]), float(y[-1]), float(y[0])]
