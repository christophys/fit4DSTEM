# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
"""Figures of the app (rendered to PNG, so they work the same in Jupyter, Voila, Solara and NOMAD)."""
from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def png(fig, dpi=90):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def pattern(frame, log=True, points=None, lattice=None, axes=None, title="", radius=None, vmax_pct=99.8):
    """Diffraction pattern with optional disc positions (points (n, 2)), lattice vectors (c, G) and crystal axes
    (dict(origin, x, y) in pixels)."""
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    a = np.asarray(frame, float)
    img = np.log10(np.clip(a, 0, None) + 1) if log else a
    ax.imshow(img, cmap="magma", vmax=np.percentile(img, vmax_pct), interpolation="nearest")
    if points is not None and len(points):
        p = np.asarray(points)
        if radius:
            for y, x in p:
                ax.add_patch(plt.Circle((x, y), radius, fill=False, ec="cyan", lw=0.6))
        else:
            ax.plot(p[:, 1], p[:, 0], "c+", ms=5)
    if lattice is not None:
        c, G = lattice
        for k, col in ((0, "lime"), (1, "yellow")):
            ax.annotate("", xy=(c[1] + G[1, k], c[0] + G[0, k]), xytext=(c[1], c[0]),
                        arrowprops=dict(arrowstyle="->", color=col, lw=1.5))
    if axes is not None:
        o = axes["origin"]
        for key, col, lab in (("x", "deepskyblue", "in-plane x"), ("y", "orange", "growth [001]")):
            v = axes[key]
            ax.annotate("", xy=(o[1] + v[1], o[0] + v[0]), xytext=(o[1], o[0]),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=2))
            ax.text(o[1] + 1.1 * v[1], o[0] + 1.1 * v[0], lab, color=col, fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    return png(fig)


def compare(data, model, title=""):
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.8))
    v = np.percentile(np.sqrt(np.clip(data, 0, None)), 99.7)
    for ax, im, t in ((axs[0], np.sqrt(np.clip(data, 0, None)), "data (sqrt)"),
                      (axs[1], np.sqrt(np.clip(model, 0, None)), "model (sqrt)")):
        ax.imshow(im, cmap="magma", vmax=v)
        ax.set_title(t, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    r = 2 * (np.sqrt(np.clip(data, 0, None) + 0.375) - np.sqrt(np.clip(model, 0, None) + 0.375))
    axs[2].imshow(r, cmap="RdBu_r", vmin=-4, vmax=4)
    axs[2].set_title("Anscombe residual", fontsize=9)
    axs[2].set_xticks([])
    axs[2].set_yticks([])
    fig.suptitle(title, fontsize=9)
    return png(fig)


def maps(items, step_nm=None, ncols=3, quiver=None, figsize_per=(3.6, 3.0), suptitle=""):
    """items: list of (title, array, cmap, (vmin, vmax) or None). quiver: dict(u, w, step, scale) overlaid on every
    map (u along x, w along growth = up on the map if the growth direction is the scan y axis)."""
    n = len(items)
    ncols = min(ncols, n) or 1
    nrows = int(np.ceil(n / ncols)) or 1
    fig, axs = plt.subplots(nrows, ncols, figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows), squeeze=False)
    for ax in axs.ravel()[n:]:
        ax.axis("off")
    for ax, (t, a, cmap, lim) in zip(axs.ravel(), items):
        a = np.asarray(a, float)
        ext = None
        if step_nm:
            ext = [0, a.shape[1] * step_nm[1], a.shape[0] * step_nm[0], 0]
        if lim is None:
            fin = a[np.isfinite(a)]
            lim = (np.percentile(fin, 1), np.percentile(fin, 99)) if fin.size else (0, 1)
            if cmap in ("RdBu_r", "coolwarm", "seismic", "PuOr"):
                m = max(abs(lim[0]), abs(lim[1]))
                lim = (-m, m)
        im = ax.imshow(a, cmap=cmap, vmin=lim[0], vmax=lim[1], extent=ext, interpolation="nearest")
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(t, fontsize=9)
        if quiver is not None and quiver.get("on_all", False) or (quiver is not None and t == quiver.get("on")):
            u, w = np.asarray(quiver["u"]), np.asarray(quiver["w"])
            s = int(quiver.get("step", 2))
            Y, X = np.mgrid[0:u.shape[0], 0:u.shape[1]]
            if step_nm:
                Yp, Xp = (Y + 0.5) * step_nm[0], (X + 0.5) * step_nm[1]
            else:
                Yp, Xp = Y, X
            ax.quiver(Xp[::s, ::s], Yp[::s, ::s], u[::s, ::s], -w[::s, ::s], color="k", scale=quiver.get("scale"),
                      width=0.006, pivot="mid")
    if suptitle:
        fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout()
    return png(fig)


def distortion_field(D, shape, n=13):
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ys, xs = np.meshgrid(np.linspace(0, shape[0] - 1, n), np.linspace(0, shape[1] - 1, n), indexing="ij")
    u = np.stack([ys.ravel(), xs.ravel()], 1)
    d = D.forward(u) - u
    ax.quiver(u[:, 1], u[:, 0], d[:, 1], -d[:, 0], np.hypot(d[:, 0], d[:, 1]), cmap="viridis", angles="xy")
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_title(f"distortion (max {np.abs(d).max():.2f} px)", fontsize=9)
    return png(fig)


def profile(x, ys, labels, xlabel="", ylabel="", title="", errors=None):
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for k, (y, lab) in enumerate(zip(ys, labels)):
        if errors is not None and errors[k] is not None:
            ax.errorbar(x, y, yerr=errors[k], marker="o", capsize=3, label=lab)
        else:
            ax.plot(x, y, marker="o", label=lab)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return png(fig)
