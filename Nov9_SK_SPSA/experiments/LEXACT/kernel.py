"""Exact in-place kernels. float64 / complex128. No truncation.

Candidate only. Does not replace the GR recipe.
"""
from __future__ import annotations

import numpy as np
from numba import njit

_WARM = False


@njit(cache=True)
def fwht_inplace(y):
    n = y.shape[0]
    h = 1
    while h < n:
        step = 2 * h
        nblocks = n // step
        for b in range(nblocks):
            base = b * step
            for j in range(h):
                a = y[base + j]
                bb = y[base + j + h]
                y[base + j] = a + bb
                y[base + j + h] = a - bb
        h = step
    scale = 1.0 / np.sqrt(float(n))
    for i in range(n):
        y[i] *= scale


@njit(cache=True)
def mix_x_all(psi, beta):
    n = psi.shape[0]
    c = np.cos(beta)
    s = np.sin(beta)
    mis = -1j * s
    h = 1
    while h < n:
        step = 2 * h
        nblocks = n // step
        for b in range(nblocks):
            base = b * step
            for j in range(h):
                a0 = psi[base + j]
                a1 = psi[base + j + h]
                psi[base + j] = c * a0 + mis * a1
                psi[base + j + h] = c * a1 + mis * a0
        h = step


@njit(cache=True)
def apply_sumX_real(v, out):
    n = v.shape[0]
    for i in range(n):
        out[i] = 0.0
    h = 1
    while h < n:
        step = 2 * h
        nblocks = n // step
        for b in range(nblocks):
            base = b * step
            for j in range(h):
                out[base + j] += v[base + j + h]
                out[base + j + h] += v[base + j]
        h = step


@njit(cache=True)
def energy_zz_x(psi, hzz, b):
    n = psi.shape[0]
    ez = 0.0
    for i in range(n):
        p = psi[i].real * psi[i].real + psi[i].imag * psi[i].imag
        ez += p * hzz[i]
    ex = 0.0
    h = 1
    while h < n:
        step = 2 * h
        nblocks = n // step
        acc_re = 0.0
        acc_im = 0.0
        for blk in range(nblocks):
            base = blk * step
            for j in range(h):
                a0 = psi[base + j]
                a1 = psi[base + j + h]
                acc_re += a0.real * a1.real + a0.imag * a1.imag
                acc_im += a0.real * a1.imag - a0.imag * a1.real
        ex += 2.0 * acc_re
        h = step
    return ez - b * ex


@njit(cache=True)
def nov3_evolve(psi, uz, snaps):
    L = snaps.shape[0]
    for ell in range(L):
        fwht_inplace(psi)
        for i in range(psi.shape[0]):
            psi[i] *= uz[i]
        fwht_inplace(psi)
        for i in range(psi.shape[0]):
            psi[i] *= snaps[ell, i]
    fwht_inplace(psi)


@njit(cache=True)
def qaoa_evolve(psi, hzz, gamma, beta):
    p = gamma.shape[0]
    for ell in range(p):
        g = gamma[ell]
        for i in range(psi.shape[0]):
            psi[i] *= np.exp(-1j * g * hzz[i])
        mix_x_all(psi, beta[ell])


@njit(cache=True)
def overlap2(gs, psi):
    re = 0.0
    im = 0.0
    for i in range(psi.shape[0]):
        re += gs[i] * psi[i].real
        im += gs[i] * psi[i].imag
    return re * re + im * im


def warmup():
    global _WARM
    if _WARM:
        return
    psi = np.zeros(8, dtype=np.complex128)
    psi[0] = 1.0
    uz = np.ones(8, dtype=np.complex128)
    snaps = np.ones((1, 8), dtype=np.complex128)
    nov3_evolve(psi, uz, snaps)
    qaoa_evolve(psi.copy(), np.zeros(8), np.array([0.1]), np.array([0.2]))
    energy_zz_x(psi, np.zeros(8), 0.5)
    apply_sumX_real(np.ones(8), np.zeros(8))
    overlap2(np.ones(8), psi)
    _WARM = True
