"""M2: in-place FWHT/mixers at frozen T=4, cis from real angles.

Physics identical to kernel.py. Not a GR recipe. T is not scanned.
"""
from __future__ import annotations

import os

os.environ.setdefault("NUMBA_NUM_THREADS", "4")

import numpy as np
from numba import get_num_threads, njit, prange, set_num_threads
from math import cos, sin, sqrt

THREADS = 4
_WARM = False


def set_threads(n=THREADS):
    set_num_threads(int(n))


@njit(parallel=True, cache=True)
def fwht_inplace(y):
    n = y.shape[0]
    h = 1
    while h < n:
        step = 2 * h
        nblocks = n // step
        for b in prange(nblocks):
            base = b * step
            for j in range(h):
                a = y[base + j]
                bb = y[base + j + h]
                y[base + j] = a + bb
                y[base + j + h] = a - bb
        h = step
    scale = 1.0 / sqrt(float(n))
    for i in prange(n):
        y[i] *= scale


@njit(parallel=True, cache=True)
def cis_mul(psi, ang):
    """psi *= exp(-i ang), ang real."""
    for i in prange(psi.shape[0]):
        c = cos(ang[i])
        s = sin(ang[i])
        a = psi[i]
        psi[i] = (a.real * c + a.imag * s) + 1j * (a.imag * c - a.real * s)


@njit(parallel=True, cache=True)
def cis_mul_scale(psi, hzz, g):
    """psi *= exp(-i g hzz)."""
    for i in prange(psi.shape[0]):
        ang = g * hzz[i]
        c = cos(ang)
        s = sin(ang)
        a = psi[i]
        psi[i] = (a.real * c + a.imag * s) + 1j * (a.imag * c - a.real * s)


@njit(parallel=True, cache=True)
def mix_x_all(psi, beta):
    n = psi.shape[0]
    c = cos(beta)
    s = sin(beta)
    mis = -1j * s
    h = 1
    while h < n:
        step = 2 * h
        nblocks = n // step
        for b in prange(nblocks):
            base = b * step
            for j in range(h):
                a0 = psi[base + j]
                a1 = psi[base + j + h]
                psi[base + j] = c * a0 + mis * a1
                psi[base + j + h] = c * a1 + mis * a0
        h = step


@njit(parallel=True, cache=True)
def energy_zz_x(psi, hzz, b):
    n = psi.shape[0]
    ez = 0.0
    for i in prange(n):
        p = psi[i].real * psi[i].real + psi[i].imag * psi[i].imag
        ez += p * hzz[i]
    ex = 0.0
    h = 1
    while h < n:
        step = 2 * h
        nblocks = n // step
        acc = 0.0
        for blk in prange(nblocks):
            base = blk * step
            local = 0.0
            for j in range(h):
                a0 = psi[base + j]
                a1 = psi[base + j + h]
                local += a0.real * a1.real + a0.imag * a1.imag
            acc += local
        ex += 2.0 * acc
        h = step
    return ez - b * ex


@njit(cache=True)
def nov3_evolve(psi, uz_ang, snap_ang):
    L = snap_ang.shape[0]
    for ell in range(L):
        fwht_inplace(psi)
        cis_mul(psi, uz_ang)
        fwht_inplace(psi)
        cis_mul(psi, snap_ang[ell])
    fwht_inplace(psi)


@njit(cache=True)
def qaoa_evolve(psi, hzz, gamma, beta):
    p = gamma.shape[0]
    for ell in range(p):
        cis_mul_scale(psi, hzz, gamma[ell])
        mix_x_all(psi, beta[ell])


@njit(parallel=True, cache=True)
def overlap2(gs, psi):
    re = 0.0
    im = 0.0
    for i in prange(psi.shape[0]):
        re += gs[i] * psi[i].real
        im += gs[i] * psi[i].imag
    return re * re + im * im


def warmup():
    global _WARM
    if _WARM:
        return
    try:
        set_threads(THREADS)
    except ValueError:
        pass
    psi = np.zeros(16, dtype=np.complex128)
    psi[0] = 1.0
    uz = np.zeros(16)
    snaps = np.zeros((1, 16))
    nov3_evolve(psi, uz, snaps)
    qaoa_evolve(psi.copy(), np.zeros(16), np.array([0.1]), np.array([0.2]))
    energy_zz_x(psi, np.zeros(16), 0.5)
    overlap2(np.ones(16), psi)
    _WARM = True


def thread_meta():
    return {"numba_threads": int(get_num_threads()), "frozen_T": THREADS,
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "")}
