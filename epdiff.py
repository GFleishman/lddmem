#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LDDMeM
Large Deformation Diffeomorphic Metric Embedding

Copyright: Greg M. Fleishman
Began: May 2019
"""

import pyfftw
import numpy as np
import scipy.ndimage as ndii

ffter, iffter = None, None

def gu_pinv(a, rcond=1e-15):
    """Return the pseudo-inverse of matrices at every voxel"""

    a = np.asarray(a)
    swap = np.arange(a.ndim)
    swap[[-2, -1]] = swap[[-1, -2]]
    u, s, v = np.linalg.svd(a)
    cutoff = np.maximum.reduce(s, axis=-1, keepdims=True) * rcond
    mask = s > cutoff
    s[mask] = 1. / s[mask]
    s[~mask] = 0
    return np.einsum('...uv,...vw->...uw',
                     np.transpose(v, swap) * s[..., None, :],
                     np.transpose(u, swap))


def initializeFFTW(sh):
    """Initialize the forward and inverse transforms"""
    global ffter, iffter
    sh, ax = tuple(sh), list(range(len(sh)))
    inp = pyfftw.empty_aligned(sh, dtype=np.float64)
    outp_sh = sh[:-1] + (sh[-1]//2+1,) 
    outp = pyfftw.empty_aligned(outp_sh, dtype=np.complex128)
    ffter = pyfftw.FFTW(inp, outp, axes=ax, threads=1)
    iffter = pyfftw.FFTW(outp, inp, axes=ax, direction='FFTW_BACKWARD', threads=1)


def initialize_metric_kernel(a, b, c, d, vox, sh):
    """Precompute the metric kernel and inverse"""

    # define some useful ingredients for later
    dim, oa = len(sh), np.ones(sh)
    sha = (np.diag(sh) - np.identity(dim) + 1).astype(int)

    # if grad of div term is 0, kernel is a scalar, else a Lin Txm
    if b == 0.0:
        L = oa * c
    else:
        L = np.zeros(sh + (dim, dim)) + np.identity(dim) * c

    # compute the scalar (or diagonal) term(s) of kernel
    for i in range(dim):
        q = np.fft.fftfreq(sh[i], d=vox[i])
        X = a * (1 - np.cos(q*2.0*np.pi))
        X = np.reshape(X, sha[i])*oa
        if b == 0.0:
            L += X
        else:
            for j in range(dim):
                L[..., j, j] += X
            L[..., i, i] += b*X/a

    # compute off diagonal terms of kernel
    # TODO: all b != 0 code is out of date and unlikely to work
    if b != 0.0:
        for i in range(dim):
            for j in range(i+1, dim):
                q = np.fft.fftfreq(sh[i], d=vox[i])
                X = np.sin(q*2.0*np.pi*vox[i])
                X1 = np.reshape(X, sha[i])*oa
                q = np.fft.fftfreq(sh[j], d=vox[j])
                X = np.sin(q*2.0*np.pi*vox[j])
                X2 = np.reshape(X, sha[j])*oa
                X = X1*X2*b/(vox[i]*vox[j])
                L[..., i, j] = X
                L[..., j, i] = X

    # I only need half the coefficients (because we're using rfft)
    # compute and store the inverse kernel for regularization
    if b == 0.0:
        L = L[..., :sh[-1]//2+1]**d
        K = L**-1.0
        L = L[..., np.newaxis]
        K = K[..., np.newaxis]
    else:
        L = L[..., :sh[-1]//2+1, :, :]
        cp = np.copy(L)
        for i in range(int(d-1)):
            L = np.einsum('...ij,...jk->...ik', L, cp)
        K = gu_pinv(L)

    return L, K


def fft(f):
    """Return the DFT of the real valued vector field f"""

    global ffter
    sh, d = f.shape[:-1], f.shape[-1]
    F = np.empty(sh[:-1] + (sh[-1]//2+1, d), dtype=np.complex128)
    for i in range(d):
        F[..., i] = ffter(f[..., i])
    return F


def ifft(F, sh):
    """Return the iDFT of the vector field F"""

    global iffter
    f = np.empty(sh, dtype='float64')
    for i in range(sh[-1]):
            f[..., i] = iffter(F[..., i])
    return f


def jacobian(v, vox):
    """Return Jacobian field of vector field v"""

    sh, d = v.shape[:-1], v.shape[-1]
    jac = np.empty(sh + (d, d))
    for i in range(d):
        grad = np.moveaxis(np.array(np.gradient(v[..., i], vox[i])), 0, -1)
        jac[..., i, :] = np.ascontiguousarray(grad)
    return jac


def divergence(v, vox, Dv=None):
    """Return the divergence of vector field v"""

    if Dv is None:
        partials = np.empty_like(v)
        for i in range(v.shape[-1]):
            partials[..., i] = np.gradient(v[..., i], vox[i], axis=i)
        return np.sum(partials, axis=-1)
    else:
        return np.sum(np.diagonal(Dv, axis1=-2, axis2=-1), axis=-1)


def adTranspose(v, m, K, vox, Dv=None, Dm=None):
    """Evaluate the transpose of the negative Jacobi-Lie bracket"""

    global ffter, iffter
    if Dv is None: Dv = jacobian(v, vox)
    if Dm is None: Dm = jacobian(m, vox)
    divv = divergence(v, vox, Dv=Dv)
    permutation = list(range(len(Dv.shape)))
    permutation.append(permutation.pop(-2))
    DvT = np.transpose(Dv, permutation)
    adT = np.einsum('...ij,...j->...i', DvT, m)
    adT += np.einsum('...ij,...j->...i', Dm, v)
    adT += m * divv[..., np.newaxis]
    return - ifft(K * fft(adT), v.shape)


def ad(v, m, vox, Dv=None, Dm=None):
    """Evaluate the negative Jacobi-Lie bracket"""
    
    if Dv is None: Dv = jacobian(v, vox)
    if Dm is None: Dm = jacobian(m, vox)
    ad = np.einsum('...ij,...j->...i', Dv, m)
    return ad - np.einsum('...ij,...j->...i', Dm, v)


def position_array(sh, vox):
    """Return a position array in physical coordinates with shape sh"""
    
    sh, vox = tuple(sh), np.array(vox, dtype=np.float)
    coords = np.array(np.meshgrid(*[range(x) for x in sh], indexing='ij'))
    return vox * np.ascontiguousarray(np.moveaxis(coords, 0, -1))


def apply_transform(img, vox, X, order=1):
    """Return img warped by transform X"""

    # TODO: learn about behavior of map_coordinates w.r.t. memory order
    vox = np.array(vox, dtype=np.float)
    if len(img.shape) == len(vox):
        img = img[..., np.newaxis]
    X *= 1./vox
    ret = np.empty(X.shape[:-1] + (img.shape[-1],))
    X = np.moveaxis(X, -1, 0)
    for i in range(img.shape[-1]):
        ret[..., i] = ndii.map_coordinates(img[..., i], X,
                                           order=order, mode='constant')
    return ret.squeeze()

