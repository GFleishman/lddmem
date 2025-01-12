#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LDDMEm
Large Deformation Diffeomorphic Metric Embedding

Copyright: Greg M. Fleishman
Began: May 2019

This code assumes that the input transform is phiinv relative to
the velocity we wish to estimate. I.e., the velocity is defined in the
coordinates of the image which was labeled as "moving" and we aim to
recover v0 which integrates via advection to the given transform phiinv
"""

import numpy as np
from lddmem import epdiff, io
from lddmem.cli import parse_command_line_arguments
import time
import scipy.ndimage as ndi
from os import makedirs
from os.path import abspath


def initialize_scale_level(
    level, transform, transform_spacing, v0, time_steps, regularizer,
):
    """Resample target transform and initial velocity
    initialize other objects for scale level"""

    fields = {}
    phi = np.copy(transform)
    full_shape = phi.shape
    if level != 0:
        epdiff.initializeFFTW(full_shape[:-1])
        aaL, aaK = epdiff.initialize_metric_kernel(
            2**level, 0, 1, 2,
            transform_spacing, full_shape[:-1],
        )
        phi = epdiff.ifft(aaK * epdiff.fft(phi), full_shape)
        phi = ndi.zoom(phi, (1./2**level,)*3 + (1,), mode='wrap')
    fields['phi'] = phi
    level_shape = phi.shape

    fields['velocity'] = np.zeros((time_steps,) + level_shape)
    if v0 is not None:
        zoom_factors = tuple(x/y for x, y in zip(level_shape[:-1], v0.shape[:-1]))
        fields['velocity'][0] = ndi.zoom(v0, zoom_factors + (1,), mode='nearest')

    epdiff.initializeFFTW(level_shape[:-1])
    fields['spacing'] = np.array(transform_spacing) * 2**level
    fields['position'] = epdiff.position_array(level_shape[:-1], fields['spacing'])
    L, K = epdiff.initialize_metric_kernel(*regularizer, fields['spacing'], level_shape[:-1])
    fields['metric'], fields['inverse_metric'] = L, K
    return fields


def forward_integration(fields, time_steps, compute_phi):
    """Integrate geodesic forward to construct inverse transform"""

    dt = 1./(time_steps-1)
    phi, phiinv = 0, 0
    v, X = fields['velocity'], fields['position']
    for i in range(time_steps-1):
        if compute_phi:
            phi += dt * epdiff.apply_transform(v[i], fields['spacing'], X+phi)
        phiinv -= dt * np.einsum('...ij,...j->...i', epdiff.jacobian(X+phiinv, fields['spacing']), v[i])
        m = epdiff.ifft(fields['metric'] * epdiff.fft(v[i]), v[i].shape)
        dvdt = epdiff.adTranspose(v[i], m, fields['inverse_metric'], fields['spacing'])
        v[i+1] = v[i] + dt * dvdt
    return phiinv, phi


def compute_residual(phi_given, phi_estimated):
    """Compute residual (SSD)"""

    residual = phi_given - phi_estimated
    energy = residual * residual
    residual_magnitudes = np.sqrt(np.sum(energy, axis=-1))
    max_residual = residual_magnitudes.max()
    mean_residual = np.mean(residual_magnitudes)
    residual *= 1./max_residual
    return residual, np.sum(energy), max_residual, mean_residual


def backward_integration(fields, residual, time_steps):
    """Integrate adjoint system backward to get gradient at t0"""

    dt = 1./(time_steps-1)
    v, K = fields['velocity'], fields['inverse_metric']
    _v, _i = np.zeros_like(residual), residual
    for i in range(1, time_steps)[::-1]:
        Dv, D_v = epdiff.jacobian(v[i], fields['spacing']), epdiff.jacobian(_v, fields['spacing'])
        _v += dt * (_i - epdiff.ad(v[i], _v, fields['spacing'], Dv=Dv, Dm=D_v) + \
                          epdiff.adTranspose(_v, v[i], K, fields['spacing'], Dv=D_v, Dm=Dv))
        _i += dt * epdiff.adTranspose(v[i], _i, K, fields['spacing'], Dv=Dv)
    _v = epdiff.ifft(K * epdiff.fft(_v), _v.shape)
    return _v


def lddmem(
    transform,
    transform_spacing,
    iterations,
    time_steps=6,
    regularizer=(12, 0, 1, 2),
    regularizer_balance=0.03,
    gradient_step=0.001,
    optimization_tolerance=1.15,
):
    """
    Embed a smooth deformable transform in the LDDMM framework

    Parameters
    ----------
    transform : numpy.ndarray
        The transform you want to embed. Only accepts 3D or 4D arrays and the
        vector axis should be the last one. That is, if you registered 2D images
        then this transform should have axes (X1, X2, V) for spatial dimensions X.
        If you registered 3D images then this transform should have axes (X1, X2, X3, V).

    transform_spacing : tuple of two or three values
        The voxel sampling rate of the transform. If your transform is a 2D vector field
        this should be two numbers, if your transform is a 3D vector field this shoud
        be three numbers.

    iterations : tuple
        The number of iterations to optimize at each scale. The optimization is multi-scale.
        The length of this tuple indicates the number of scales you wish to use. Scales are
        always a factor of two different along each axis. For example, if iterations==(100x50x25)
        then optimization will run 100 iterations at 4x downsampling along each axis, then
        50 iterations at 2x downsampling along each axis, then 25 iterations at full resolution.

    time_steps : int (default: 6)
        The number of discrete time points at which the velocity flow integration is sampled.

    regularizer : tuple of four numbers (default: (12, 0, 1, 2))
        The Riemannian metric used is A*divgrad + B*graddiv + C)**D
        This input indicates (A, B, C, D). If A, B, and C are all non-zero then this
        is a multiple of an elastic operator. If A and C are non-zero then this is
        a multiple of a diffusion operator.

    regularizer_balance : float (default: 0.03)
        The optimization loss function is: (1/S**2) * image-match + regularizer; this
        parameter is S. Smaller values prioritize more accurate matching but optimizations
        can become unstable.

    gradient_step : float (default: 0.001)
        Initial gradient descent step size. On any iteration that the objective function
        increases more than a specified threshold, the gradient_step is cut in half.

    optimization_tolerance : float greater than 1.0 (default: 1.15)
        A multiplicative factor that determines how much the objective function is allowed to
        increase on any given iteration before the gradient_step is cut in half.

    Returns
    -------
    phiinv : transform

    phi : transform

    fields : extra crap

    log_string : string
    """

    # multiscale loop
    start_time = time.perf_counter()
    fields = {'velocity':(None,)}
    for level, local_iterations in enumerate(iterations):

        # level specific loop
        fields = initialize_scale_level(
            len(iterations)-level-1, transform, transform_spacing,
            fields['velocity'][0],
            time_steps, regularizer,
        )
        local_step = gradient_step
        lowest_v0 = None
        lowest_energy = np.sum(transform**2)
        for iteration in range(local_iterations):

            # only construct forward transform on last iteration of last level
            compute_phi = level == len(iterations)-1 and iteration == local_iterations-1
            phiinv, phi = forward_integration(fields, time_steps, compute_phi)
            residual, energy, max_residual, mean_residual = compute_residual(fields['phi'], phiinv)
            if energy > optimization_tolerance * lowest_energy:
                energy, fields['velocity'][0] = lowest_energy, lowest_v0
                local_step *= 0.5
            elif not compute_phi:
                if energy < lowest_energy:
                    lowest_energy, lowest_v0 = energy, np.copy(fields['velocity'][0])
                _v = backward_integration(fields, residual, time_steps)
            # the gradient descent update
            gradient = fields['velocity'][0] + (1./regularizer_balance**2) * _v
            fields['velocity'][0] -= local_step * gradient

            # record progress
            message = f'level-iteration: {level}-{iteration}\tenergy: {energy:.3f}\t'+ \
                      f'mean|max err: {mean_residual:.3f}|{max_residual:.3f}\t'+\
                      f'time: {time.perf_counter() - start_time:.3f}'
            print(message)
    return phiinv, phi, fields


if __name__ == "__main__":

    # initialize containers, counters, and flags
    inputs = parse_command_line_arguments()
    extension = inputs.pop('extension')
    output_directory = inputs.pop('output_directory')
    log = inputs.pop('log')
    phiinv, phi, fields = lddmem(**inputs)
    makedirs(abspath(output_directory), exist_ok=True)
    io.write_field(phi, output_directory+'/reconPhi', extension)
    io.write_field(phiinv, output_directory+'/reconPhiinv', extension)
    io.write_field(fields['velocity'][0], output_directory+'/reconV0', extension)

