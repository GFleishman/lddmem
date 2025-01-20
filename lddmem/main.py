#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LDDMEm
Large Deformation Diffeomorphic Metric Embedding

Copyright: Greg M. Fleishman
Began: May 2019
"""

import numpy as np
from lddmem import epdiff, io
from lddmem.cli import parse_command_line_arguments
import time
from scipy.ndimage import zoom, gaussian_filter
from os import makedirs
from os.path import abspath


def initialize_geodesic(
    transform, transform_spacing, v0, space_scale, time_steps, regularizer,
    prioritize_speed, threads,
):
    """
    Determine the geodesic objects for a given scale level

    A geodesic is the following dictionary:
        endpoint: the transform we want to match
        velocity_flow: the discritized velocity field flow
        spacing: the voxel sample spacing
        position: the position field of the spatial domain
        metric: the Riemannian metric sampled on the position field
        inverse_metric: the inverse Riemannian metric sampled on the position field
    """

    geodesic = {}
    transform = np.copy(transform)
    if space_scale is not None:
        sigma = tuple(space_scale / x for x in transform_spacing)
        transform = gaussian_filter(transform, sigma, axes=range(transform.ndim-1))
        factors = tuple(min(1., 1/x) for x in sigma)
        transform = zoom(transform, factors + (1,), mode='nearest')
        transform_spacing = 1./np.array(factors) * transform_spacing
    geodesic['endpoint'] = transform
    geodesic['spacing'] = np.array(transform_spacing)

    geodesic['velocity_flow'] = np.zeros((time_steps,) + transform.shape)
    if v0 is not None:
        factors = tuple(x/y for x, y in zip(transform.shape[:-1], v0.shape[:-1]))
        geodesic['velocity_flow'][0] = zoom(v0, factors + (1,), mode='nearest')
    geodesic['position'] = epdiff.position_array(transform.shape[:-1], geodesic['spacing'])

    if prioritize_speed:
        shape = (time_steps,) + transform.shape + (transform.shape[-1],)
        geodesic['jacobian_flow'] = np.empty(shape)

    epdiff.initializeFFTW(transform.shape[:-1], threads)
    L, K = epdiff.initialize_metric_kernel(*regularizer, transform.shape[:-1], geodesic['spacing'])
    geodesic['metric'] = L
    geodesic['inverse_metric'] = K
    return geodesic


# TODO: STILL SCALING ISSUES TO WORK OUT
#       WHEN INITIAL ENERGY IS HIGH, THE TOLERANCE CAN RESULT IN NEVER CUTTING THE STEP
#       ALSO - SETTING REGULARIZER a=12 SCALLED ALL ERRORS DOWN CONSIDERABLY
#       SO, SOMETHING IS AMPLIFYING THE MAGNITUDE OF THE VELOCITY AND THEREFORE TRANSFORM
#       AS IT IS BEING INTEGRATED FORWARD IN TIME
#       LOOK FOR WAYS TO CORRECT SCALING BALANCE
def forward_integration(geodesic, time_steps, endpoint_time, compute_inverse):
    """Integrate geodesic forward to construct inverse transform"""

    dt = endpoint_time/(time_steps-1)
    transform, inverse = 0, 0
    v = geodesic['velocity_flow']
    X, spacing = geodesic['position'], geodesic['spacing']
    L, K = geodesic['metric'], geodesic['inverse_metric']
    for i in range(time_steps-1):
        transform += dt * epdiff.apply_transform(v[i], X+transform, spacing)
        if compute_inverse:
            jacobian = epdiff.jacobian(X+inverse, spacing)
            inverse -= dt * np.einsum('...ij,...j->...i', jacobian, v[i])
        m = epdiff.ifft(L * epdiff.fft(v[i]), v[i].shape)
        Dv = epdiff.jacobian(v[i], spacing)
        if 'jacobian_flow' in geodesic.keys():
            geodesic['jacobian_flow'][i] = Dv
        v[i+1] = v[i] + dt * epdiff.adTranspose(v[i], m, K, spacing, Dv=Dv)
    return transform, inverse


def backward_integration(geodesic, residual, time_steps, endpoint_time):
    """Integrate adjoint system backward to get gradient at t0"""

    dt = endpoint_time/(time_steps-1)
    v, K = geodesic['velocity_flow'], geodesic['inverse_metric']
    spacing = geodesic['spacing']
    _v, _i = np.zeros_like(residual), residual
    for i in range(1, time_steps)[::-1]:
        if 'jacobian_flow' in geodesic.keys() and i < time_steps-1:
            Dv = geodesic['jacobian_flow'][i]
        else:
            Dv = epdiff.jacobian(v[i], spacing)
        D_v = epdiff.jacobian(_v, spacing)
#        _v += dt * (epdiff.ifft(K * epdiff.fft(_i), _i.shape) + \
        _v += dt * (_i + \
                    epdiff.ad(v[i], _v, spacing, Dv=Dv, Dm=D_v) - \
                    epdiff.adTranspose(_v, v[i], K, spacing, Dv=D_v, Dm=Dv))
        _i += dt * epdiff.adTranspose(v[i], _i, K, spacing, Dv=Dv)
    return _v


def compute_residual(transform, embedded_transform, spacing):
    """Compute residual (SSD)"""

    residual = transform - embedded_transform
    energy = residual * residual
    residual_magnitudes = np.sqrt(np.sum(energy, axis=-1))
    max_residual = residual_magnitudes.max()
    mean_residual = residual_magnitudes.mean()
    residual *= spacing.min()/max_residual
    return residual, np.sum(energy), max_residual, mean_residual


def lddmem(
    transform,
    transform_spacing,
    multiscale_schedule,
    endpoint_time=1.,
    regularizer=(12, 0, 1, 2),
    regularizer_balance=0.3,
    gradient_step=0.1,
    optimization_tolerance=1.1,
    prioritize_speed=False,
    threads=1,
):
    """
    Embed a smooth deformable transform in the LDDMM framework
    Optimization can be multiscale in both space and time.

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

    multiscale_schedule : list of tuples, e.g. [(A1, B1, C1), (A2, B2, C2), ...]
        The spatio-temporal multiscale optimization schedule. All tuples in list must
        contain exactly three numbers in this format: (int, float, int). Using variables
        from the example above, A is the number of iterations for that spatio-temporal scale
        level. B is the desired isotropic voxel spacing in the same units as those used
        in transform_spacing; if B is None then there is no spatial down sampling.
        C is the number of time points along which the geodesic path is sampled.
        For example: multiscale_schedule=[(100,2.0,3), (50,2.0,6), (20,None,6)]
        will optimize for 100 iterations on a downsampled transform with 2.0 unit
        spacing along all axes using 3 time points along the geodesic path (including
        the initial and final time points). After, 50 iterations will run at the same
        2.0 spatial sampling but with 6 time points along the geodesic. Finally, 20
        iterations will run at full resolution with 6 time points along the geodesic.
        A must always be greater than or equal to 1.
        B will never result in up sampling. That is, if transform_spacing==(3., 1.,)
        and B==2.0, then transform will be resampled to have spacing==(3., 2.,).
        C must be greater than or equal to 3.

    endpoint_time : strictly positive float (default: 1.)
        The integration is assumed to run from time point 0 to time point endpoint_time.
        This is useful if you plan to combine initial velocities later with Simple
        Geodesic Regression. For example if you had a time series of images collected
        at 6 month intervals, then you would embed the T0 --> T6-months transform
        with endpoint_time==1., then embed the T0 --> T12-months transform with
        endpoint_time==2. That way, the initial velocities you get back are scaled
        properly for Simple Geodesic Regression.

    regularizer : tuple of four numbers (default: (12, 0, 1, 2))
        The Riemannian metric used is A*divgrad + B*graddiv + C)**D
        This input indicates (A, B, C, D). If A, B, and C are all non-zero then this
        is a multiple of an elastic operator. If A and C are non-zero then this is
        a multiple of a diffusion operator.

    regularizer_balance : float (default: 0.03)
        The optimization loss function is: (1/S**2) * field-match + regularizer; this
        parameter is S. Smaller values prioritize more accurate matching but optimizations
        can become unstable.

    gradient_step : float (default: 0.001)
        Initial gradient descent step size. On any iteration that the objective function
        increases more than a specified threshold, the gradient_step is cut in half.

    optimization_tolerance : float greater than 1.0 (default: 1.15)
        A multiplicative factor that determines how much the objective function is allowed to
        increase on any given iteration before the gradient_step is cut in half.

    prioritize_speed : bool (default: False)
        If true, Jacobian of velocity flow will be stored, preventing a redundant calculation
        Jacobian of velocity flow is 3 times larger than the velocity flow.

    threads : int (default: 1)
        The number of threads that FFTW should use

    Returns
    -------
    embedded_transform :

    embedded_transform_inverse :

    initial_velocity :
    """

    # space multiscale loop
    start_time = time.perf_counter()
    local_step = gradient_step
    for level, (iterations, space_scale, time_steps) in enumerate(multiscale_schedule):

        # resample all fields for space level
        geodesic = initialize_geodesic(
            transform,
            transform_spacing,
            geodesic['velocity_flow'][0] if level > 0 else None,
            space_scale,
            time_steps,
            regularizer,
            prioritize_speed,
            threads,
        )

        # level specific loop
        local_step = (gradient_step + local_step) / 2
        lowest_energy = np.inf
        lowest_v0 = np.copy(geodesic['velocity_flow'][0])
        lowest_gradient = np.zeros_like(lowest_v0)
        for iii in range(iterations):

            # forward integrate and compute residual
            compute_inverse = level == len(multiscale_schedule)-1 and iii == iterations-1
            embedded_transform, inverse = forward_integration(
                geodesic, time_steps, endpoint_time, compute_inverse,
            )
            residual, energy, max_residual, mean_residual = compute_residual(
                geodesic['endpoint'], embedded_transform, geodesic['spacing'],
            )

            # if previous step was bad, reset
            if energy > optimization_tolerance * lowest_energy:
                local_step *= 0.5
                energy = lowest_energy
                geodesic['velocity_flow'][0] = lowest_v0 + local_step * lowest_gradient

            # otherwise, backward integrate and update initial velocity
            elif not compute_inverse:
                if energy < lowest_energy:
                    lowest_energy = energy
                    lowest_v0 = np.copy(geodesic['velocity_flow'][0])
                _v = backward_integration(geodesic, residual, time_steps, endpoint_time)
                gradient = geodesic['velocity_flow'][0] - (1./regularizer_balance**2) * _v
                geodesic['velocity_flow'][0] -= local_step * gradient
                if energy < lowest_energy:
                    lowest_gradient = gradient

            # guarantee last iteration uses best result
            if iii == iterations-2:
                energy = lowest_energy
                geodesic['velocity_flow'][0] = lowest_v0

            # record progress
            message = f'scale-time_steps-iteration: ' + \
                      f'{space_scale}-{time_steps}-{iii}    ' + \
                      f'energy: {energy:.3f}    ' + \
                      f'mean|max err: {mean_residual:.3f}|{max_residual:.3f}    ' + \
                      f'time: {time.perf_counter() - start_time:.3f}'
            print(message)

    embedded_transform = zoom(embedded_transform, np.array(transform.shape) / embedded_transform.shape)
    inverse = zoom(inverse, np.array(transform.shape) / inverse.shape)
    x = zoom(geodesic['velocity_flow'][0], np.array(transform.shape) / geodesic['velocity_flow'][0].shape)
    return embedded_transform, inverse, x

    return embedded_transform, inverse, geodesic['velocity_flow'][0]


if __name__ == "__main__":

    # initialize containers, counters, and flags
    inputs = parse_command_line_arguments()
    extension = inputs.pop('extension')
    output_directory = inputs.pop('output_directory')
    log = inputs.pop('log')
    embedded_transform, inverse, initial_velocity = lddmem(**inputs)
    makedirs(abspath(output_directory), exist_ok=True)
    io.write_field(embedded_transform, output_directory+'/embedded_transform', extension)
    io.write_field(inverse, output_directory+'/embedded_transform_inverse', extension)
    io.write_field(initial_velocity, output_directory+'/initial_velocity', extension)

