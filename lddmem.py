#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LDDMeM
Large Deformation Diffeomorphic Metric Embedding

Copyright: Greg M. Fleishman
Began: May 2019

This code assumes that the input transform is phiinv relative to
the velocity we wish to estimate. I.e., the velocity is defined in the
coordinates of the image which was labeled as "moving" and we aim to
recover v0 which integrates via advection to the given transform phiinv
"""

# to get rid of annoying hdf5 warning; comment out if you want
# to read the warning
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import epdiff
import time
import scipy.ndimage as ndi
from interface import parser
import os
from os.path import splitext, abspath


class constants_container:
    """A container class to hold constant value parameters and data"""

    def __init__(self):
        pass

    def set_iterations(self, iterations):
        self.iterations = iterations
    def set_step(self, step):
        self.step = step
    def set_tolerance(self, tolerance):
        self.tolerance = tolerance
    def set_time_steps(self, time_steps):
        self.time_steps = time_steps
        self.dt = 1./(time_steps - 1)
    def set_abcd(self, abcd):
        self.a, self.b, self.c, self.d = abcd
    def set_sigma(self, sigma):
        self.sigma = sigma
    def set_phi(self, phi):
        self.phi = phi
    def set_spacing(self, spacing):
        self.spacing = spacing
    def set_grid(self, grid):
        self.grid = grid
    def set_log(self, log):
        self.log = log


class fields_container:
    """A container class for mutable variables,
    reinitialized with each level"""

    def __init__(self):
        self.phi = None
        self.v = None
        pass

    def set_phi(self, phi):
        self.phi = phi
    def set_spacing(self, spacing):
        self.spacing = spacing
    def set_velocity(self, v):
        self.v = v
    def set_position(self, X):
        self.X = X
    def set_metric(self, L):
        self.L = L
    def set_inverse_metric(self, K):
        self.K = K


def read_image(path):
    """Read an image and its voxel spacing"""

    x = splitext(path)[1]
    ext = x if x != '.gz' else splitext(splitext(path)[0])[1]
    if ext == '.nii':
        import nibabel
        img = nibabel.load(abspath(path))
        img_meta = img.header
        img_data = img.get_data().squeeze()
        img_vox = np.array(img.header.get_zooms()[0:3])
    elif ext == '.nrrd':
        import nrrd
        img_data, img_meta = nrrd.read(abspath(path))
        img_vox = img_meta['spacings'] # TODO: may not work for all images
    return img_data, img_vox, img_meta


def get_crop_region(mask):
    """Determine slices bounding foreground region in mask"""

    labels, num_features = ndi.label(mask > 0, np.ones((3,3,3)))
    vols = ndi.labeled_comprehension(mask > 0, labels, None, sum, int, 0)
    mask = labels == np.argmax(vols) + 1
    slices = ndi.find_objects(mask, 1)[0]
    return slices


def crop(img, slices, extend=10, pad=5):
    """Crop and pad an image using slices from get_crop_region()"""

    R = lambda x: 0 if x < 0 else x
    slices = [slice(R(s.start-extend), s.stop+extend) for s in slices]
    cropped = img[slices[0], slices[1], slices[2]]
    pad = [(pad, pad)]*3
    if len(cropped.shape) == 4: pad += [(0, 0)]
    return np.pad(cropped, pad, mode='constant'), slices


def initialize_parameters(args):
    """Process command line args, read input transform/mask
    all constants stored in a constants_container"""

    params = constants_container()
    params.set_iterations([int(x) for x in args.iterations.split('x')])
    params.set_step(float(args.gradient_step))
    params.set_tolerance(float(args.optimization_tolerance))
    params.set_time_steps(int(args.time_steps))
    params.set_abcd([float(x) for x in args.regularizer.split('x')])
    params.set_sigma(float(args.regularizer_balance))
    params.set_log(open(args.output_directory+'/recon.log', 'w'))
    phi, spacing, meta = read_image(args.transform)
    params.set_grid(phi.shape)
    params.set_spacing(spacing)
    if args.mask:
        mask, _not_used_, meta = read_image(args.mask)
        slices = get_crop_region(mask)
        phi, slices = crop(phi, slices)
    params.set_phi(phi)
    return params, slices, meta


def initialize_scale_level(params, v, level):
    """Resample target transform and initial velocity
    initialize other objects for scale level"""

    fields = fields_container()

    # anti-aliasing required before downsampling
    phi_smooth = params.phi
    if level != 0:
        epdiff.initializeFFTW(params.phi.shape[:-1])
        aaL, aaK = epdiff.initialize_metric_kernel(2**level, 0, 1, 2,
                                                   params.spacing,
                                                   params.phi.shape[:-1])
        phi_smooth = epdiff.ifft( aaK * epdiff.fft(params.phi), params.phi.shape )

    phi = [ndi.zoom(phi_smooth[..., i], 1./2**level, mode='wrap') for i in range(3)]
    fields.set_phi(np.ascontiguousarray(np.moveaxis(np.array(phi), 0, -1)))
    shape = fields.phi.shape
    fields.set_velocity(np.zeros((params.time_steps,) + shape))
    if v is not None:
        zoom_factor = np.array(shape[:-1]) / np.array(v[0].shape[:-1])
        v_ = [ndi.zoom(v[0, ..., i], zoom_factor, mode='nearest') for i in range(3)]
        fields.v[0] = np.ascontiguousarray(np.moveaxis(np.array(v_), 0, -1))
    epdiff.initializeFFTW(shape[:-1])
    fields.set_spacing(params.spacing * 2**level)
    fields.set_position(epdiff.position_array(shape[:-1], fields.spacing))
    L, K = epdiff.initialize_metric_kernel(params.a, params.b,
                                           params.c, params.d,
                                           fields.spacing, shape[:-1])
    fields.set_metric(L)
    fields.set_inverse_metric(K)
    return fields


def forward_integration(params, fields, compute_phi):
    """Integrate geodesic forward to construct inverse transform"""

    phi, phiinv = 0, 0
    v, X = fields.v, fields.X
    for i in range(params.time_steps-1):
        if compute_phi:
            phi += params.dt * epdiff.apply_transform(v[i], fields.spacing, X+phi)
        phiinv -= params.dt * np.einsum('...ij,...j->...i', epdiff.jacobian(X+phiinv, fields.spacing), v[i])
        m = epdiff.ifft(fields.L * epdiff.fft(v[i]), v[i].shape)
        dvdt = epdiff.adTranspose(v[i], m, fields.K, fields.spacing)
        v[i+1] = v[i] + params.dt * dvdt
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


def backward_integration(params, fields, residual):
    """Integrate adjoint system backward to get gradient at t0"""

    v, K = fields.v, fields.K
    _v, _i = np.zeros_like(residual), residual
    for i in range(1, params.time_steps)[::-1]:
        Dv, D_v = epdiff.jacobian(v[i], fields.spacing), epdiff.jacobian(_v, fields.spacing)
        _v += params.dt * (_i - epdiff.ad(v[i], _v, fields.spacing, Dv=Dv, Dm=D_v) + \
                          epdiff.adTranspose(_v, v[i], K, fields.spacing, Dv=D_v, Dm=Dv))
        _i += params.dt * epdiff.adTranspose(v[i], _i, K, fields.spacing, Dv=Dv)
    _v = epdiff.ifft(K * epdiff.fft(_v), _v.shape)
    return _v


def write_field(field, path, transform):
    """Write estimated field"""

    x = splitext(transform)[1]
    ext = x if x != '.gz' else splitext(splitext(transform)[0])[1]
    if ext == '.nii':
        import nibabel
        img = nibabel.load(abspath(transform))
        aff = img.affine
        img = nibabel.Nifti1Image(field, aff)
        nibabel.save(img, path+'.nii.gz')
    elif ext == '.nrrd':
        import nrrd
        img, meta = nrrd.read(transform)
        nrrd.write(field, path+'.nrrd', header=meta)



# initialize containers, counters, and flags
args = parser.parse_args()
os.makedirs(abspath(args.output_directory), exist_ok=True)
params, slices, meta = initialize_parameters(args)
fields = fields_container()
level = len(params.iterations) - 1
compute_phi = False

# record the arguments
print(args)
print(args, file=params.log)

# record initial energy
energy = np.sum(params.phi**2)
message = 'initial energy: ' + str(energy)
print(message)
print(message, file=params.log)

# multiscale loop
start_time = time.clock()
for local_iterations in params.iterations:

    # fields contianer for level and convergence criteria params
    fields = initialize_scale_level(params, fields.v, level)
    iteration, converged, local_step = 0, False, params.step
    lowest_energy, lowest_v0 = (np.finfo(np.float).max-1)/params.tolerance, 0

    # optimization loop for current level
    while iteration < local_iterations and not converged:
        t0 = time.clock()
        # only construct forward transform on last iteration of last level
        if level == 0 and iteration == local_iterations - 1:
            compute_phi = True
        phiinv, phi = forward_integration(params, fields, compute_phi)
        residual, energy, max_residual, mean_residual = compute_residual(fields.phi, phiinv)
        if energy > params.tolerance * lowest_energy:
            energy, fields.v[0] = lowest_energy, lowest_v0
            local_step *= 0.5
            iteration -= 1
        elif not compute_phi:
            if energy < lowest_energy:
                lowest_energy, lowest_v0 = energy, np.copy(fields.v[0])
            _v = backward_integration(params, fields, residual)
        # the gradient descent update
        fields.v[0] = fields.v[0] - (local_step * (fields.v[0] + (1./params.sigma**2) * _v))

        # record progress
        message = 'it: ' + str(iteration) + ', en: ' + str(energy) + \
                  ', max err: ' + str(max_residual) + ', mean err: ' + str(mean_residual) + \
                  ', time: ' + str(time.clock() - t0)
        print(message)
        print(message, file=params.log)
             
        iteration += 1
    level -= 1

message = 'total optimization time: ' + str(time.clock() - start_time)
print(message)
print(message, file=params.log)

# save all outputs
# TODO: this assumes a mask was given
pad = 5    # TODO: magic number here, need better system for padding
phi_out = np.zeros(params.grid)
phi_out[slices[0], slices[1], slices[2]] = phi[pad:-pad, pad:-pad, pad:-pad]
write_field(phi_out, args.output_directory+'/reconPhi', args.transform)

phiinv_out = np.zeros(params.grid)
phiinv_out[slices[0], slices[1], slices[2]] = phiinv[pad:-pad, pad:-pad, pad:-pad]
write_field(phiinv_out, args.output_directory+'/reconPhiinv', args.transform)

v_out = np.zeros(params.grid)
v_out[slices[0], slices[1], slices[2]] = fields.v[0][pad:-pad, pad:-pad, pad:-pad]
write_field(v_out, args.output_directory+'/reconV0', args.transform)

