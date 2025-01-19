#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LDDMEm
Large Deformation Diffeomorphic Metric Embedding

Copyright: Greg M. Fleishman
Began: May 2019
"""

import argparse
from argparse import RawTextHelpFormatter
from lddmem import io
from os.path import splitext

# VERSION INFORMATION
VERSION = 'LDDMEm - Version: 0.0.0'

# DESCRIPTION
DESCRIPTION = """
~~~***~~**~*         LDDMEm         *~**~~***~~~
Large Deformation Diffeomorphic Metric Embedding

Embed existing deformation fields computed using
established tools (e.g. ANTs, Elastix, Greedy,
CMTK) in the LDDMM framework.

Finds the closest approximation to a given
deformation, in the least squares sense,
by geodesic shooting on the manifold of
diffeomorphisms. Returns an initial velocity
field which specifies the geodesic connecting
the identity transform to the embedded transform.

One deformation is fit at a time, however multiple
initial velocities can be combined via the included
Simple Geodesic Regression to obtain a geodesic
which travels through a longer time series.
----    ---    ---    ----    ---    ---    ----
"""

# EPILOGUE
EPILOGUE = """
OUTPUTS
    reconV0: initial velocity specifying embedding geodesic
    reconPhiinv: reconstructed transform in LDDMM space
    reconPhi: inverse of reconstructed transform (you get this for free!)
    recon.log: log of parameter values and optimization results
"""

# ARGUMENTS
ARGUMENTS = {
'transform':'file path displacement vector field, the transform to embed (.nrrd, .nii.gz, or .tiff)',
'transform_spacing':'spatial samping rate of transform in physical units, e.g. 2x1x1',
'multiscale_schedule':'The spatio-temporal multiscale optimization schedule; e.g. 100x2.x3|50x1.x6',
'output_directory':'path to folder where all the amazing results will be written',
'--endpoint_time':'geodesic is integrated to this time point value; default 1.0',
'--regularizer':'AxBxCxD for metric (A*divgrad + B*graddiv + C)^D; default 12x0x1x2',
'--regularizer_balance':'S in (1/S^2) * image-match + regularizer; default 0.03',
'--gradient_step':'initial gradient descent step size; default 0.001',
'--optimization_tolerance':'factor by which energy may *increase* between iterations; default 1.15',
'--threads':'number of threads FFTW should use; default 1',
'--prioritize_speed':'including this flag will use more RAM but run a little faster',
}

# OPTIONS
OPTIONS = {a:{'help':ARGUMENTS[a]} for a in ARGUMENTS.keys()}
OPTIONS['--endpoint_time'] = {**OPTIONS['--endpoint_time'], 'default':'1.0'}
OPTIONS['--regularizer'] = {**OPTIONS['--regularizer'], 'default':'12.0x0x1x2'}
OPTIONS['--regularizer_balance'] = {**OPTIONS['--regularizer_balance'], 'default':'.03'}
OPTIONS['--gradient_step'] = {**OPTIONS['--gradient_step'], 'default':'.001'}
OPTIONS['--optimization_tolerance'] = {**OPTIONS['--optimization_tolerance'], 'default':'1.15'}
OPTIONS['--threads'] = {**OPTIONS['--threads'], 'default':'1'}
OPTIONS['--prioritize_speed'] = {**OPTIONS['--prioritize_speed'], 'action':argparse.BooleanOptionalAction}

# BUILD PARSER
parser = argparse.ArgumentParser(description=DESCRIPTION,
	                             epilog=EPILOGUE,
	                             formatter_class=RawTextHelpFormatter)
for arg in ARGUMENTS.keys():
	parser.add_argument(arg, **OPTIONS[arg])


def parse_command_line_arguments():
    """Read input transform and process command line args
       args are not type or format checked, user must do it right"""

    args = parser.parse_args()
    inputs = {}

    # cli specific inputs
    inputs['extension'] = splitext(args.transform)[1]
    inputs['output_directory'] = args.output_directory
    inputs['log'] = open(inputs['output_directory']+'/lddmem.log', 'w')

    # function inputs
    inputs['transform'] = io.read_field(args.transform, inputs['extension'])
    inputs['transform_spacing'] = tuple(float(x) for x in args.transform_spacing.split('x'))
    inputs['multiscale_schedule'] = []
    for x in args.multiscale_schedule.split('|'):
        y = x.split('x')
        A = int(y[0])
        B = float(y[1]) if y[1] != "None" else None
        C = int(y[2])
        inputs['multiscale_schedule'].append((A, B, C,))
    inputs['endpoint_time'] = float(args.endpoint_time)
    inputs['regularizer'] = tuple(float(x) for x in args.regularizer.split('x'))
    inputs['regularizer_balance'] = float(args.regularizer_balance)
    inputs['gradient_step'] = float(args.gradient_step)
    inputs['optimization_tolerance'] = float(args.optimization_tolerance)
    inputs['threads'] = int(args.threads)
    inputs['prioritize_speed'] = args.prioritize_speed is not None
    return inputs

