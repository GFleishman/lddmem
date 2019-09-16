#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LDDMeM
Large Deformation Diffeomorphic Metric Embedding

Copyright: Greg M. Fleishman
Began: May 2019
"""

import argparse
from argparse import RawTextHelpFormatter

# VERSION INFORMATION
VERSION = 'LDDMeM - Version: 0.1'

# DESCRIPTION
DESCRIPTION = """
~~~***~~**~*         LDDMeM         *~**~~***~~~
Large Deformation Diffeomorphic Metric Embedding

Embed existing deformation fields computed using
established tools (e.g. ANTs, Elastix, Greedy,
CMTK) in the LDDMM framework.

Finds the closest approximation to a given
deformation, in the least squares sense,
by geodesic shooting on the manifold of
diffeomorphisms. Writes out an initial velocity
field which specifies the geodesic connecting
the identity transform to the embedded transform.
----    ---    ---    ----    ---    ---    ----
"""

# EPILOGUE
EPILOGUE = """
OUTPUTS
    reconV0: initial velocity specifying embedding geodesic
    reconPhiinv: reconstructed transform in LDDMM space
    reconPhi: inverse of reconstructed transform (you get this for free!)
    recon.log: log of parameter values and optimization results
    ^^ NOT IMPLEMENTED YET; CURRENTLY ALL TEXT OUTPUT PRINTED TO STDOUT
"""

# ARGUMENTS
ARGUMENTS = {
'transform':'transform to embed - should be displacement not position field',
'output_directory':'where to write all the amazing results',
'iterations':'iterations per subsampling level, example: 100x75x25',
'--mask':'foreground mask, positive where transform should me matched',
'--time_steps':'number of time steps in geodesic shooting integration; default 6',
'--regularizer':'AxBxCxD for metric (A*divgrad + B*graddiv + C)^D; default 3x0x1x3',
'--regularizer_balance':'S in (1/S^2) * image-match + regularizer; default 0.03',
'--gradient_step':'initial gradient descent step size; default 0.001',
'--optimization_tolerance':'factor by which energy may *increase* between iterations; default 1.15'
}

# OPTIONS
OPTIONS = {a:{'help':ARGUMENTS[a]} for a in ARGUMENTS.keys()}
OPTIONS['--time_steps'] = {**OPTIONS['--time_steps'], 'default':'6'}
OPTIONS['--regularizer'] = {**OPTIONS['--regularizer'], 'default':'3x0x1x3'}
OPTIONS['--regularizer_balance'] = {**OPTIONS['--regularizer_balance'], 'default':'.03'}
OPTIONS['--gradient_step'] = {**OPTIONS['--gradient_step'], 'default':'.001'}
OPTIONS['--optimization_tolerance'] = {**OPTIONS['--optimization_tolerance'], 'default':'1.15'}


# BUILD PARSER
parser = argparse.ArgumentParser(description=DESCRIPTION,
	                             epilog=EPILOGUE,
	                             formatter_class=RawTextHelpFormatter)
for arg in ARGUMENTS.keys():
	parser.add_argument(arg, **OPTIONS[arg])