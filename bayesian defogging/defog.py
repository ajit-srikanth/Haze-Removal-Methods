#!/usr/bin/env python

#
#defog.py
#Author: Gabriel Schwartz (gbs25@drexel.edu)
#License: See LICENSE.
#

import numpy as np
np.seterr(all="ignore")

from cv2 import imread, imwrite

from cmdline import make_arg_parser, GRAB_WIN_NAME

gamma = 1.4

args = make_arg_parser().parse_args()

imgname = args.image.split("/")[-1].split(".")[0]
image = imread(args.image) / 255.


if args.albedo_output is None:
    args.albedo_output = imgname + "_albedo.png"
if args.depth_output is None:
    args.depth_output = imgname + "_depth.png"

if args.airlight:
    airlight = np.array(args.airlight)
elif args.airlight_rect:
    X,Y,W,H = args.airlight_rect
    try:
        airlight_region = image[Y:Y+H, X:X+W]
        airlight = airlight_region.mean(0).mean(0)
        if args.verbose:
            print("Airlight:", airlight)
    except IndexError:
        print("Invalid airlight region given.")
        exit(1)
else:
    print(GRAB_WIN_NAME)
    from util import grab_image_region
    airlight_region = grab_image_region(image, GRAB_WIN_NAME)
    if airlight_region is None:
        print("Airlight selection canceled.")
        exit(1)
    airlight = airlight_region.mean(0).mean(0)
    if args.verbose:
        print("Airlight:", airlight)

from fmrf import FMRF
if args.save_initial_depth:
    from fmrf import compute_initial_depth
    I_n = image / airlight; I_n /= I_n.max()
    d = compute_initial_depth(I_n)
    imwrite("initial_depth.png", 1-(d/d.max()))

fmrf = FMRF(args.apw, args.dpw, args.dpt)

if args.multiscale:
    scales = [0.5, 1.0]
    final_albedo, final_depth = fmrf.factorize_multiscale(
            image, airlight, scales,
            n_outer_iterations = args.n_outer_iterations,
            n_inner_iterations = args.n_inner_iterations,
            verbose = args.verbose)
else:
    final_albedo, final_depth = fmrf.factorize(image, airlight,
            n_outer_iterations = args.n_outer_iterations,
            n_inner_iterations = args.n_inner_iterations,
            verbose = args.verbose)

final_depth /= final_depth.max()
final_albedo = np.power(final_albedo, 1/gamma)

if args.verbose:
    print("Saving albedo to %s..." % args.albedo_output)
imwrite(args.albedo_output, np.clip(final_albedo * 255,0,255).astype(np.uint8))

if args.verbose:
    print("Saving depth to %s..." % args.depth_output)
imwrite(args.depth_output, ((1-(final_depth))*255).astype(np.uint8))
