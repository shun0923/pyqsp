import argparse
import json
import os
import sys

import numpy as np

import pyqsp
from pyqsp import angle_sequence, ham_sim, response
from pyqsp.phases import phase_generators
from pyqsp.poly import polynomial_generators

# -----------------------------------------------------------------------------


class VAction(argparse.Action):
    def __call__(self, parser, args, values, option_string=None):
        curval = getattr(args, self.dest, 0) or 0
        values = values.count('v') + 1
        setattr(args, self.dest, values + curval)

# -----------------------------------------------------------------------------


def CommandLine(args=None, arglist=None):
    '''
    Main command line.  Accepts args, to allow for simple unit testing.
    '''
    import pkg_resources  # part of setuptools

    version = pkg_resources.require("pyqsp")[0].version
    help_text = """usage: pyqsp [options] cmd

Version: {}
Commands:

    poly2angles - compute QSP phase angles for the specified polynomial (use --poly)
    hamsim      - compute QSP phase angles for Hamiltonian simulation using the Jacobi-Anger expansion of exp(-i tau sin(2 theta))
    invert      - compute QSP phase angles for matrix inversion, i.e. a polynomial approximation to 1/a, for given delta and epsilon parameter values
    angles      - generate QSP phase angles for the specified --seqname and --seqargs
    poly        - generate QSP phase angles for the specified --polyname and --polyargs, e.g. sign and threshold polynomials

Examples:

    pyqsp --poly=-1,0,2 poly2angles
    pyqsp --poly=-1,0,2 --plot poly2angles
    pyqsp --signal_operator=Wz --poly=0,0,0,1 --plot  poly2angles
    pyqsp --plot --tau 10 hamsim
    pyqsp --plot --tolerance=0.01 invert
    pyqsp --plot-npts=4000 --plot-positive-only --plot-magnitude --plot --seqargs=1000,1.0e-20 --seqname fpsearch angles
    pyqsp --plot-npts=100 --plot-magnitude --plot --seqargs=23 --seqname erf_step angles
    pyqsp --plot-npts=100 --plot-positive-only --plot --seqargs=23 --seqname erf_step angles
    pyqsp --plot-real-only --plot --polyargs=20,20 --polyname poly_thresh poly
    pyqsp --plot-positive-only --plot --polyargs=19,10 --plot-real-only --polyname poly_sign poly
    pyqsp --plot-positive-only --plot-real-only --plot --polyargs 20,3.5 --polyname gibbs poly
    pyqsp --plot-positive-only --plot-real-only --plot --polyargs 20,0.2,0.9 --polyname efilter poly

""".format(version)

    parser = argparse.ArgumentParser(
        description=help_text,
        formatter_class=argparse.RawTextHelpFormatter)

    def float_list(value):
        return list(map(float, value.split(",")))

    parser.add_argument("cmd", help="command")
    parser.add_argument(
        '-v',
        "--verbose",
        nargs=0,
        help="increase output verbosity (add more -v to increase versbosity)",
        action=VAction,
        dest='verbose')
    parser.add_argument("-o", "--output", help="output filename", default="")
    parser.add_argument(
        "--signal_operator",
        help="QSP sequence signal_operator, either Wx (signal is X rotations) or Wz (signal is Z rotations)",
        type=str,
        default="Wx")
    parser.add_argument(
        "--plot",
        help="generate QSP response plot",
        action="store_true")
    parser.add_argument(
        "--hide-plot",
        help="do not show plot (but it may be saved to a file if --output is specified)",
        action="store_true")
    parser.add_argument(
        "--return-angles",
        help="return QSP phase angles to caller",
        action="store_true")
    parser.add_argument(
        "--poly",
        help="comma delimited list of floating-point coeficients for polynomial, as const, a, a^2, ...",
        action="store",
        type=float_list)
    parser.add_argument(
        "--tau",
        help="time value for Hamiltonian simulation (hamsim command)",
        type=float,
        default=100)
    parser.add_argument(
        "--delta",
        help="parameter for polynomial approximation to the theshold function using erf(delta * x)",
        type=float,
        default=3)
    parser.add_argument(
        "--kappa",
        help="parameter for polynomial approximation to 1/a, valid in the regions 1/kappa < a < 1 and -1 < a < -1/kappa",
        type=float,
        default=3)
    parser.add_argument(
        "--degree",
        help="parameter for polynomial approximation to erf(delta*x)",
        type=int,
        default=3)
    parser.add_argument(
        "--epsilon",
        help="parameter for polynomial approximation to 1/a, giving bound on error",
        type=float,
        default=0.3)
    parser.add_argument(
        "--seqname",
        help="name of QSP phase angle sequence to generate using the 'angles' command, e.g. fpsearch",
        type=str,
        default=None)
    parser.add_argument(
        "--seqargs",
        help="arguments to the phase angles generated by seqname (e.g. length,delta,gamma for fpsearch)",
        action="store",
        type=float_list)
    parser.add_argument(
        "--polyname",
        help="name of polynomial generate using the 'poly' command, e.g. 'sign'",
        type=str,
        default=None)
    parser.add_argument(
        "--polyargs",
        help="arguments to the polynomial generated by poly (e.g. degree,kappa for 'sign')",
        action="store",
        type=float_list)
    parser.add_argument(
        "--plot-magnitude",
        help="when plotting only show magnitude, instead of separate imaginary and real components",
        action="store_true")
    parser.add_argument(
        "--plot-real-only",
        help="when plotting only real component, and not imaginary",
        action="store_true")
    parser.add_argument(
        "--output-json",
        help="output QSP phase angles in JSON format",
        action="store_true")
    parser.add_argument(
        "--plot-positive-only",
        help="when plotting only a-values (x-axis) from 0 to +1, instead of from -1 to +1 ",
        action="store_true")
    parser.add_argument(
        "--plot-tight-y",
        help="when plotting scale y-axis tightly to real part of data",
        action="store_true")
    parser.add_argument(
        "--plot-npts",
        help="number of points to use in plotting",
        type=int,
        default=100)
    parser.add_argument(
        "--tolerance",
        help="error tolerance for phase angle optimizer",
        type=float,
        default=0.1)

    if not args:
        args = parser.parse_args(arglist)

    phiset = None
    plot_args = dict(plot_magnitude=args.plot_magnitude,
                     plot_positive_only=args.plot_positive_only,
                     plot_real_only=args.plot_real_only,
                     plot_tight_y=args.plot_tight_y,
                     npts=args.plot_npts,
                     show=(not args.hide_plot)
                     )

    if args.cmd == "poly2angles":
        coefs = args.poly
        if not coefs:
            print(
                f"[pyqsp.main] must specify polynomial coeffients using --poly, e.g. --poly -1,0,2")
            sys.exit(0)
        if isinstance(coefs, str):
            coefs = list(map(float, coefs.split(",")))
        print(f"[pyqsp] polynomial coefficients={coefs}")
        phiset = angle_sequence.QuantumSignalProcessingPhases(
            coefs, signal_operator=args.signal_operator)
        if args.plot:
            response.PlotQSPResponse(
                phiset, pcoefs=coefs, signal_operator=args.signal_operator, **plot_args)

    elif args.cmd == "hamsim":
        phiset, telapsed = ham_sim.ham_sim(args.tau, 1.0e-4, 1 - 1.0e-4)
        if args.plot:
            response.PlotQSPResponse(phiset, signal_operator="Wz", **plot_args)

    elif args.cmd == "invert":
        pg = pyqsp.poly.PolyOneOverX()
        pcoefs, scale = pg.generate(
            args.kappa,
            args.epsilon,
            return_coef=True,
            ensure_bounded=True,
            return_scale=True)
        phiset = angle_sequence.QuantumSignalProcessingPhases(
            pcoefs, tolerance=args.tolerance)
        if args.plot:
            response.PlotQSPResponse(
                phiset,
                target=lambda x: scale * 1/x,
                signal_operator="Wx",
                title="Inversion",
                **plot_args)

    elif args.cmd == "poly_sign":
        pg = pyqsp.poly.PolySign()
        pcoefs, scale = pg.generate(
            args.degree,
            args.delta,
            ensure_bounded=True,
            return_scale=True)
        phiset = angle_sequence.QuantumSignalProcessingPhases(
            pcoefs, tolerance=args.tolerance)
        if args.plot:
            response.PlotQSPResponse(
                phiset,
                target=lambda x: scale * np.sign(x),
                signal_operator="Wx",
                title="Sign Function",
                **plot_args)

    elif args.cmd == "poly_thresh":
        pg = pyqsp.poly.PolyThreshold()
        pcoefs, scale = pg.generate(
            args.degree,
            args.delta,
            ensure_bounded=True,
            return_scale=True)
        phiset = angle_sequence.QuantumSignalProcessingPhases(
            pcoefs, tolerance=args.tolerance)
        if args.plot:
            response.PlotQSPResponse(
                phiset,
                target=lambda x: scale *
                (np.sign(x + 0.5) - np.sign(x - 0.5)) / 2,
                signal_operator="Wx",
                title="Threshold Function",
                **plot_args)

    elif args.cmd == "poly_rect":
        pg = pyqsp.poly.PolyRect()
        pcoefs, scale = pg.generate(
            args.degree,
            args.delta,
            args.kappa,
            ensure_bounded=True,
            return_scale=True)
        phiset = angle_sequence.QuantumSignalProcessingPhases(
            pcoefs, tolerance=args.tolerance)
        if args.plot:
            response.PlotQSPResponse(
                phiset,
                target=lambda x: scale *
                1 - (np.sign(x + 1/args.kappa) -
                     np.sign(x - 1/args.kappa)) / 2,
                signal_operator="Wx",
                title="Rect Function",
                **plot_args)

    elif args.cmd == "invert_rect":
        pg = pyqsp.poly.PolyOneOverXRect()
        pcoefs, scale = pg.generate(
            args.degree,
            args.delta,
            args.kappa,
            args.epsilon,
            ensure_bounded=True,
            return_scale=True)
        phiset = angle_sequence.QuantumSignalProcessingPhases(
            pcoefs, tolerance=args.tolerance)
        if args.plot:
            response.PlotQSPResponse(
                phiset,
                target=lambda x: scale * 1/x,
                signal_operator="Wx",
                title="Poly Rect * 1/x",
                **plot_args)

    elif args.cmd == "poly":
        if not args.polyname or args.polyname not in polynomial_generators:
            print(
                f'Known polynomial generators: {",".join(polynomial_generators.keys())}')
            return
        pg = polynomial_generators[args.polyname]()
        if not args.polyargs:
            print(pg.help())
            return
        pcoefs = pg.generate(*args.polyargs)
        print(f"[pyqsp] polynomial coefs = {pcoefs}")
        phiset = angle_sequence.QuantumSignalProcessingPhases(
            pcoefs, tolerance=args.tolerance)
        if args.plot:
            response.PlotQSPResponse(
                phiset, pcoefs=pcoefs, signal_operator="Wx", **plot_args)

    elif args.cmd == "angles":
        if not args.seqname or args.seqname not in phase_generators:
            print(
                f'Known phase generators: {",".join(phase_generators.keys())}')
            return
        pg = phase_generators[args.seqname]()
        if not args.seqargs:
            print(pg.help())
            return
        phiset = pg.generate(*args.seqargs)
        print(f"[pysqp] phiset={phiset}")
        if args.plot:
            response.PlotQSPResponse(phiset, signal_operator="Wx", **plot_args)

    else:
        print(f"[pyqsp.main] Unknown command {args.cmd}")
        print(help_text)

    if (phiset is not None):
        if args.return_angles:
            return phiset
        if args.output_json:
            print(
                f"QSP Phase angles (for signal_operator={args.signal_operator}) in JSON format:")
            phiset = phiset.tolist()
            print(json.dumps(phiset, indent=4))
