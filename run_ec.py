#!/usr/bin/env python3

import sys
import argparse
import time
import logging
logger = logging.getLogger(__name__)

from src import parse, io

def _setup_logging(verbose=0):
    if verbose == 0:
        level = logging.WARNING
    elif verbose == 1:
        level = logging.INFO
    else:
        level = logging.DEBUG

    logging.basicConfig(
            level = level,
            format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt = "%H:%M:%S"
        )

def _parse_args():
    parser = argparse.ArgumentParser(description="Run eigenvector continuation on eigenpair data loaded from a file.")
    parser.add_argument("input_file", type=str, help="Path to the input file.")
    parser.add_argument("-m", "--model", type=str, default="gaussian.Gaussian1d:N=128,V0=-4.0,R=2.0", help="Name of the physics model to run EC with.")
    parser.add_argument("-L", "--parameters", type=str, default="5.0,20.0:20", help="Parameter values at which to predict energies.")
    parser.add_argument("-k", "--knum", type=str, default=None, help="Number of eigenvalues to print per parameter value. Default is all.")
    parser.add_argument("-o", "--out", type=str, default=None, help="Name of file to output energy data to.")
    parser.add_argument("-v", "--verbose", action=count, default=0, help="Increase verbosity (-v, -vv).")

def main():
    pass

if __name__=="__main__":
    main()
