#!/usr/bin/env python3

import sys
import argparse
import time
import logging
logger = logging.getLogger(__name__)

from src import parse, io, ec

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

    # silence EC logs unless verbose is high
    if verbose < 3:
        logging.getLogger("src.ec").setLevel(logging.CRITICAL)

def _parse_args():
    parser = argparse.ArgumentParser(description="Run eigenvector continuation on eigenpair data loaded from a file.")
    parser.add_argument("input_file", type=str, help="Path to the input file.")
    parser.add_argument("-m", "--model", type=str, default=None, help="Name of the physics model to run EC with.")
    parser.add_argument("-L", "--parameters", type=str, default="5.0,20.0:20", help="Parameter values at which to predict energies.")
    parser.add_argument("-k", "--knum", type=int, default=None, help="Number of eigenvalues to print per parameter value. Default is all.")
    parser.add_argument("-o", "--out", type=str, default=None, help="Name of file to output energy data to.")
    parser.add_argument("--vectors", action="store_true", help="Output eigenvectors as well as eigenvalues.")
    parser.add_argument("--dilate", action="store_true", help="Dilate the sample vectors to the predicted volume.")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity (-v, -vv).")
    args = parser.parse_args()
    return args

def _load_data_from_input_file(input_file):
    if not input_file.endswith(".h5"):
        logger.warning(f"{input_file} must be an HDF5 file but does not end in .h5. Attempting to load as an HDF5 file anyway.")
    sample_Ls, sample_energies, sample_eigenvectors, metadata = io.load_energies_from_h5(input_file)
    logger.debug(f"Loaded energy data from HDF5 file.")
    
    # look for model in file metadata
    model = metadata.get("model", None)
    if model is not None: 
        logger.debug(f"Found model = {model} in input file.")
    return sample_Ls, sample_energies, sample_eigenvectors, model

def _get_ec_instance(inputfile_model, args_model): 
    if args_model is not None: 
        if inputfile_model is not None and inputfile_model != args_model:
            logger.warning(f"Model found in input file ({inputfile_model}) does not match model given in command line ({args_model}). " 
                           f"Using {args_model}.")
        model_instance = parse.parse_model_instance(args_model)
    else:
        if inputfile_model is None:
            raise RuntimeError(f"No model given in command line and no model written in input file's metadata.")
        model_instance = parse.parse_model_instance(inputfile_model)
    
    logger.info(f"Using model {type(model_instance).__name__}(" + ','.join(f"{k[1:]}={v} ({type(v).__name__})" for k, v in vars(model_instance).items()) + ").")
    ec_instance = ec.EC(model_instance)
    return ec_instance

def _write_results(args, inputfile_model, Ls, eigenvalues, eigenvectors):
    # create metadata
    metadata = {"timestamp"      : time.strftime("%A, %b %d, %Y %H:%M:%S"),
                "input_file" : args.input_file,
                "model"      : args.model if args.model is not None else inputfile_model,
                "parameters" : args.parameters,
                "command"        : ' '.join(sys.argv)
                }
    # if file ends in h5, use h5py format, otherwise treat everything like .dat
    if args.out.endswith(".h5"):
        io.write_energies_to_h5(args.out, Ls, eigenvalues, eigenvectors if args.vectors else None, metadata)
        logging.debug("Finished writing HDF5 file.")
    else:             
        if args.vectors:
            logger.warning("You selected --vectors but chose an output file that does not support storing full eigenvectors. "
                  "Use a .h5 file if you want full eigenvector output.")
        io.write_energies_to_dat(args.out, Ls, eigenvalues, metadata)
        logging.debug("Finished writing text output file.")

def _print_results(args, Ls, eigenvalues, eigenvectors):
    for i, L in enumerate(Ls):
        print(f"Spectrum at L = {L:.3f}")
        trnc = 5 if args.knum is None else args.knum
        print(f"\t{eigenvalues[i][:trnc]} ... [{len(eigenvalues[i])} energies total]") # if knum isn't specified, truncate to the first 5
        if args.vectors:
            for k, vec in enumerate(eigenvectors[i][:trnc]):
                # print only the first 4 elements of each eigenvector
                # (prints all elements if 4 > len(vec))
                formatted_eigenvecs = [f"{v.real:.4f} + {v.imag:.4f}j" for v in vec[:4]]
                print(f"\tEigenvector {k}: {formatted_eigenvecs} ... [{len(vec)} entries total]")
            print(f"... [{len(eigenvectors[i])} eigenvectors total].")

def main():
    args = _parse_args()
    _setup_logging(args.verbose)
    start = time.time()
    print(f"Input = {args.input_file}. Running EC...")

    # load data from file
    logger.info("Loading data from input file.")
    sample_Ls, sample_energies, sample_eigenvectors, inputfile_model = _load_data_from_input_file(args.input_file)
    
    # grab model instance
    logger.info("Grabbing ec instance.")
    ec_instance = _get_ec_instance(inputfile_model, args.model)

    # sample ec with data read from file
    logger.info("Sampling ec instance with data read from file.")
    ec_instance.sample_from_given_data(sample_Ls, sample_energies, sample_eigenvectors)

    # parse parameters and predict from ec
    logger.info(f"Parsing parameters {args.parameters}.")
    predict_Ls = parse.parse_parameter_values(args.parameters)
    logger.info("Using EC to predict energies at parameters.")
    predicted_energies, predicted_eigenvectors = ec_instance.ec_predict(predict_Ls, k_num=args.knum, dilate=args.dilate)

    # write eigenpair data to file if requested
    if args.out:
        logger.info(f"Writing results to {args.out}.")
        _write_results(args, inputfile_model, predict_Ls, predicted_energies, predicted_eigenvectors)

    # print results
    _print_results(args, predict_Ls, predicted_energies, predicted_eigenvectors)
    end = time.time()
    print(f"Done.\nElapsed time: {end - start:.3f} seconds.")

if __name__=="__main__":
    main()
