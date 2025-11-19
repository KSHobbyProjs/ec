#!/usr/bin/env python3

import numpy as np
from . import pmm

import logging
logger = logging.getLogger(__name__)

def parse_parameter_values(parameter_string):
    """
    Parses a flexible CLI argument for parameter values.

    Parameters
    ----------
    parameter_string : str
        CLI string containing parameter info.

    Returns
    -------
    parameter_values : ndarray
        1D array of parameter values.

    Examples
    --------
    '1.5'            -> np.array([1.5])
    '1.0,2.0,3.0'    -> np.array([1.0, 2.0, 3.0])
    '5.0,20.0:50'    -> np.linspace(5, 20, 50)
    '5.0,20.0:50,1.5 -> 5 + np.linspace(0, 1, 50)**1.5 * (20 - 5)
    """
    s = parameter_string.strip()

    # If colon syntax (linspace)
    if ":" in s:
        try:
            lmin_lmax, llen_lexp = s.split(":")
            lmin, lmax = lmin_lmax.split(",")
            if "," in llen_lexp:
                llen, lexp = llen_lexp.split(",")
                lexp = float(lexp)
            else:
                llen = llen_lexp
                lexp = 1.0
            lmin, lmax, llen = float(lmin), float(lmax), int(llen)
            return lmin + np.linspace(0.0, 1.0, llen)**lexp * (lmax - lmin)
        except Exception as e:
            raise ValueError(f"Invalid linspace format: {s}. Use 'start,end:len' or 'start,end:len,exp'") from e
    
    # otherwise, assume comma-separated list of numbers
    if "," in s:
        return np.array([float(x) for x in s.split(",")])

    # otherwise, assume a single float
    return np.array([float(s)])
 
def parse_kwargs(kwargs_string):
    """
    Parses a comma-separated list of key=val pairs into a dictionary.

    Parameters
    ----------
    kwargs_string : str
        Comma-separated string containing kwarg info.

    Returns
    -------
    kwargs : dict
        Dictionary containing the key=val pairs in `kwargs_string`.
    
    Example
    -------
    'N=32,V0=-4.0,R=2.0' -> {"N" : 32, "V0" : -4.0, "R" : 2.0}.
    """
    s = kwargs_string.strip()
    kwargs = {}
    for kv in s.split(","):
        if not kv.strip():
            continue
        if "=" not in kv:
            raise RuntimeError(f"Invalid argument input: '{kv}'. Kwarg arguments need to be input in the form 'key1=val1,key2=val2'")
        k, v = kv.split("=", 1)   
        kwargs[k.strip()] = convert_value(v) 
    return kwargs

def convert_value(v):
    """
    Takes a string and determines if it's meant to be an int, float, bool, or str
    """
    v = v.strip().lower()
    try:
        return int(v)       # check int
    except ValueError:
        pass

    try:
        return float(v)     # check float
    except ValueError:
        pass

    if v == "true":
        return True
    elif v == "false":
        return False        
    return v                # fallback string
