#!/usr/bin/env python3
#  from tkmessagebox import *
"""
Validators
"""
import numpy as np


def show_number(val, tol=None):
    """
    Formatea un número a la tolerancia dada.
    """
    if np.isnan(val):
        return " nan "
    if tol is None:
        return "%.3g" % val
    if np.isnan(tol):
        return "nan"

    tol = 10.0 ** int(np.log(tol) / np.log(10.0) - 1)
    if abs(val) < tol:
        return "0"
    val = int(val / tol) * tol
    return "%.3g" % val


def validate_pinteger(
    action,
    index,
    value_if_allowed,
    prior_value,
    text,
    validation_type,
    trigger_type,
    widget_name,
) -> bool:
    """
    Validate an integer entry
    """
    if text == "":
        return True
    if all(c in "0123456789" for c in text):
        return True

    return False


def validate_float(
    action,
    index,
    value_if_allowed,
    prior_value,
    text,
    validation_type,
    trigger_type,
    widget_name,
):
    """Validate a float entry"""
    if text == "":
        return True

    try:
        float(value_if_allowed)
        return True
    except ValueError:
        return False
    else:
        return False
