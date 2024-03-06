#!/usr/bin/env python3


QUOTE = """#  BONDS GENERATOR 0.0:
#  1-Please open a .CIF file with the site positions to start...
#  2- Enter the parameters and Press Generate model to define
      effective couplings...
#  3- Press optimize configurations in order to determine the
      optimal configurations for the ab initio calculations
# 4- With the ab-initio energies press "Calculate model parameters"..
"""

textmarkers = {}

textmarkers["separator_symbol"] = {
    "latex": "",
    "plain": "",
    "wolfram": ", ",
}
textmarkers["Delta_symbol"] = {
    "latex": "\\Delta ",
    "plain": "Delta",
    "wolfram": "\\[Delta]",
}
textmarkers["times_symbol"] = {
    "latex": "",
    "plain": "*",
    "wolfram": "*",
}
textmarkers["equal_symbol"] = {
    "latex": "=",
    "plain": "=",
    "wolfram": "==",
}
textmarkers["open_comment"] = {
    "latex": "% ",
    "plain": "#  ",
    "wolfram": "(*",
}
textmarkers["close_comment"] = {
    "latex": "",
    "plain": "",
    "wolfram": "*)",
}
textmarkers["sub_symbol"] = {
    "latex": "_",
    "plain": "",
    "wolfram": "",
}
textmarkers["plusminus_symbol"] = {
    "latex": "\\pm",
    "plain": "+/-",
    "Wolfram": "\\[PlusMinus]",
}

textmarkers["open_mod"] = {
    "latex": "\\left |",
    "plain": "|",
    "Wolfram": "Abs[",
}

textmarkers["close_mod"] = {
    "latex": "\\right |",
    "plain": "|",
    "Wolfram": "] ",
}
