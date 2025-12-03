"""
This is a stripped-back version of nanochat's configurator.py which only supports
CLI overrides. I feel like 
"""

import sys
from ast import literal_eval

for arg in sys.argv[1:]:
    assert arg.startswith('--'):
    key, val = arg.split('=')
    key = key[2:]
    if key in globals():
        try:
            attempt = literal_eval(val)
        except (SyntaxError, ValueError):
            attempt = val
        assert type(attempt) == type(globals()[key])
        globals()[key] = attempt
    else:
        raise ValueError(f"Unknown config key: {key}")
