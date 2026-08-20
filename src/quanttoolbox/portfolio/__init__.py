"""Portfolio construction: risk budgeting, mean-variance, tracking error, Black-Litterman.

Maps from: QuantToolbox/rpb/*.m, QuantToolbox/crb/*.m, QuantToolbox/mloapa/*.m
The ~95 original files (many near-duplicate ADMM/CCD/Newton/fmincon solver
variants) are consolidated into a handful of classes parameterized by
`method=` and `constraints=`. See docs/migration_map.md for the mapping.
"""
