"""Portfolio construction: risk budgeting, mean-variance, tracking error, Black-Litterman.

Maps from: QuantToolBox/rpb/*.m, QuantToolBox/crb/*.m, QuantToolBox/mloapa/*.m
The ~95 original files (many near-duplicate ADMM/CCD/Newton/fmincon solver
variants) are consolidated into a handful of classes parameterized by
`method=` and `constraints=`. See docs/migration_map.md for the mapping.
"""
