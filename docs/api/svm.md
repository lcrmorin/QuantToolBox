# `quanttoolbox.svm`

!!! info "Python alternatives"
    **Switch** to `sklearn.svm.SVC`/`SVR` for standalone classification/regression — backed by `libsvm`/`liblinear` (compiled C), extremely well optimized and battle-tested. Already verified to match this module to 3+ decimal places. **Keep** this module only if the SVM needs to be composed with other constraints inside the same `solve_qp`/cvxpy optimization — that composability is the one thing sklearn's opaque solver can't offer.

## `svm.svm`

::: quanttoolbox.svm.svm
