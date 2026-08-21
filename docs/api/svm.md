# `quanttoolbox.svm`

!!! info "Python alternatives"
    **Switch** to `sklearn.svm.SVC`/`SVR` for standalone classification/regression — backed by `libsvm`/`liblinear` (compiled C), extremely well optimized and battle-tested. Already verified to match this module to 3+ decimal places. **Keep** this module only if the SVM needs to be composed with other constraints inside the same `solve_qp`/cvxpy optimization — that composability is the one thing sklearn's opaque solver can't offer.

## `svm.svm`

::: quanttoolbox.svm.svm

### Examples

??? example "Hard- and soft-margin SVM classification, dual formulation — svm/svm5.py"
    ```python
    --8<-- "examples/svm/svm5.py"
    ```

??? example "Hard-margin SVM classification, dual formulation — svm/svm2.py"
    ```python
    --8<-- "examples/svm/svm2.py"
    ```

??? example "OLS, LAD, quantile, and SVM regression compared — svm/svm6.py"
    ```python
    --8<-- "examples/svm/svm6.py"
    ```

??? example "OLS/SVM-LS and quantile/SVM-epsilon regression on synthetic data — svm/svm8.py"
    ```python
    --8<-- "examples/svm/svm8.py"
    ```

??? example "Soft-margin SVM classification, dual formulation — svm/svm4.py"
    ```python
    --8<-- "examples/svm/svm4.py"
    ```

??? example "Soft-margin SVM classification, primal formulation — svm/svm3.py"
    ```python
    --8<-- "examples/svm/svm3.py"
    ```

??? example "SVM regression, dual formulation, vs. the primal result — svm/svm7.py"
    ```python
    --8<-- "examples/svm/svm7.py"
    ```
