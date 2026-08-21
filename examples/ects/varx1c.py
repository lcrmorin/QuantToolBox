"""Translated from Examples/ects/varx1c.m -- Lutkepohl [1991], section
3.6: Wald test for no instantaneous causality from income/consumption to
investment, on the ML estimate of the VAR(2) model from varx1a.py/
varx1b.py's data. `varx_ls`/`varx_ml` map to `varx_estimate(...,
method="ls"|"ml")`; with method="ml" the returned `theta` is the
21-element stacked coefficient vector followed by the 6-element vech of
the Cholesky factor of Sigma (`varx_estimate`'s documented ML
convention), so theta[23]/theta[24] (1-indexed) fall within that
appended Cholesky block -- exactly as in the original."""

import io

import numpy as np

from quanttoolbox.econometrics.estimation import wald_test
from quanttoolbox.econometrics.var import varx_estimate

_LUTKEPOHL_DATA = """
       601       180       451       415
       602       179       465       421
       603       185       485       434
       604       192       493       448
       611       211       509       459
       612       202       520       458
       613       207       521       479
       614       214       540       487
       621       231       548       497
       622       229       558       510
       623       234       574       516
       624       237       583       525
       631       206       591       529
       632       250       599       538
       633       259       610       546
       634       263       627       555
       641       264       642       574
       642       280       653       574
       643       282       660       586
       644       292       694       602
       651       286       709       617
       652       302       734       639
       653       304       751       653
       654       307       763       668
       661       317       766       679
       662       314       779       686
       663       306       808       697
       664       304       785       688
       671       292       794       704
       672       275       799       699
       673       273       799       709
       674       301       812       715
       681       280       837       724
       682       289       853       746
       683       303       876       758
       684       322       897       779
       691       315       922       798
       692       339       949       816
       693       364       979       837
       694       371       988       858
       701       375      1025       881
       702       432      1063       905
       703       453      1104       934
       704       460      1131       968
       711       475      1137       983
       712       496      1178      1013
       713       494      1211      1034
       714       498      1256      1064
       721       526      1290      1101
       722       519      1314      1102
       723       516      1346      1145
       724       531      1385      1173
       731       573      1416      1216
       732       551      1436      1229
       733       538      1462      1242
       734       532      1493      1267
       741       558      1516      1295
       742       524      1557      1317
       743       525      1613      1355
       744       519      1642      1371
       751       526      1690      1402
       752       510      1759      1452
       753       519      1756      1485
       754       538      1780      1516
       761       549      1807      1549
       762       570      1831      1567
       763       559      1873      1588
       764       584      1897      1631
       771       611      1910      1650
       772       597      1943      1685
       773       603      1976      1722
       774       619      2018      1752
       781       635      2040      1774
       782       658      2070      1807
       783       675      2121      1831
       784       700      2132      1842
"""

data = np.loadtxt(io.StringIO(_LUTKEPOHL_DATA))
data = np.log(data)

investment = data[:, 1]
income = data[:, 2]
consumption = data[:, 3]


def _diff1(a: np.ndarray) -> np.ndarray:
    d = np.full_like(a, np.nan)
    d[1:] = a[1:] - a[:-1]
    return d


y = np.column_stack([_diff1(investment), _diff1(income), _diff1(consumption)])
n = y.shape[0]
x = np.ones((n, 1))

result = varx_estimate(y, x, p=2, method="ml")
theta = result.theta
vcv = result.vcv

# Test for no instantaneous causality from income/consumption to investment:
# theta[23] = theta[24] = 0 (1-indexed)
idx = np.array([23, 24]) - 1


def wald_function(th: np.ndarray) -> np.ndarray:
    return th[idx]


wald_result = wald_test(wald_function, theta, vcv, n_obs=data.shape[0])

print("Wald test -- no instantaneous causality from income/consumption to investment:")
print("chi2 stat:", round(wald_result.chi2_stat, 4), " p-value:", round(wald_result.chi2_pvalue, 4))
print("F stat:   ", round(wald_result.f_stat, 4), " p-value:", round(wald_result.f_pvalue, 4))
