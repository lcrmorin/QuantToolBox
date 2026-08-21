"""Translated from Examples/ects/varx1e.m -- Lutkepohl [1991], chapter 5:
lag-order selection (BIC/AICa/AICc/SIC/FPE/AIC/HQ) for a VAR model on the
same log-differenced West German investment/income/consumption data as
varx1a.py, over p = 1..5.

`varx_order(y, 1, seqa(1, 1, 5))` in the original again passes the scalar
`1` for the exogenous regressor `x`, i.e. "just a constant"; `seqa(1,
1, 5)` is the sequence 1, 2, 3, 4, 5 (`np.arange(1, 6)`)."""

import io

import numpy as np

from quanttoolbox.econometrics.var import varx_order

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


d_investment = _diff1(investment)
d_income = _diff1(income)
d_consumption = _diff1(consumption)
y = np.column_stack([d_investment, d_income, d_consumption])

n = y.shape[0]
x = np.ones((n, 1))

result = varx_order(y, x, np.arange(1, 6))

print("p values:", result.p_values)
print("\ncriteria (columns: BIC, AICa, AICc, SIC, FPE, AIC, HQ):")
print(np.round(result.criteria, 4))
print("\noptimal p per criterion:", result.optimal_p)
