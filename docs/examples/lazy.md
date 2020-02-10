## Simple Example

![](https://raw.githubusercontent.com/timkpaine/tributary/master/docs/img/lazy/example.gif)

## Symbolic Example
Here we will construct a lazy pricer for a vanilla european option

```python
import numpy as np
import sympy as sy
from IPython.display import display, HTML
from sympy.stats import Normal as syNormal, cdf
sy.init_printing()

# adapted from https://gist.github.com/raddy/bd0e977dc8437a4f8276
#spot, strike, vol, days till expiry, interest rate, call or put (1,-1)
spot, strike, vol, dte, rate, cp = sy.symbols('spot strike vol dte rate cp')

T = dte / 260.
N = syNormal('N', 0.0, 1.0)

d1 = (sy.ln(spot / strike) + (0.5 * vol ** 2) * T) / (vol * sy.sqrt(T))
d2 = d1 - vol * sy.sqrt(T)

TimeValueExpr = sy.exp(-rate * T) * (cp * spot * cdf(N)(cp * d1) - cp * strike  * cdf(N)(cp * d2))
```

Let's take a look at the sympy expression

![](https://raw.githubusercontent.com/timkpaine/tributary/master/docs/img/lazy/example3.png)

Now using tributary, we construct a lazily-evaluated graph

```python
import tributary.symbolic as ts
PriceClass = ts.construct_lazy(TimeValueExpr)

price = PriceClass(spot=210.59, strike=205, vol=14.04, dte=4, rate=.2175, cp=-1)

price.evaluate()()  # 124.819

price.strike = 210
price.evaluate()()  # 124.032

price.evaluate().graphviz()
```

![](https://raw.githubusercontent.com/timkpaine/tributary/master/docs/img/lazy/example4.png)

If we adjust an input, we can see the nodes that will be recalculated on next evaluation


```python
price.strike = 205
price.evaluate().graphviz()
```

![](https://raw.githubusercontent.com/timkpaine/tributary/master/docs/img/lazy/example5.png)
