## Simple Example
```python
# Two example classes with lazy members
class Foo1(t.BaseClass):
    def __init__(self, *args, **kwargs):
        self.x = self.node('x', readonly=False, default_or_starting_value=1, trace=True)

class Foo2(t.BaseClass):
    def __init__(self, *args, **kwargs):
        self.y = self.node('y', readonly=False, default_or_starting_value=2, trace=True)

f1 = Foo1()
f2 = Foo2()
z = f1.x + f2.y
print(z())
print(z())  # no recalc
f1.x = 2
print(z())
print(z.value()) # no recalc
f2.y = 4
print(z())
print(z)) # no recalc

z.graphviz().render()
```

![](https://raw.githubusercontent.com/timkpaine/tributary/master/docs/img/lazy/example1.png)

```python
# if we update a dependent variable without recalculating, we see the dependencies
f2.y = 4

z.graphviz().render()
```
![](https://raw.githubusercontent.com/timkpaine/tributary/master/docs/img/lazy/example2.png)
