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
print(z.value())
print(z.value())  # no recalc
f1.x = 2
print(z.value())
print(z.value()) # no recalc
f2.y = 4
print(z.value())
print(z.value()) # no recalc

z.graphviz().render()
```

![](https://raw.githubusercontent.com/timkpaine/tributary/master/docs/img/lazy/example1.png)
