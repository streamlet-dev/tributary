

def rsi(node, rsi_periods=14):
    def calc_up(*val):
        total = 0
        old_price = 0
        new_price = 0
        for i, element in enumerate(val[0]):
            if i:
                new_price = technicalsField(element)
                change = new_price - old_price
                old_price = new_price
                if change > 0:
                    total += change
            else:
                old_price = technicalsField(element)
        return total

    def calc_down(*val):
        total = 0
        old_price = 0
        new_price = 0
        for i, element in enumerate(val[0]):
            if i:
                new_price = technicalsField(element)
                change = new_price - old_price
                old_price = new_price
                if change < 0:
                    total -= change
            else:
                old_price = technicalsField(element)
        return total

    def get_last(*val):
        return val[0][-1]

    def filter_function(*val):
        if len(val[0]) > 1:
            if val[0][1]['time'] != val[0][0]['time']:
                return val[0][1]
        return ts.StreamNone()

    price_window = Window(Apply(Window(n, size=2), filter_function), size=rsi_periods, full_only=True)

    avg_up = Div(Apply(n, calc_up), Const(value=rsi_periods))

    avg_down = Div(Apply(n, calc_down), Const(value=rsi_periods))

    rsi = [Sub(Const(value=100), Div(Const(value=100),
                         Sum(Const(value=1), Div(up, down))))
           for up, down in zip(avg_up, avg_down)]

    node = ts.Reduce(*[ts.Reduce(ts.Apply(n, get_last), rsi_node, reducer=rejoin("rsa'"))
                       for n, rsi_node in
                       zip(price_windows, rsi)])
