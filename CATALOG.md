# Sources and Sinks
## Sources
- Python Function/Generator/Async Function/Async Generator
- Curve - yield through an iterable
- Const - yield a constant
- Timer - yield on an interval
- Random - generates a random dictionary of values
- File - streams data from a file, optionally loading each line as a json
- HTTP - polls a url with GET requests, streams data out
- HTTPServer - runs an http server and streams data sent by clients
- Websocket - strams data from a websocket
- WebsocketServer - runs a websocket server and streams data sent by clients
- SocketIO - streams data from a socketIO connection
- SocketIOServer - streams data from a socketIO connection
- SSE - streams data from an SSE connection
- Kafka - streams data from kafka
- Postgres - streams data from postgres

## Sinks
- Foo - data to a python function
- File - data to a file
- HTTP - POSTs data to an url
- HTTPServer - runs an http server and streams data to connections
- Websocket - streams data to a websocket
- WebsocketServer - runs a websocket server and streams data to connections
- SocketIO - streams data to a socketIO connection
- SocketIOServer - runs a socketio server and streams data to connections
- SSE - runs an SSE server and streams data to connections
- Kafka - streams data to kafka
- Postgres - streams data to postgres
- Email - streams data and sends it in emails
- TextMessage - streams data and sends it via text message

# Transforms
## Modulate
- Delay - Streaming wrapper to delay a stream
- Throttle - Streaming wrapper to only tick at most every interval
- Debounce - Streaming wrapper to only tick on new values
- Apply - Streaming wrapper to apply a function to an input stream
- Window - Streaming wrapper to collect a window of values
- Unroll - Streaming wrapper to unroll an iterable stream
- UnrollDataFrame - Streaming wrapper to unroll a dataframe into a stream
- Merge - Streaming wrapper to merge 2 inputs into a single output
- ListMerge - Streaming wrapper to merge 2 input lists into a single output list
- DictMerge - Streaming wrapper to merge 2 input dicts into a single output dict. Preference is given to the second input (e.g. if keys overlap)
- Reduce - Streaming wrapper to merge any number of inputs
- FixedMap - Map input stream to fixed number of outputs
- Subprocess - Open a subprocess and yield results as they come. Can also stream data to subprocess (either instantaneous or long-running subprocess)


## Calculations
Note that `tributary` can also be configured to operate on **dual numbers** for things like lazy or streaming autodifferentiation.

### Arithmetic Operators
- Noop (unary) - Pass input to output
- Negate (unary) - -1 * input
- Invert (unary) - 1/input
- Add (binary) - add 2 inputs
- Sub (binary) - subtract second input from first
- Mult (binary) - multiple inputs
- Div (binary) - divide first input by second
- RDiv (binary) - divide second input by first
- Mod (binary) - first input % second input
- Pow (binary) - first input^second input
- Sum (n-ary) - sum all inputs
- Average (n-ary) - average of all inputs
- Round (unary)
- Floor (unary)
- Ceil (unary)

### Boolean Operators
- Not (unary) - `Not` input
- And (binary) - `And` inputs
- Or (binary) - `Or` inputs

### Comparators
- Equal (binary) - inputs are equal
- NotEqual (binary) - inputs are not equal
- Less (binary) - first input is less than second input
- LessOrEqual (binary) - first input is less than or equal to second input
- Greater (binary) - first input is greater than second input
- GreaterOrEqual (binary) - first input is greater than or equal to second input

### Math
- Log (unary)
- Sin (unary)
- Cos (unary)
- Tan (unary)
- Arcsin (unary)
- Arccos (unary)
- Arctan (unary)
- Sqrt (unary)
- Abs (unary)
- Exp (unary)
- Erf (unary)

### Financial Calculations
- RSI - Relative Strength Index
- MACD - Moving Average Convergence Divergence

## Converters
- Int (unary)
- Float (unary)
- Bool (unary)
- Str (unary)

## Basket Functions
- Len (unary)
- Count (unary)
- Min (unary)
- Max (unary)
- Sum (unary)
- Average (unary)

## Rolling
- RollingCount - Node to count inputs
- RollingMin - Node to take rolling min of inputs
- RollingMax - Node to take rolling max of inputs
- RollingSum - Node to take rolling sum inputs
- RollingAverage - Node to take the running average
- SMA - Node to take the simple moving average over a window
- EMA - Node to take an exponential moving average over a window

## Node Type Converters
- Lazy->Streaming
