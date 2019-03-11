import LinearRegression

import random
import math
import matplotlib.pyplot as plt

import pprint

from tabulate import tabulate

from plotly import plotly as py
from plotly import graph_objs as go



def get_data(m):
    w0 = 0.2 #0.000001 * random.randrange(-100, 100, 1)
    w1 = 0.9 #0.000001 * random.randrange(-100, 100, 1)

    print("w0={w0}".format(w0=w0))
    print("w1={w1}".format(w1=w1))

    xs = []
    ys = []

    for i in range(m):
        bias_x = 0.001 * random.randrange(-100, 100, 5)
        bias_y = 0.0001 * (random.randrange(-100, 100, 10) ** 3)

        x = i + bias_x
        y = w0 + w1 * x + bias_y
        
        xs.append(x)
        ys.append(y)

    return (xs, ys)


traces = []

def add_hypothensies_trace(xs, w0, w1, name):
    hs = []

    for x in xs:
        hs.append(w0 + w1 * x)

    trace = go.Scatter(x = xs,
        y = hs,
        mode = 'lines',
        name = name)

    traces.append(trace)



def train(xs, ys, n):
    
    w0 = 0.5
    w1 = 0.5

    ldw0 = 0.00001
    ldw1 = 0.0001

    pp = pprint.PrettyPrinter(indent=0)
       
    d = []

    for i in range(n):
           
        def h(x):
            return w0 + w1 * x
    
        def pdw0(xi, yi):
            return h(xi) - yi

        def pdw1(xi, yi):
            return (h(xi) - yi) * xi

        j = LinearRegression.cost(xs, ys, h)
    
        
        dw0 = LinearRegression.partial_derivative(xs, ys, pdw0)
        dw1 = LinearRegression.partial_derivative(xs, ys, pdw1)
        
        d.append([i, j, w0, dw0, ldw0, ldw0 * dw0, w1, dw1, ldw1, ldw1 * dw1])

        w0 = w0 - (ldw0 * dw0)
        w1 = w1 - (ldw1 * dw1)

        previous_dw0 = dw0
        previous_dw1 = dw1

        add_hypothensies_trace(xs, w0, w1, 'h' + str(i))

    print(tabulate(d, headers=['#', 'J', 'w0', 'dw0', 'lw0', 'lw0 * dw0', 'w1', 'dw1', 'lw1', 'lw0 * dw0']))

    return (w0, w1)

n = 300
steps = 5

# generate array from n elements;
data = get_data(n)
xs = data[0]
ys = data[1]

data_trace = go.Scatter(x = xs, y = ys, mode = 'markers', name = 'markers')
traces.append(data_trace)

# train model
result = train(xs, ys, steps)

w0 = result[0]
w1 = result[1]

py.plot(traces, filename='linear-regression')