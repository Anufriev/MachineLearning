import LinearRegression

import random
import math
import csv
import numpy as np

import pprint

from tabulate import tabulate

from plotly import plotly as py
from plotly import graph_objs as go



def get_data(m):

    w0 = random.randrange(-100, 100, 1)
    w1 = 0.01 * random.randrange(-100, 100, 1)

    print("w0={w0}".format(w0=w0))
    print("w1={w1}".format(w1=w1))

    xs = []
    ys = []

    for i in range(m):
        bias_x = 0.01 * random.randrange(-100, 100, 5)
        bias_y = 0.001 * (random.randrange(-100, 100, 10) ** 3)

        x = i + bias_x
        y = w0 + w1 * x + bias_y
        
        xs.append(x)
        ys.append(y)

    return (xs, ys)


traces = []
costs = []

def add_hypothensies_trace(xs, w0, w1, name):
    hs = []

    for x in xs:
        hs.append(w0 + w1 * x)

    trace = go.Scatter(x = xs,
        y = hs,
        mode = 'lines',
        name = name)

    traces.append(trace)

def train(xs, ys, steps):
    
    w0 = 0.5
    w1 = 0.5

    l = 0.00001

    pp = pprint.PrettyPrinter(indent=0)
       
    d = []

    for i in range(steps):
           
        def h(x):
            return w0 + w1 * x
    
        def pdw0(xi, yi):
            return h(xi) - yi

        def pdw1(xi, yi):
            return (h(xi) - yi) * xi

        j = LinearRegression.cost(xs, ys, h)
        costs.append(j)

        dw0 = LinearRegression.partial_derivative(xs, ys, pdw0)
        dw1 = LinearRegression.partial_derivative(xs, ys, pdw1)
        
        d.append([i, j, w0, dw0, w1, dw1])

        w0 = w0 - (l * dw0)
        w1 = w1 - (l * dw1)
        
        #add_hypothensies_trace(xs, w0, w1, 'h' + str(i))

    print(tabulate(d, headers=['#', 'J', 'w0', 'dw0', 'lw0', 'lw0 * dw0', 'w1', 'dw1', 'lw1', 'lw0 * dw0']))

    return (w0, w1)

n = 500
steps = 500

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

add_hypothensies_trace(xs, w0, w1, 'gradient descent')

costTrace = go.Scatter(x = list(range(steps)), y = costs, mode = 'lines', name = "J")
traces.append(costTrace)

py.plot(traces, filename='linear-regression')
