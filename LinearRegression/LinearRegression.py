def average(list):
    """ Returns the average of the array elements

    Parameters: 
    a (array): Array containing numbers whose mean is desired

    Returns:
    float: the average of the array elements

    """
    return sum(list) / float(len(list))

def partial_derivative(xs, ys, dx):
    list = []

    for i in range(len(xs)):
        list.append(dx(xs[i], ys[i]))

    return sum(list) / float(len(list))

def cost(xs, ys, h):
    """Cost function (loss function)

    Parameters:
    xs (array): 
    ys (array): 
    h (function): Hypothesis

    Returns:
    float: Value of cost function

    """

    list = []

    for i in range(len(xs)):
        xi = xs[i]
        hi = h(xi)
        yi = ys[i]
        list.append((hi - yi) ** 2)

    return 0.5 * average(list)