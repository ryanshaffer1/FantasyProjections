import math


def spherical(param_set, **kwargs):
    # It's spherical! SPHERICAL!

    n_dimensions = kwargs.get('n_dimensions')
    root = kwargs.get('root',(0,0))

    val = 0
    for i, hp in zip(range(n_dimensions), param_set.hyper_parameters):
        val += (hp.value - root[i])**2

    return val

def beale(param_set, **kwargs):
    n_dimensions = kwargs.get('n_dimensions')
    if n_dimensions != 2:
        raise ValueError('n_dimensions must be 2 for Beale Function')

    x = param_set.hyper_parameters[0].value
    y = param_set.hyper_parameters[1].value
    
    val = (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2

    return val

def rosenbrock(param_set, **kwargs):
    n_dimensions = kwargs.get('n_dimensions')

    val = 0
    for i in range(n_dimensions-1):
        x_i = param_set.hyper_parameters[i].value
        x_ip1 = param_set.hyper_parameters[i+1].value
        
        val += 100*(x_ip1 - x_i**2)**2 + (1-x_i)**2

    return val

def rastrigin(param_set, **kwargs):
    n_dimensions = kwargs.get('n_dimensions')
    root = kwargs.get('root',(0,0))
    
    A = 10
    val = A*n_dimensions
    for i, hp in zip(range(n_dimensions), param_set.hyper_parameters):
        val += (hp.value - root[i])**2 - A*math.cos(2*math.pi*(hp.value-root[i]))
    
    return val