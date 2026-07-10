# type: ignore
import sympy as sp
import jax

def test_substitution():
    # breakpoint()
    x, y = sp.symbols('x y')
    # x = 2
    expr = (x + 1)**(2*y)
    print(type(expr))
    expr = expr.subs(x, 2)
    print(type(expr))
    print(expr)
    exprfn = sp.lambdify(x, expr, 'numpy')
    print(type(exprfn))
    print(exprfn(2))
    # assert isinstance(expr, sp.Symbol)

class MyNode:
    def __new__(cls, *args, **kwargs):
        print("MyNode object created")
        instance = super().__new__(cls)
        return instance 
    def __init__(self, value):
        print(value)
        self.value = value

class Other:
    def __new__(cls, value):
        if value > 0:
            instance = MyNode(value)
            return instance 
        else:
            instance = super().__new__(cls)
            print("Other object created")    
            instance.value = value 
            return instance     

def test_class():
    node = MyNode(10)
    newnode = Other(-1)
    newnewnode = Other(10)
    print(
        f"node has type {type(node)} and stores {node.value}\n"
        f"newnode has type {type(newnode)} and stores {newnode.value}\n"
        f"newnewnode has type {type(newnewnode)} and stores {newnewnode.value}\n"
    )

def test_symbolequality():
    beta, K, mu = sp.symbols("beta K mu", real=True)
    A = sp.Matrix(
        [
            [0, 1, 0, 0],
            [-1, -beta, -mu * (K**2), 0],
            [0, 0, 0, 1],
            [0, 0, -(K**2), 0],
        ],
    )
    # zero_mat = sp.zeros(4, 4)
    bool_matrix = sp.Matrix(A.rows, A.cols, lambda i, j: A[i, j].equals(0))
    print(type(bool_matrix))
    print(bool_matrix)

def test_keys():
    key = jax.random.key(2)
    keys = jax.random.split(key, (2, 4, 5))
    print(keys.shape)
    assert keys.shape == (2, 4, 5)

if __name__=="__main__":
    test_keys()
    # test_symbolequality()
    # test_substitution()
    # test_class()