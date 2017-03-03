import numpy as np

import pyximport
pyximport.install()

import hw13
import hw14
import hw14opt

import numpy as np

def generate_test_matrices(n):

    matrices = []

    for i in range(n):
        n1 = np.random.randint(low=2,high=10)
        n2 = np.random.randint(low=2,high=10)
        n3 = np.random.randint(low=2,high=10)

        A = np.random.normal(size=(n1,n2))
        B = np.random.normal(size=(n2,n3))
        C = np.dot(A,B)

        matrices+= [(A,B,C)]

    return matrices



def test_mult(F):
    matrices = generate_test_matrices(5)

    for A,B,C in matrices:

        D = np.empty_like(C)
        F(A,B,D,4)

        result = np.allclose(C,D)

        print(np.abs(C-D))

    print(result)


test_mult(hw14.matMult_serial)
test_mult(hw14.matMult_naive)