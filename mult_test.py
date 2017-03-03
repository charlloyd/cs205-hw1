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

        A = A.astype(np.float64)
        B = B.astype(np.float64)

        C = np.dot(A,B)

        matrices+= [(A,B,C)]

    return matrices



def test_mult_1():
    matrices = generate_test_matrices(5)

    result = []

    for A,B,C in matrices:

        D = np.zeros_like(C,dtype=np.float64)

        hw14.matMult_serial(A,B,D,2)

        result.append(np.allclose(C,D))

    assert(result)


def test_mult_2():
    matrices = generate_test_matrices(5)

    result = []

    for A,B,C in matrices:

        D = np.zeros_like(C,dtype=np.float64)
        hw14.matMult_naive(A,B,D,2)

        result.append(np.allclose(C,D))

    assert(result)


def test_mult_3():
    matrices = generate_test_matrices(5)

    result = []

    for A,B,C in matrices:

        D = np.zeros_like(C,dtype=np.float64)

        hw14.matMult_thread(A,B,D,2,1)

        result.append(np.allclose(C,D))

    assert(result)



test_mult_1()
test_mult_2()
test_mult_3()
