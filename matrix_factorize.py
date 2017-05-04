# Matrix factorization for link prediction

try:
    import numpy
except:
    print "This code performs matrix factorization and uses the NumPy module."
    exit(0)


@INPUT:
    R     : Input matrix, that will be factorized
    P     : N x K dimensional matrix
    Q     : M x K dimensional matrix
    K     : No. of latent features
    steps : Steps required for optimal result
    alpha : the learning rate
    beta  : Regularization parameter
@OUTPUT:
    the final matrices P and Q
"""
def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
    Q = Q.T
    for step in xrange(steps):
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - numpy.dot(P[i,:],Q[:,j])
                    for k in xrange(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR = numpy.dot(P,Q)
        e = 0
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - numpy.dot(P[i,:],Q[:,j]), 2)
                    for k in xrange(K):
                        e = e + (beta/2) * ( pow(P[i][k],2) + pow(Q[k][j],2) )
        if e < 0.001:
            break
    return P, Q.T

#Change the matrix R with adjacency matrix or any relevant matrix for factorization
if __name__ == "__main__":
    R = [
         [1,1,0,1],
         [1,0,0,1],
         [1,1,0,1],
         [1,0,0,1],
         [0,1,1,1],
        ]

    R = numpy.array(R)

    N = len(R)
    M = len(R[0])
    K = 2

    P = numpy.random.rand(N,K)
    Q = numpy.random.rand(M,K)

    nP, nQ = matrix_factorization(R, P, Q, K)}