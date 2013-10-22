#!usr/bin/env python
"""Functions for doing principal coordinates analysis on a distance matrix

Calculations performed as described in:

Principles of Multivariate analysis: A User's Perspective. W.J. Krzanowski 
Oxford University Press, 2000. p106.

Note: There are differences in the signs of some of the eigenvectors between
R's PCoA function (cmdscale) and the implementation provided here. This is due
to numpy's eigh() function. The eigenvalues returned by eigh() match those
returned by R's eigen() function, but some of the eigenvectors have swapped
signs. Numpy's eigh() function uses a different set of Fortran routines (part
of LAPACK) to calculate the eigenvectors than R does. Numpy uses DSYEVD and
ZHEEVD, while R uses DSYEVR and ZHEEV. As far as the Fortran documentation
goes, those routines should produce the same results, they just use different
algorithms to obtain them. The differences in sign do not affect the overall
results as they are just a different reflection. R's cmdscale documentation
also confirms the possibility of obtaining different signs between different R
platforms. Please feel free to send questions to jai.rideout@gmail.com.
"""
from numpy import add, sum, sqrt, argsort, newaxis, abs, dot, hstack
from numpy.linalg import eigh, qr, svd
from numpy.random import standard_normal
from cogent.util.dict2d import Dict2D
from cogent.util.table import Table
from cogent.cluster.UPGMA import inputs_from_dict2D

try:
    from scipy.sparse.linalg import eigsh
except ImportError:
    pass

__author__ = "Catherine Lozupone"
__copyright__ = "Copyright 2007-2012, The Cogent Project"
__credits__ = ["Catherine Lozuopone", "Rob Knight", "Peter Maxwell",
               "Gavin Huttley", "Justin Kuczynski", "Daniel McDonald",
               "Jai Ram Rideout"]
__license__ = "GPL"
__version__ = "1.5.3-dev"
__maintainer__ = "Catherine Lozupone"
__email__ = "lozupone@colorado.edu"
__status__ = "Production"

def PCoA(pairwise_distances):
    """runs principle coordinates analysis on a distance matrix
    
    Takes a dictionary with tuple pairs mapped to distances as input. 
    Returns a cogent Table object.
    """
    items_in_matrix = []
    for i in pairwise_distances:
        if i[0] not in items_in_matrix:
            items_in_matrix.append(i[0])
        if i[1] not in items_in_matrix:
            items_in_matrix.append(i[1])
    dict2d_input = [(i[0], i[1], pairwise_distances[i]) for i in \
            pairwise_distances]
    dict2d_input.extend([(i[1], i[0], pairwise_distances[i]) for i in \
            pairwise_distances])
    dict2d_input = Dict2D(dict2d_input, RowOrder=items_in_matrix, \
            ColOrder=items_in_matrix, Pad=True, Default=0.0)
    matrix_a, node_order = inputs_from_dict2D(dict2d_input)
    point_matrix, eigvals = principal_coordinates_analysis(matrix_a)
    return output_pca(point_matrix, eigvals, items_in_matrix)

def principal_coordinates_analysis(distance_matrix, algorithm, dimensions):
    """Takes a distance matrix and returns principal coordinate results

    point_matrix: each row is an axis and the columns are points within the axis
    eigvals: correspond to the rows and indicate the amount of the variation
        that that the axis in that row accounts for
    NOT NECESSARILY SORTED
    """
    E_matrix = make_E_matrix(distance_matrix)
    F_matrix = make_F_matrix(E_matrix)
    
    # selecting the exact algorithm
    eigvals, eigvecs = globals()[algorithm](F_matrix, dimensions)
    
    #drop imaginary component, if we got one
    eigvals = eigvals.real
    eigvecs = eigvecs.real
    point_matrix = get_principal_coordinates(eigvals, eigvecs)
    
    # fixing coordinates - used to be in qiime
    pcnts = (abs(eigvals)/sum(abs(eigvals)))*100
    idxs_descending = pcnts.argsort()[::-1]
    point_matrix = point_matrix[idxs_descending]
    eigvals = eigvals[idxs_descending]
    pcnts = pcnts[idxs_descending]
    
    return point_matrix.T[:,:dimensions], eigvals[:dimensions], pcnts[:dimensions]

def make_E_matrix(dist_matrix):
    """takes a distance matrix (dissimilarity matrix) and returns an E matrix

    input and output matrices are numpy array objects of type Float

    squares and divides by -2 each element in the matrix
    """
    return (dist_matrix * dist_matrix) / -2.0

def make_F_matrix(E_matrix):
    """takes an E matrix and returns an F matrix

    input is output of make_E_matrix

    for each element in matrix subtract mean of corresponding row and 
    column and add the mean of all elements in the matrix
    """
    num_rows, num_cols = E_matrix.shape
    #make a vector of the means for each row and column
    #column_means = (add.reduce(E_matrix) / num_rows)
    column_means = (add.reduce(E_matrix) / num_rows)[:,newaxis]
    trans_matrix = E_matrix.transpose()
    row_sums = add.reduce(trans_matrix)
    row_means = row_sums / num_cols
    #calculate the mean of the whole matrix
    matrix_mean = sum(row_sums) / (num_rows * num_cols)
    #adjust each element in the E matrix to make the F matrix

    E_matrix -= row_means
    E_matrix -= column_means
    E_matrix += matrix_mean

    #for i, row in enumerate(E_matrix):
    #    for j, val in enumerate(row):
    #        E_matrix[i,j] = E_matrix[i,j] - row_means[i] - \
    #                column_means[j] + matrix_mean
    return E_matrix

def run_eigh(F_matrix, dimensions=10):
    """takes an F-matrix and returns eigenvalues and eigenvectors"""

    #use eig to get vector of eigenvalues and matrix of eigenvectors
    #these are already normalized such that
    # vi'vi = 1 where vi' is the transpose of eigenvector i
    eigvals, eigvecs = eigh(F_matrix)
    #NOTE: numpy produces transpose of Numeric!

    return eigvals, eigvecs.transpose()
    
def run_ssvd(F_matrix, k=10, p=10, qiter = 0):
    """takes an F-matrix and returns eigenvalues and eigenvectors of the ssvd method
    Based on algorithm described in 'Finding Structure with Randomness:
    Probabilistic Algorithms for Constructing Approximate Matrix Decompositions'
    by N. Halko, P.G. Martinsson, and J.A. Tropp
    Code adapted from R:
    https://cwiki.apache.org/confluence/display/MAHOUT/Stochastic+Singular+Value+Decomposition
    
    F_matrix: F_matrix
    k: dimensions
    p: oversampling parameter - this is added to k to boost accuracy
    qiter: iterations to go through - to boost accuracy
    
    NOTE: the lower the k (dimensions) the __worse__ the resulting matrix is
    """
    m, n = F_matrix.shape
    p = min(min(m,n)-k, p)     # an mxn matrix M has at most p = min(m,n) unique 
                               # singular values
    r = k+p                    # rank plus oversampling parameter p

    omega = standard_normal(size = (n,r))   # generate random matrix omega
    y = dot(F_matrix, omega)    # compute a sample matrix Y: apply F_matrix to random 
                                # vectors to identify part of its range corresponding
                                # to largest singular values
    Q, R = qr(y)                # find an ON matrix st. Y = QQ'Y
    b = dot(Q.transpose(), F_matrix)    # multiply F_matrix by Q whose columns form 
                                        # an orthonormal basis for the range of Y
                                        
    #often, no iteration required to small error in eqn. 1.5
    for i in xrange(1, qiter):
        y = dot(F_matrix, b.transpose())
        Q,R = qr(y)
        b = dot(Q.transpose(), F_matrix)

    bbt = dot(b, b.transpose())     # compute eigenvalues of much smaller matrix bbt
    if globals().get(eigsh, True):
        eigvals, eigvecs = eigsh(bbt,k)   # eigsh faster, returns sorted eigenvals/vecs
    else:
        eigvals, eigvecs = eigh(bbt,k)   # eigsh faster, returns sorted eigenvals/vecs
    U_ssvd = dot(Q,eigvecs) #[:,1:k]
    # don't need to compute V
    # V_ssvd = dot(transpose(b),dot(eigvecs, diag(1/eigvals,0))) [:, 1:k] 
    eigvecs = U_ssvd.real

    return eigvals, eigvecs.transpose()

def run_fsvd(F_matrix, k=10, i=1, usePowerMethod=0):    #fsvd function instead of eigh
    """takes an F-matrix and returns eigenvalues and eigenvectors of the ssvd method
    Based on algorithm described in 'An Algorithm for the Principal Component analysis
    of Large Data Sets' by N. Halko, P.G. Martinsson, Y. Shkolnisky, and M. Tygert
    and Matlab code: http://stats.stackexchange.com/questions/2806/best-pca-algorithm-for-huge-number-of-features
    
    F_matrix: F_matrix
    k: dimensions
    i: is the number of levels of the Krylov method to use, for most applications, i=1 or i=2 is sufficient
    userPowerMethod: changes the power of the spectral norm (minimizing the error). See p11/eq8.1 DOI = {10.1137/100804139}  
    """
    m, n = F_matrix.shape
    
    if m < n:
        F_matrix = F_matrix.transpose()
    
    m, n = F_matrix.shape    #dimensions could have changed in above Transpose
    l = k + 2

    G = standard_normal(size = (n,l))   #entries independent, identically distributed Gaussian random variables of zero mean and unit variance
    if usePowerMethod == 1:
        H = dot(F_matrix,G)
        for x in xrange(2, i+2):
            H = dot(F_matrix,dot(F_matrix.transpose(),H))   #enhance decay of singular values
    else:
        H = dot(F_matrix,G)
        H = hstack((H,dot(F_matrix,dot(F_matrix.transpose(),H))))
        for x in xrange(3, i+2):
            H = hstack((H,dot(F_matrix,dot(F_matrix.transpose(),tmp))))
    
    Q, R = qr(H)    #pivoted QR-decomposition
    T = dot(F_matrix.transpose(),Q) #step 3
    
    Vt, St, W = svd(T, full_matrices=False) #step 4 (as documented in paper)
    W = W.transpose()
    
    Ut = dot(Q,W)

    if m < n:
        V_fsvd = Ut[:,:k]
        U_fsvd = Vt[:,:k]
    else:
        U_fsvd = Ut[:,:k]
        V_fsvd = Vt[:,:k]
    
    #drop imaginary component, if we got one
    eigvals = (St[:k]**2).real
    eigvecs = U_fsvd.real

    return eigvals, eigvecs.transpose()

def get_principal_coordinates(eigvals, eigvecs):
    """converts eigvals and eigvecs to point matrix
    
    normalized eigenvectors with eigenvalues"""

    #get the coordinates of the n points on the jth axis of the Euclidean
    #representation as the elements of (sqrt(eigvalj))eigvecj
    #must take the absolute value of the eigvals since they can be negative
    return eigvecs * sqrt(abs(eigvals))[:,newaxis]

def output_pca(PCA_matrix, eigvals, names):
    """Creates a string output for principal coordinates analysis results. 

    PCA_matrix and eigvals are generated with the get_principal_coordinates 
    function. Names is a list of names that corresponds to the columns in the
    PCA_matrix. It is the order that samples were represented in the initial
    distance matrix.
    
    returns a cogent Table object"""
    
    output = []
    #get order to output eigenvectors values. reports the eigvecs according
    #to their cooresponding eigvals from greatest to least
    vector_order = list(argsort(eigvals))
    vector_order.reverse()
    
    # make the eigenvector header line and append to output
    vec_num_header = ['vec_num-%d' % i for i in range(len(eigvals))]
    header = ['Label'] + vec_num_header
    #make data lines for eigenvectors and add to output
    rows = []
    for name_i, name in enumerate(names):
        row = [name]
        for vec_i in vector_order:
            row.append(PCA_matrix[vec_i,name_i])
        rows.append(row)
    eigenvectors = Table(header=header,rows=rows,digits=2,space=2,
                    title='Eigenvectors')
    output.append('\n')
    # make the eigenvalue header line and append to output
    header = ['Label']+vec_num_header
    rows = [['eigenvalues']+[eigvals[vec_i] for vec_i in vector_order]]
    pcnts = (eigvals/sum(eigvals))*100
    rows += [['var explained (%)']+[pcnts[vec_i] for vec_i in vector_order]]
    eigenvalues = Table(header=header,rows=rows,digits=2,space=2, 
                    title='Eigenvalues')
    
    return eigenvectors.appended('Type', eigenvalues, title='')

