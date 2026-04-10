import matplotlib.pyplot as plt
import numpy as np

def j0(x):
    ''' returns the zeroth order spherical Bessel function of the first kind'''
    return np.sin(x)/x;

def j2(x):
    ''' returns the second order spherical Bessel function of the first kind'''
    return (3/x**2 - 1) * np.sin(x)/x - 3 * np.cos(x)/x**2;

def n0(x):
    ''' returns the zeroth order spherical Neumann function of the first kind'''
    return -np.cos(x)/x;

def n2(x):
    ''' returns the second order spherical Neumann function of the first kind'''
    return -((3/x**2 - 1) * np.cos(x)/x + 3 * np.sin(x)/x**2);

def h0(x):
    ''' returns the zeroth order spherical Hankel function of the first kind'''
    return j0(x) + 1j * n0(x);

def h2(x):
    ''' returns the second order spherical Hankel function of the first kind'''
    return j2(x) + 1j * n2(x);

def green(x):
    ''' returns the Green's function for the Helmholtz equation in 3D; polarization is perpendicular to separation vector'''
    if x == 0:
        return 1.;
    return h0(x) - 1/2. * h2(x);

def greenpara(x):
    ''' returns the Green's function for the Helmholtz equation in 3D; polarization is parallel to separation vector'''
    if x == 0:
        return 1.;
    return h0(x) + h2(x);

def greencirc(x):
    ''' returns the Green's function for the Helmholtz equation in 3D; polarization is x+iy/sqrt(2); motion is in the x-y plane'''
    if x == 0:
        return 1.;
    return h0(x) + 1/4. * h2(x);

def green_1D(x):
    ''' returns the Green's function for the Helmholtz equation in 1D'''
    return np.exp(1j * np.abs(x));

def g_3D_pol(kr, q):
    ''' returns the dyadic green's function in 3D free space for sepration phase kr, and polarization direction q, normalized so that lim g(0) = 1'''
    r_norm = np.sqrt(sum(kr[i]**2 for i in range(3)))
    if r_norm < 1e-6:
        return 1.;
    q_norm = np.sqrt(sum(np.abs(q[i])**2 for i in range(3)))
    r_unit = tuple(kr[i] / r_norm for i in range(3))
    q_unit = tuple(q[i] / q_norm for i in range(3))
    q_unit_conj = (np.conj(q_unit[0]), np.conj(q_unit[1]), np.conj(q_unit[2]))
    # Dot product between r_unit and q_unit
    dot_r_q = sum(r_unit[i] * q_unit[i] for i in range(3))
    # Dot product between r_unit and q_unit_conj
    dot_r_q_conj = sum(r_unit[i] * q_unit_conj[i] for i in range(3))
    factor = (3 * dot_r_q * dot_r_q_conj - 1) /2
    #print(factor)
    return h0(r_norm) + factor * h2(r_norm);

def test_g_3D_pol():
    q = (1. , 1.* 1j, 0)
    print(greencirc(19.75))
    print(g_3D_pol((19.75,0,0),q))
    return

def test_green():
    x = np.linspace(.2, 10, 100 + 1)
    z = [np.imag(green(i)) for i in x]
    plt.plot(x, z)
    plt.grid()
    plt.xlabel('x')
    plt.ylabel('Green(x)')
    plt.ylim(-1, 1)
    plt.show()
    return

def G00_ij_fixed(q, loc, i,j):
    '''returns the matrix element <0,e_i|G|0,e_j>
    g is the greens function for specified polarization q
    loc is the tuple x, y, z of phases (k_0 distantce) for all atoms
    '''
    xi = loc[0][i]
    yi = loc[1][i]
    zi = loc[2][i]
    xj = loc[0][j]
    yj = loc[1][j]
    zj = loc[2][j]
    
    if i == j:
        return 1.0
    else:
        dx = xj - xi
        dy = yj - yi
        dz = zj - zi

        return g_3D_pol((dx, dy, dz), q) # without spread term

def Gfs_N(q, loc, N):
    '''returns the matrix G00'''
    G = np.zeros((N,N),dtype=complex)
    for i in range(N):
        for j in range(N):
            G[i,j] = G00_ij_fixed(q, loc,i,j)
    return G

if __name__ == '__main__':
    test_g_3D_pol()
