import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from six.moves import cPickle as pickle
import scipy.optimize as optimize
from numpy import linalg as LA
import copy as cp
from gen_Hk import build_Hk_brute
from gen_Hk import build_Hk_smart
build_Hk = build_Hk_smart
# build_Hk = build_Hk_brute
np.set_printoptions(precision=4, suppress=True,
                    threshold=6, linewidth=100)


pi = np.pi
fs = 15
# % Pauli matrices
s0 = np.array([[1, 0],
               [0, 1]], dtype=complex)
sx = np.array([[0, 1],
               [1, 0]])
sy = np.array([[0, -1j],
               [1j, 0]])
sz = np.array([[1, 0],
               [0, -1]])
s_10 = np.array([[0, 0],
                 [1, 0]])
s_01 = np.array([[0, 1],
                 [0, 0]])
s_00 = np.array([[1, 0],
                 [0, 0]])
s_11 = np.array([[0, 0],
                 [0, 1]])

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['xtick.direction'] = "in"
matplotlib.rcParams['ytick.direction'] = "in"

# % ###################################################


def save_dict(di_, filename):
    with open(f'{filename}.pkl', 'wb') as f:
        pickle.dump(di_, f)


def load_dict(filename):
    with open(f'{filename}.pkl', 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di


def extract_deltas(terms_all):

    del_lay1 = cp.deepcopy(terms_all['1'][:, 2])
    del_lay2 = cp.deepcopy(terms_all['2'][:, 2])
    deltas = np.array(np.hstack((del_lay1, del_lay2)),
                      dtype=complex)

    return deltas


# @gif.frame
def plot_dirac_points(params, terms_all):

    valence_band = MF_band_struc(params, terms_all, plot=False)

    # get the indices of the 8 highest points of the band
    indices = valence_band.ravel().argsort()[-8:]

    # recast the indices into a 2*2 form
    dps = np.unravel_index(indices, valence_band.shape)

    # get the maximum values
    max_values = valence_band.ravel()[indices]

    k_list = np.linspace(-pi, pi, N)
    # get k-points corresponding to the max values
    dp_kx = [k_list[i] for i in dps[0]]
    dp_ky = [k_list[i] for i in dps[1]]

    plt.figure()
    ax = plt.gca()
    ax.set_aspect(1)
    plt.title(rf'$g={g:.2f}$ | $\mu={mu:.2f}$')
    plt.scatter(dp_kx, dp_ky)  # , s=area, c=colors, alpha=0.5)
    # plt.title('Scatter plot pythonspot.com')
    plt.xticks([- pi, -pi / 2, 0, pi / 2, pi],
               [r'-$\pi$', r'-$\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'], fontsize=fs)
    plt.yticks([- pi, -pi / 2, 0, pi / 2, pi],
               [r'-$\pi$', r'-$\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'], fontsize=fs)
    plt.xlabel(r'$k_x$', fontsize=fs)
    plt.ylabel(r'$k_y$', fontsize=fs)
    # plt.xlim([-1, 1])
    # plt.ylim([-1, 1])
    plt.tight_layout()


def berry_curvature(n, evals, evecs, grad_Hk, num_all_bands, N):

    ber_cur_band_n = np.zeros((N, N), dtype=float)
    ket_n = evecs[:, :, :, n]
    E_n = evals[:, :, n]

    for m in [x for x in range(num_all_bands) if x != n]:

        ket_m = evecs[:, :, :, m]
        E_m = evals[:, :, m]

        a = np.einsum('xyi, cxyij, xyj -> xyc',
                      ket_n.conj(), grad_Hk, ket_m, optimize=True)
        b = np.einsum('xyi, cxyij, xyj -> xyc',
                      ket_m.conj(), grad_Hk, ket_n, optimize=True)

        ber_cur_band_n += -np.imag(np.cross(a, b)) / (E_m - E_n)**2

    return ber_cur_band_n


def calc_chern(H, prnt_band_bc=False):

    N = int(np.sqrt(len(H)))
    num_all_bands = len(H[0])
    Hk = H.reshape((N, N, num_all_bands, num_all_bands))
    evals, evecs = LA.eigh(Hk)
    grad_Hk = np.gradient(Hk, axis=[0, 1])

    bc_filled_bands = np.zeros((N, N), dtype=float)
    for n in range(num_all_bands // 2):
        bc_band = berry_curvature(n, evals, evecs, grad_Hk, num_all_bands, N)
        bc_filled_bands += bc_band

        if prnt_band_bc:
            print(f'bc of band {n}: {np.sum(bc_band) / (2 * pi):.3f}')

    chern = np.sum(bc_filled_bands) / (2 * pi)

    # return abs(chern)
    return chern


def calc_gap_v0(H):

    N = int(np.sqrt(len(H)))
    evals = LA.eigvalsh(H, UPLO="U")
    evals = evals.reshape(N, N, 4)
    E0 = evals[:, :, 0]
    E1 = evals[:, :, 1]

    return - np.amax(E1)


def ener_low_pos_band(k_pt, phi, params):
    '''return the energy of the lowest positive energy band at the given k point. '''

    # obtain the Hamiltonian at just the point (kx,ky)
    H = build_Hk(phi, *params, k_pt=k_pt)

    evals = LA.eigvalsh(H)

    # pick the lowest positive energy band
    return evals[0][2 * uc]


def calc_gap(k_pt, phi, params):

    global t, t1, mu, g, uc, N

    t, d, mu, tp, g, uc, N = params

    rang = (-pi, pi)
    bnds = (rang, rang)
    # bnds = ((0, pi), rang)

    result1 = optimize.minimize(ener_low_pos_band, k_pt,
                                args=(phi, [params]),
                                # method='Nelder-Mead',
                                # method='Powell',
                                # method='BFGS',
                                # method='TNC',
                                method='SLSQP',
                                # method='L-BFGS-B',
                                bounds=bnds,
                                tol=1e-9)
    result2 = optimize.minimize(ener_low_pos_band, -k_pt + 1.2,
                                args=(phi, [params]),
                                method='Nelder-Mead',
                                tol=1e-9)
    if result1.success and result2.success:
        ans1 = round(result1.fun, 4)
        ans2 = round(result2.fun, 4)
        if ans1 != ans2:
            print('-' * 30)
            print(f'mu:{mu:.2f} | g:{g:.2f}')
            print('minimizations produce different gaps!',
                  ans1, ans2, 'using the smaller one')
            # print(result1.x, result2.x)
            # return 2 * min(ans1, ans2)
        return 2 * min(result1.fun, result2.fun)
    else:
        raise ValueError(result1.message)

    return


# %% ###################################################
