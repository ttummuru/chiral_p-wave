from gen_Hk import build_Hk_smart
import matplotlib
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib
from matplotlib import rc
fs_large = 20  # fontsize of plot labels
fs_small = 15  # fontsize of axes ticks
# rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
rc('xtick', labelsize=fs_small)
rc('ytick', labelsize=fs_small)
matplotlib.rcParams['xtick.direction'] = "in"
matplotlib.rcParams['ytick.direction'] = "in"
# comment for line legend
# matplotlib.rcParams['legend.handlelength'] = 0
matplotlib.rcParams['legend.numpoints'] = 1

np.set_printoptions(precision=2, suppress=False,
                    threshold=10, linewidth=100)
pi = np.pi
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

# % #######################################################


def plot_edges(Nx, evals, pos_of_states, save, wid):

    ###############################

    fig, axs = plt.subplots(1, 1, sharex=True, sharey=True)

    # fig = plt.figure()
    # ax = fig.add_subplot(111)

    # Create a continuous norm to map from data points to colors
    # norm = plt.Normalize(pos_of_states.min(), pos_of_states.max())
    norm = plt.Normalize(0, 1)
    plt.axvline(x=pi, color='grey', linestyle='--', zorder=0)
    plt.axhline(y=0, color='grey', linestyle='--', zorder=0)

    for i in range(len(evals[0])):
        E = evals[:, i]
        pos_one_band = pos_of_states[:, i]

        points = np.array([kx, E]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap='cividis', norm=norm)

        # Set the values used for colormapping
        lc.set_array(pos_one_band)
        lc.set_linewidth(1.5)
        line = axs.add_collection(lc)

    fig.colorbar(line, ax=axs)

    # plt.title(
    #     rf'$t^\prime={tp:.1f}$, $\mu = {mu:.1f}$, $g={g:.1f}$, $\phi={phi:.1f}\pi$', fontsize=fs_small)
    axs.set_xlim(kx.min(), kx.max())
    if wid:
        axs.set_ylim(-wid, wid)
    axs.set_xlabel(r'$k$', fontsize=fs_large)
    axs.set_ylabel(r'$E_k$', fontsize=fs_large)

    # plt.xticks([0, pi / 2,  pi, 3 * pi / 2, 2 * pi],
    #            [r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'], fontsize=fs_small)
    plt.xticks([3 * pi / 4,  pi, 5 * pi / 4],
               [r'$3\pi/4$', r'$\pi$', r'$5\pi/4$'], fontsize=fs_small)
    # plt.yticks(np.array(range(-wid, wid + 1)), fontsize=fs - 4)

    if save:
        plt.savefig(f'{name}.pdf',
                    bbox_inches='tight', format='pdf', dpi=600, transparent=False)
    plt.show()


def MF_band_struc_strip(phi, params, save=False):

    Hk = build_Hk_smart(phi, params, geo='cylinder', k_pt=None)
    evals, evecs = LA.eigh(Hk)

    # t, d, mu, tp, g, uc, Nx, Ny = params

    ###############################

    site_labl_Y = np.diag(np.array(range(1, Ny + 1)))
    unt_cell_space = np.diag(np.array([1] * uc))
    pos_mat_lay1_Adag_A_block = np.kron(unt_cell_space, site_labl_Y)
    pos_mat_lay1 = np.kron(s0, pos_mat_lay1_Adag_A_block)
    pos_mat_full = np.kron(s0, pos_mat_lay1)

    pos_of_states = []
    for evecs_kpoint in evecs:
        mat = []
        for i in range(len(evecs_kpoint)):
            evec = evecs_kpoint[:, i]
            pos = np.real_if_close(
                np.matmul(evec.conj().T, np.matmul(pos_mat_full, evec)),
                tol=1e+4)
            mat.append(float(pos))
        pos_of_states.append(mat)
    pos_of_states = np.array(pos_of_states) / Ny

    ###############################

    if save:
        np.savez(f'{name}',
                 t=t, tp=tp, d=d, g=g, mu=mu, phase=phi,
                 kx=kx, N_perd=Nx, N_open=Ny,
                 evals=evals, pos_of_states=pos_of_states)

    return evals, pos_of_states


# % #######################################################

Nx, Ny = 300, 150    # N_perd, N_open
### C = 1 ###
g = 0.8      # interlayer coupling
mu = -2.4     # chemical potential in wach wire
wid = 0.3
### C = 2 ###
# g = 0.5      # interlayer coupling
# mu = -1.8     # chemical potential in wach wire
# wid = 0.15
###############
t = 1.0      # intra-wire tunneling
tp = 0.3   # inter-wire tunneling
phi = 0.5    # phase difference in units of pi
d = 0.2  # SC order parameter
uc = 1  # sites in the unit cell in each layer

####### full BZ ########
# # kx = ((np.arange(Nx) / Nx) - 0.5) * 2 * pi
# kx = np.linspace(0, 2 * pi, Nx)
# name = f'./data/edges_Ny{Ny}_tp{tp:.1f}_d{d:.1f}_g{g:.1f}_mu{mu:.1f}'

####### BZ around \pi ########
kx = np.linspace(3 * pi / 4, 5 * pi / 4, Nx)
name = f'./data/edges_Ny{Ny}_tp{tp:.1f}_d{d:.1f}_g{g:.1f}_mu{mu:.1f}_zoom'

save = False
params = t, d, mu, tp, g, uc, Nx, Ny, kx

read_saved = True
if read_saved:

    # name = './data/edges_Ny150_tp0.3_g0.8_mu-2.4_zoom.npz'

    data = np.load(f'{name}.npz')
    t = data['t']
    tp = data['tp']
    g = data['g']
    d = data['d']
    mu = data['mu']
    phase = data['phase']
    N_open = data['N_open']
    N_perd = data['N_perd']
    kx = data['kx']
    evals = data['evals']
    pos_of_states = data['pos_of_states']

    name = f'./data/edges_Ny{Ny}_tp{tp:.1f}_g{g:.1f}_mu{mu:.1f}_zoom'

    plot_edges(Nx, evals, pos_of_states, save=save, wid=.5)

else:
    evals, pos_of_states = MF_band_struc_strip(phi, params, save)
    plot_edges(Nx, evals, pos_of_states, save=save, wid=wid)

# %% #######################################################
