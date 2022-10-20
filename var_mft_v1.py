import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize
from gen_Hk import build_Hk_brute
from gen_Hk import build_Hk_smart
import seaborn as sns
import pandas as pd
from ext_mft_funcs import *
from numpy import linalg as LA
import itertools
from matplotlib import rc
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
fs_large = 20  # fontsize of plot labels
fs_small = 15  # fontsize of axes ticks
# rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
rc('xtick', labelsize=fs_small)
rc('ytick', labelsize=fs_small)
matplotlib.rcParams['xtick.direction'] = "in"
matplotlib.rcParams['ytick.direction'] = "in"
# comment for line legend
matplotlib.rcParams['legend.handlelength'] = 0
# matplotlib.rcParams['legend.numpoints'] = 1

np.set_printoptions(precision=4, suppress=True,
                    threshold=4, linewidth=100)
pi = np.pi

# % #######################################################


def plot_ener_vs_phase(params):
    '''plot energy as a function of the SC phase diff between layers'''
    phi_list = np.linspace(-1, 1, 50)
    e_list = []
    for phi in phi_list:
        e_list.append(ener_per_uc(phi, [params]))

    plt.figure()
    plt.plot(phi_list, e_list)
    plt.show()


def plot_heatmap(z_list, title=None, save=False):

    if title:
        plt.title(rf'{title} $t{t}$_$\Delta{d}$_$\mu{mu}$')
    plt.imshow(z_list.T, origin='lower',
               extent=[min(g_list), max(g_list),
                       min(tp_list), max(tp_list)])
    # ax = plt.gca()
    # ax.set_aspect('equal')
    plt.xlabel(r'$g$')
    plt.ylabel(r'$t^\prime$')
    plt.colorbar()
    plt.tight_layout()
    if save:
        plt.savefig(f't{t}_mu{mu}_{title}.pdf', bbox_inches='tight', format='pdf')
    plt.show()


def plot_phase_diag_v0(data, save=False):

    # sns.set_theme(style="white")
    sns.set_style("ticks")
    # sns.axes_style("ticks")
    sns.set_context('talk', rc={"font.size": 20,
                                "axes.titlesize": 20, "axes.labelsize": 20})

    g = sns.relplot(x='mu', y='g', hue='chern', size='gap',
                    sizes=(25, 125), alpha=.8, palette='muted',
                    legend='brief',
                    height=6, data=data)
    g.set(xlabel=r'$\mu$', ylabel=r'$g$', aspect="equal")

    if save:
        g.savefig(f'{name}.pdf',
                  bbox_inches='tight', format='pdf', dpi=300, transparent=False)


def plot_phase_diag(data, save=False):

    color_pall = ['lightblue', 'darkorange',
                  'magenta', 'blueviolet', 'darkgreen', 'lightgreen', 'lightslategray']
    # 'lightslategray',  'cornflowerblue', 'plum'
    legend_elements = []

    plt.figure()
    for c in [1, 2, 0, -2, -1, 'indet']:
        subset = data[data.chern == c]

        if c == 'indet':
            color = color_pall[-1]
            label = None
        else:
            if c >= 0:
                color = color_pall[c]
            else:
                color = color_pall[c + 5]
            label = rf'$C={c}$'
            legend_elements.append(Line2D([0], [0], marker='o',  color='w', label=label,
                                          markerfacecolor=color, markersize=8))

        if c == 0:
            plt.scatter(subset.mu, subset.g, s=subset.gap *
                        70, c=color, label=label)

        else:
            plt.scatter(subset.mu, subset.g, s=subset.gap *
                        70, c=color, label=label)

    plt.ylabel(r'$g$', fontsize=fs_large)
    plt.xlabel(r'$\mu$', fontsize=fs_large)
    plt.legend(loc='upper center', handles=legend_elements, fontsize=fs_small)
    # plt.legend(loc=0, numpoints=1, fontsize=fs_small)
    plt.tight_layout()
    if save:
        plt.savefig(f'{name}.pdf',
                    bbox_inches='tight', format='pdf', dpi=300, transparent=False)
    plt.show()


# plot_phase_diag(data, save)


# % #######################################################


def ener_per_uc(phi, params):

    H = build_Hk(phi, *params)
    evals = LA.eigvalsh(H, UPLO="U")
    evals = evals.reshape(N, N, 4)
    E0 = evals[:, :, 0]
    E1 = evals[:, :, 1]
    E = np.sum(E0 + E1)

    return E / N**2


def phase_with_min_ener_v0(params):
    ''' find the phi with min energy for a given twist angle'''

    result2 = optimize.minimize(ener_per_uc, phi,
                                args=([params]),
                                method='Nelder-Mead',
                                tol=1e-9)

    if result.success:
        return result.x[0]
    else:
        print(params)
        raise ValueError(result.message)


def phase_with_min_ener(params):
    ''' find the phi with min energy for a given twist angle'''

    result1 = optimize.minimize(ener_per_uc, phi,
                                args=([params]),
                                # method='Nelder-Mead',
                                # method='Powell',
                                # method='BFGS',
                                method='TNC',
                                # method='SLSQP',
                                # method='L-BFGS-B',
                                bounds=((1e-6, 1.0 + 1e-6),),
                                tol=1e-9)
    result2 = optimize.minimize(ener_per_uc, phi - .2,
                                args=([params]),
                                method='Nelder-Mead',
                                tol=1e-9)

    ans1 = round(result1.x[0], 1)
    ans2 = round(result2.x[0], 1)

    if result1.success and result2.success:
        if ans1 == ans2:
            return result2.x[0]
        elif ans1 != ans2:
            print('-' * 30)
            print(f'mu:{mu:.2f} | g:{g:.2f}')
            print('two different phases produced',
                  ans1, ans2, 'Using 0.5')
            plot_ener_vs_phase(params)
            return 0.5
    elif result2.success:
        # print('params', params)
        print('-' * 30)
        print(f'mu:{mu:.2f} | g:{g:.2f}')
        print('two phases produced', ans1, ans2,
              'Using optimal phase from Nelder-Mead')
        print('TNC error message:', result1.message)
        plot_ener_vs_phase(params)
        return result2.x[0]
    else:
        print('params', params)
        plot_ener_vs_phase(params)
        print(result1.message)
        print(result2.message)
        raise ValueError()


def phase_diag(save=False):

    df = pd.DataFrame({'chern': [0.0, 1.0, 2.0, 'indet']})
    for g in g_list:
        for tp in tp_list:
            for mu in mu_list:

                params = t, d, mu, tp, g, uc, N

                phse = phase_with_min_ener(params)

                gap = calc_gap(k_pt, phse, params)

                params = params[:-1] + (200, )
                # changes the number of unitcells (and hence the size of the BZ) to 200*200
                H = build_Hk(phse, params)
                chrn = calc_chern(H, prnt_band_bc=False)

                print(f'g:{g:.2f}|mu:{mu:.2f}|C:{chrn:.3f}|gap:{gap:.2f}|phase:{phse:.2f}')

                if chrn > 2.5 or chrn < -2.5:
                    chrn = 'indet'
                else:
                    chrn = round(chrn)

                data = pd.DataFrame({'g': [g], 'mu': [mu],
                                     'tp': [tp],
                                     'phase': [phse],
                                     'gap': [gap],
                                     'chern': [chrn]})
                df = pd.concat([df, data], axis=0)

    # print(df.info())
    if save:
        df.to_pickle(f'{name}.pkl')
    return df


# % #######################################################

N = 40      # length of wach wire
t = 1.0     # intra-wire tunneling
d = 0.8     # SC order parameter
mu = -2.3   # chemical potential in wach wire
tp = .5     # inter-wire tunneling
g = 0.8    # interlayer coupling
phi = 0.5   # phase difference in units of pi
uc = 1      # sites in the unit cell in each layer
num_all_bands = 4 * uc
k_pt = np.array([-0.8, 1.7])  # guess for finding the minimum gap
params = t, d, mu, tp, g, uc, N
tp_list = [tp]

# g_len, mu_len = 2, 2
# g_list = np.linspace(0.5, 1, g_len)
# mu_list = np.linspace(-2, -1, mu_len)

### for paper ###
g_len, mu_len = 10, 30  # num of data points along the axes
g_list = np.linspace(0.05, 1, g_len)
mu_list = np.linspace(-3.7, 3.7, mu_len)

save = False
name = f'./data/phase_diag_tp{tp:.1f}_d{d:.1f}'

# build_Hk = build_Hk_brute
build_Hk = build_Hk_smart

read_saved = False
if read_saved:
    data = pd.read_pickle(f'{name}.pkl')
    plot_phase_diag(data, save)
else:
    # ener_per_uc(phi, [params])
    # phase_with_min_ener(params)
    # plot_ener_vs_phase(params)

    data = phase_diag(save)
    plot_phase_diag(data, save)


# % ##################################

# params = params[:-1] + (200, )
# # changes the number of unitcells (and hence the size of the BZ) to 200*200
# H = build_Hk(phi, params)
# chrn = calc_chern(H)
# print(f'C:{chrn:.3f}')


# %% #######################################################

'''
def build_Hk_single_layer(params, layer):
    t, d, mu, tp, g, uc, N = params

    # % #############################################

    H = np.zeros((len(kx), 2 * uc, 2 * uc), dtype="complex")

    if layer == 1:
        e_k_top = -2 * t * np.cos(kx) - 2 * tp * np.cos(ky) - mu
        # normal terms
        H[:, 0, 0] = e_k_top
        H[:, 1, 1] = -1.0 * e_k_top
        # pairing terms
        H[:, 0, 1] = 2.0j * d * np.sin(kx)
        H[:, 1, 0] = np.conjugate(2.0j * d * np.sin(kx))

    elif layer == 2:
        e_k_bot = -2 * t * np.cos(ky) - 2 * tp * np.cos(kx) - mu
        # normal terms
        H[:, 0, 0] = e_k_bot
        H[:, 1, 1] = -1.0 * e_k_bot
        # pairing terms
        H[:, 0, 1] = 2.0j * d * np.sin(ky)
        H[:, 1, 0] = np.conjugate(2.0j * d * np.sin(ky))

    else:
        raise ValueError('invalid layer')

    return H


def plot_band_contour(kx_mesh, ky_mesh, band):

    plt.figure()
    ax = plt.gca()
    ax.set_aspect(1)
    plt.contourf(kx_mesh, ky_mesh, band,
                 levels=np.linspace(np.amin(band), np.amax(band), 50))
    # plt.xticks([- pi, -pi / 2, 0, pi / 2, pi],
    #            [r'-$\pi$', r'-$\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'], fontsize=fs)
    # plt.yticks([- pi, -pi / 2, 0, pi / 2, pi],
    #            [r'-$\pi$', r'-$\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'], fontsize=fs)
    plt.xlabel(r'$k_x$', fontsize=fs)
    plt.ylabel(r'$k_y$', fontsize=fs)
    plt.colorbar()
    plt.tight_layout()
    # plt.savefig(f'../notes/figures/bs_tbl_d{d}_g{g}_phi{phi}.pdf')
    plt.savefig(f'lay1_d.pdf', bbox_inches='tight')
    plt.show()


def MF_band_struc(params):

    t, d, mu, tp, g, uc, N = params

    H = build_Hk_single_layer(params, layer)
    num_of_neg_bnds = 1

    #####################

    evals = LA.eigvalsh(H)
    evals = evals.reshape(N, N, num_of_neg_bnds * 2)

    bands = []
    for i in range(num_of_neg_bnds * 2):
        band_i = evals[:, :, i]
        bands.append(band_i)

    #####################

    plot_band_contour(kx_mesh, ky_mesh, bands[num_of_neg_bnds - 1])


N = 80      # length of wach wire
t = 0.7      # intra-wire tunneling
d = 1.0  # SC order parameter
mu = -0.5     # chemical potential in wach wire
tp = t   # inter-wire tunneling
g = 0.0      # interlayer coupling
phi = 0.0    # phase difference in units of pi
uc = 1  # sites in the unit cell in each layer
num_all_bands = 2 * uc
params = t, d, mu, tp, g, uc, N
layer = 2

# k_list = ((np.arange(N) / N) - 0.5) * 2 * pi
# k_list = np.linspace(0, 2 * pi, N)
k_list = np.linspace(-pi, pi, N)
kx_mesh, ky_mesh = np.meshgrid(k_list, k_list)
kx = kx_mesh.flatten()
ky = ky_mesh.flatten()

# MF_band_struc(params)
H = build_Hk_single_layer(params, layer)
calc_chern(H)
'''

# %% #######################################################
