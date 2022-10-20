import numpy as np
from gen_terms import generate
# from numpy import linalg as LA
# import matplotlib
# import matplotlib.pyplot as plt
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

# % ###################################################


def build_Hk_brute(phi, params, k_pt=None):

    t, d, mu, tp, g, uc, N = params
    uc_size = uc * 2

    k_list = ((np.arange(N) / N) - 0.5) * 2 * pi
    kx_mesh, ky_mesh = np.meshgrid(k_list, k_list)
    kx = kx_mesh.flatten()
    ky = ky_mesh.flatten()

    try:
        kx, ky = [k_pt[0]], [k_pt[1]]
    except:
        pass

    # % #############################################

    H = np.zeros((len(kx), 2 * uc_size, 2 * uc_size), dtype="complex")

    e_k_top = -2 * t * np.cos(kx) - 2 * tp * np.cos(ky) - mu
    e_k_bot = -2 * t * np.cos(ky) - 2 * tp * np.cos(kx) - mu

    # normal terms
    H[:, 0, 0] = e_k_top
    H[:, 1, 1] = -1.0 * e_k_top
    # ----
    H[:, 2, 2] = e_k_bot
    H[:, 3, 3] = -1.0 * e_k_bot

    # pairing terms
    H[:, 0, 1] = 2.0j * d * np.sin(kx)
    H[:, 1, 0] = np.conjugate(2.0j * d * np.sin(kx))
    # ----
    d_lay2 = d * np.exp(1j * phi * pi)
    H[:, 2, 3] = 2.0j * d_lay2 * np.sin(ky)
    H[:, 3, 2] = np.conjugate(2.0j * d_lay2 * np.sin(ky))

    # interlayer coupling
    H[:, 0, 2] = g
    H[:, 2, 0] = g
    H[:, 1, 3] = -1.0 * g
    H[:, 3, 1] = -1.0 * g

    return H


# % ###################################################


def get_matrix(N, off_diag):
    ''' N: size of the matrix
    off_diag: represents the distance of the non - zero diagonal from the longest diagonal '''

    off_diag = int(off_diag)
    # print(off_diag)
    return np.diag(np.ones(N - abs(off_diag)), k=off_diag)


def insert_element(mat, term, type):

    site_1 = term[0]
    site_2 = term[1]
    lay = term[4][0]
    pos_vec = term[3]

    # % ###################################################

    if type == 'norm':
        if lay == 1 and term[2] == '+x':
            amp = t
        elif lay == 1 and term[2] == '+y':
            amp = tp
        elif lay == 2 and term[2] == '+y':
            amp = t
        elif lay == 2 and term[2] == '+x':
            amp = tp
        else:
            raise ValueError('unrecognized layer and bond direction')

    elif type == 'pair':
        if lay == 1 and term[2] == '+x':
            amp = d
        elif lay == 2 and term[2] == '+y':
            amp = d
        else:
            amp = 0

    elif type == 'coup':
        amp = g

    else:
        raise ValueError('term kind not recognized')

    # % ###################################################

    if geometry == 'torus':
        k_r = kx * pos_vec[0] + ky * pos_vec[1]

        mat[:, site_1, site_2] += amp * np.exp(1j * k_r)

    elif geometry == 'cylinder':
        k_r = kx * pos_vec[0]

        inter_u_cell_space = get_matrix(Ny, pos_vec[1])
        sub_latt_space = np.zeros((num_k_points, uc, uc), dtype=complex)

        sub_latt_space[:, site_1, site_2] += amp * np.exp(1j * k_r)

        mat += np.kron(sub_latt_space, inter_u_cell_space)

    else:
        raise ValueError('unrecognized geometry!')

    # % ###################################################

    return mat


def single_layer_Hk(terms, phase):

    AdagA_block = \
        np.zeros((num_k_points, dim_sml_blck, dim_sml_blck), dtype=complex)
    AA_block = \
        np.zeros((num_k_points, dim_sml_blck, dim_sml_blck), dtype=complex)

    for term in terms:
        AdagA_block = \
            insert_element(AdagA_block, term, type='norm')
        AA_block = \
            insert_element(AA_block, term, type='pair')

    ###############

    AdagA_block = AdagA_block + AdagA_block.transpose((0, 2, 1)).conj()
    AA_block = AA_block - AA_block.transpose((0, 2, 1)).conj()
    # because of the fact that it is spin triplet pairing, the sum has to split to two: kx>0 and kx<0. As a result of anti-commutation, the negative sign accounts comes about.

    ###############

    chem = -mu * np.eye(dim_sml_blck, dtype=complex)

    AdagA_block = AdagA_block + chem
    # adds chem to each array in AdagA_block. Each array represents one k-point

    ###############

    AA_block = AA_block * np.exp(1j * phase * pi)

    ###############

    H_norm = np.kron(s_00, AdagA_block) + \
        np.kron(s_11, -AdagA_block)

    H_pair = np.kron(s_10, AA_block) + \
        np.kron(s_01, AA_block.transpose((0, 2, 1)).conj())

    H = H_norm + H_pair

    assert np.allclose(H, H.transpose((0, 2, 1)).conj(),
                       rtol=1e-05, atol=1e-08)
    return H


def coupled_layers_Hk(phi, terms_all):

    H_A = single_layer_Hk(terms_all['1'], phase=0)
    H_B = single_layer_Hk(terms_all['2'], phase=phi)

    #####################

    AdagB_block = \
        np.zeros((num_k_points, dim_sml_blck, dim_sml_blck), dtype=complex)

    for term in terms_all['12']:
        AdagB_block = insert_element(AdagB_block, term, type='coup')

    AB_block = np.kron(s_00, AdagB_block) + \
        np.kron(s_11, -AdagB_block)
    # the second term (lower-right block in AB_block) has a negative sign because this block has terms that correspond to the Hermitian conjugated down spin terms.

    #####################

    H_c = np.kron(s_01, AB_block) + \
        np.kron(s_10, AB_block.transpose((0, 2, 1)).conj())
    # upper-right block + lower-left block

    H = np.kron(s_00, H_A) + np.kron(s_11, H_B) + H_c

    assert np.allclose(H, H.transpose((0, 2, 1)).conj(),
                       rtol=1e-05, atol=1e-08)
    return H


def build_Hk_smart(phi, params, geo='torus', k_pt=None):

    global kx, ky, t, d, mu, tp, g, uc, N, Nx, Ny
    global num_k_points, dim_sml_blck, geometry
    geometry = geo

    # % ###################################################

    if geo == 'torus':
        t, d, mu, tp, g, uc, N = params
        num_k_points = N**2
        dim_sml_blck = uc

        k_list = ((np.arange(N) / N) - 0.5) \
            * 2 * pi
        kx_mesh, ky_mesh = np.meshgrid(k_list, k_list)
        kx = kx_mesh.flatten()
        ky = ky_mesh.flatten()

        # % ############################################

        try:
            kx, ky = np.array([k_pt[0]]), np.array([k_pt[1]])
            num_k_points = 1
        except:
            pass

    # % ###################################################

    elif geo == 'cylinder':
        t, d, mu, tp, g, uc, Nx, Ny, kx = params
        num_k_points = Nx
        dim_sml_blck = uc * Ny

        # kx = ((np.arange(Nx) / Nx) - 0.5) \
        #     * 2 * pi
        # kx = np.linspace(0, 2 * pi, Nx)

    else:
        raise ValueError('unrecognized geometry!')

    # % ############################################

    terms_all = {'1': [[0, 0, '+y', [0, 1], [1, 1]],
                       [0, 0, '+x', [1, 0], [1, 1]]],
                 '2': [[0, 0, '+y', [0, 1], [2, 2]],
                       [0, 0, '+x', [1, 0], [2, 2]]],
                 '12': [[0, 0, 1.0, [0, 0], [1, 2]]]}

    # % ###################################################

    return coupled_layers_Hk(phi, terms_all)


# % ###################################################
'''
Nx, Ny = 25, 70    # N_perd, N_open
t = 1.0      # intra-wire tunneling
d = 1.0  # SC order parameter
mu = -2.6     # chemical potential in wach wire
tp = .3   # inter-wire tunneling
g = 0.26      # interlayer coupling
uc = 1  # sites in the unit cell in each layer
phi = 0.5    # phase difference in units of pi

params = t, d, mu, tp, g, uc, Nx, Ny

Hk = build_Hk_smart(phi, params, geo='cylinder', k_pt=None)
'''
# %% ###################################################
