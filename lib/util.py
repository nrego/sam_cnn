## Basic two layer NN for sam surface (input vec is all positions)

import numpy as np

import numpy as np
import pickle

from scipy.spatial import cKDTree

from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


## Miscellaneous helper functions ##

colors = [(0,0,1), (0.85,0.85,0.85)]
mymap = LinearSegmentedColormap.from_list('mymap', colors, N=100)

do_cnn = False
no_run = False


## Generate a hexagonally-packed grid of points
def gen_pos_grid(ny=6, nz=None, z_offset=False, shift_y=0, shift_z=0):
    if nz is None:
        nz = ny
    ## Generate grid of center points
    z_space = 0.5 # 0.5 nm spacing
    y_space = np.sqrt(3)/2.0 * z_space

    y_pos = 0 + shift_y*y_space
    pos_row = np.arange(0,0.5*(nz+1), 0.5) + shift_z*z_space

    positions = []
    for i in range(ny):
        if not z_offset:
            this_pos_row = pos_row if i % 2 == 0 else pos_row + z_space/2.0
        else:
            this_pos_row = pos_row if i % 2 != 0 else pos_row + z_space/2.0


        for j in range(nz):
            z_pos = this_pos_row[j]
            positions.append(np.array([y_pos, z_pos]))

        y_pos += y_space


    return np.array(positions)

def plot_pattern(pos_ext, patch_indices, methyl_mask):
    pos = pos_ext[patch_indices]
    plt.plot(pos_ext[:,0], pos_ext[:,1], 'bo')
    plt.plot(pos[methyl_mask, 0], pos[methyl_mask, 1], 'ko')

    #plt.show()

# Plot a feature from a feature vector corresponding to a list of points
#   Plots each point colored according its feature
def plot_from_feat(pos_ext, feat, this_map=mymap):
    fig, ax = plt.subplots(figsize=(6,7))
    ax.scatter(pos_ext[:,0], pos_ext[:,1], c=feat, cmap=this_map, s=400)
    ax.set_xticks([])
    ax.set_yticks([])

    #plt.show()

# Generate 6 rotated patch patterns on extended grid
#   non-patch hydroxyl: 0
#   patch hydroxyl: -1
#   patch methyl: +1
def hex_rotate(methyl_mask, ny=13, nz=11):

    positions = gen_pos_grid(6)
    theta_60 = (60*np.pi)/180.
    rot_60 = np.array([[np.cos(theta_60), -np.sin(theta_60)], [np.sin(theta_60), np.cos(theta_60)]])

    z_space = 0.5
    y_space = np.sqrt(3)*0.5 * z_space

    pos_ext = gen_pos_grid(ny, nz, z_offset=True)
    ext_y_min, ext_z_min = pos_ext.min(axis=0)
    ext_y_max, ext_z_max = pos_ext.max(axis=0)

    ext_y_len = ext_y_max - ext_y_min
    ext_z_len = ext_z_max - ext_z_min - 0.5*z_space

    for i in range(6):
        rot_mat = np.array( np.matrix(rot_60)**i )

        # Rotate 6x6 grid and find mapping to pos_ext
        this_pos = np.dot(positions, rot_mat)

        y_min, z_min = this_pos.min(axis=0)
        y_max, z_max = this_pos.max(axis=0)

        y_len = y_max - y_min
        z_len = z_max - z_min

        # Shift onto pos_ext grid, and center as best as we can
        shift_y = y_space * np.round(0.5*(ext_y_len - y_len) / y_space)
        shift_z = z_space * np.round(0.5*(ext_z_len - z_len) / z_space) #+ 0.25
        shift = np.array([ext_y_min-y_min + shift_y, ext_z_min-z_min + shift_z])

        this_pos += shift

        # Check if we're on-register with the z-shift
        py = this_pos[0,0]
        idx = np.argmin(np.abs(pos_ext[:,0] - py))
        on_register = np.abs(this_pos[0,1] - pos_ext[idx, 1]) % z_space < 0.4*z_space
        if not on_register:
            this_pos += np.array([0, 0.5*z_space])
        #print("On register? {}".format(on_register))

        #plt.plot(pos_ext[:,0], pos_ext[:,1], 'x')
        #plt.plot(this_pos[:,0], this_pos[:,1], 'o')
        
        #plt.show()

        # patch_indices: shape: (36,): patch_indices[i] gives global index (on pos_ext)
        #   of local patch point i
        d, patch_indices = cKDTree(pos_ext).query(this_pos, k=1)
        assert np.unique(patch_indices).size == patch_indices.size == 36

        patch_methyl_indices = patch_indices[methyl_mask]
        patch_hydroxyl_indices = patch_indices[~methyl_mask]
        #plt.plot(pos_ext[patch_methyl_indices, 0], pos_ext[patch_methyl_indices, 1], 'ko')
        #plt.plot(pos_ext[patch_hydroxyl_indices, 0], pos_ext[patch_hydroxyl_indices, 1], 'bo')

        augmented_feature = np.zeros(pos_ext.shape[0])
        augmented_feature[patch_methyl_indices] = 1
        augmented_feature[patch_hydroxyl_indices] = -1


        yield augmented_feature


def hex_augment_data(feat_vec, y, ny=13, nz=11):
    n_feat = feat_vec.shape[0]
    n_aug = n_feat*6

    aug_feat_vec = np.zeros((n_aug, ny*nz))

    if y.ndim == 1:
        aug_y = np.zeros(n_aug)
    else:
        aug_y = np.zeros((n_aug, y.shape[1]))

    for i_feat in range(n_feat):
        feat = feat_vec[i_feat]
        this_y = y[i_feat]
        methyl_mask = feat.astype(bool)
        gen_hex = hex_rotate(methyl_mask, ny, nz)

        for i, aug_feat in enumerate(gen_hex):
            idx = i_feat + i*n_feat

            aug_feat_vec[idx] = aug_feat
            aug_y[idx] = this_y


    return (aug_feat_vec, aug_y)

# Flip every dataset to get a 'new' feature (except for k=0, k=36)
def augment_data(feat_vec, y):
    n_feat = feat_vec.shape[0]

    n_aug = n_feat*2 - 2 # no aug for k=0, k=36

    aug_feat_vec = np.zeros((n_aug, feat_vec.shape[1]))
    aug_feat_vec[:n_feat, :] = feat_vec

    if y.ndim == 1:
        aug_y = np.zeros(n_aug)
    else:
        aug_y = np.zeros((n_aug, y.shape[1]))
    aug_y[:n_feat] = y

    for i in range(n_feat, n_aug):
        orig_feat = feat_vec[i - n_feat]
        orig_y = y[i - n_feat]

        # Rasterize, flip axes, and re-ravel
        new_feat = orig_feat.reshape(6,6)[::-1, ::-1].ravel()

        aug_feat_vec[i] = new_feat
        aug_y[i] = orig_y


    return (aug_feat_vec, aug_y)

# Load in data (energies and methyl positions)
def load_and_prep(fname='sam_pattern_data.dat.npz'):
    ds = np.load(fname)
    energies = ds['energies']
    k_vals = ds['k_vals']

    # (y,z) positions of each of the 36 hexagonal points on the 6x6 grid, flattened
    # shape: (36, 2)
    positions = ds['positions']
    # details the methyl positions of each config
    # Shape: (n_data, 6x6)
    methyl_pos = ds['methyl_pos']
    poly_4 = ds['poly_4']
    poly_5 = ds['poly_5']

    beta_phi_stars = ds['beta_phi_stars']

    n_data = energies.size

    pos_ext = positions.copy()

    # patch_idx is list of patch indices in pos_ext 
    #   (pos_ext[patch_indices[i]] will give position[i], ith patch point)
    d, patch_indices = cKDTree(pos_ext).query(positions, k=1)

    tree = cKDTree(pos_ext)
    neighbors = tree.query_ball_tree(tree, r=0.51)

    adj_mat = np.zeros((pos_ext.shape[0], pos_ext.shape[0]), dtype=np.uint8)
    for i in range(pos_ext.shape[0]):
        indices = np.array(neighbors[i])
        adj_mat[i, indices] = 1
        #adj_mat[i,i] = 0

    # shape: (n_data_points, 12*12)
    feat_vec = np.zeros((n_data, pos_ext.shape[0]), dtype=np.float32) # might as well keep this shit small

    for i_dat, methyl_mask in enumerate(methyl_pos):
        feat_vec[i_dat][patch_indices] = methyl_mask

    f_mean = feat_vec.mean()
    f_std = feat_vec.std()


    return feat_vec, energies, poly_4, beta_phi_stars, pos_ext, patch_indices, methyl_pos, positions


def save_net(net, foutname='net.pkl'):
    with open(foutname, 'wb') as fout:
        pickle.dump(net, fout)

def load_net(fname='net.pkl'):
    with open(fname, 'rb') as fin:
        return pickle.load(fin)


## Some SAM data extraction/analysis functions to load in the dataset in sam_pattern_data.dat.npz ##

# Given a list of patch atom positions, and an extended list of positions (including non-patch atoms), generate
#  determine a list of neighbors to each patch atom
#
#  returns nn, nn_ext, d, d_ext
#    nn: nearest neighbor patch indices
#         for patch index i, nn[i] = {j}; j is index of patch atom adjacent to atom i
#    nn_ext: nearest neighbor extended (non-patch) indices
def construct_neighbor_dist_lists(positions, pos_ext):

    n_positions = positions.shape[0]
    n_pos_ext = pos_ext.shape[0]
    ## Set up dict of nearest neighbors
    tree = cKDTree(positions)
    pairs = tree.query_pairs(r=0.51)

    tree_ext = cKDTree(pos_ext)
    ext_neighbors = tree.query_ball_tree(tree_ext, r=0.51)

    d_self, i_self = tree.query(positions, k=n_positions)
    d_ext, i_ext = tree_ext.query(positions, k=n_pos_ext)

    # Dict of nearest neighbor patch indices
    #   for patch index i, nn[i] = {j}; j is index of patch atom that is nearest neighbor to patch atom i
    # Dict of nearest neighbor extended (non-patch) indices
    #    for patch index i, nn_ext[i] = {k}; k is index of extended pos (non-patch) atom that is nearest neighbor to patch atom i
    nn = dict()
    nn_ext = dict()
    for i in range(n_positions):
        nn[i] = np.array([], dtype=int)
        nn_ext[i] = np.array(ext_neighbors[i], dtype=int)
    for i,j in pairs:
        assert j>i
        nn[i] = np.sort(np.append(nn[i], j))
        nn[j] = np.sort(np.append(nn[j], i))
        

    # Dict of patch i's distance to every other patch point - its distance to itself is set to infinity
    #  dd[i,k] is distance from point i to point k
    dd = np.zeros((n_positions,n_positions))
    dd_ext = np.zeros((n_positions,n_pos_ext))
    for i in range(n_positions):
        try:
            assert np.array_equal(np.sort(i_self[i]), np.arange(n_positions))
            sort_idx_self = np.argsort(i_self[i])
            dd[i] = d_self[i][sort_idx_self] #- 0.5
        except:
            pass
        dd[i,i] = np.inf

        assert np.array_equal(np.sort(i_ext[i]), np.arange(n_pos_ext))
        sort_idx_ext = np.argsort(i_ext[i])
        dd_ext[i] = d_ext[i][sort_idx_ext] #- 0.5


    return nn, nn_ext, dd, dd_ext


def extract_data(fname="sam_pattern_data.dat.npz"):
    ds = np.load(fname)

    energies = ds['energies']
    k_vals = ds['k_vals']
    methyl_pos = ds['methyl_pos']
    positions = ds['positions']

    # All methyl patterns
    methyl_pos = ds['methyl_pos']
    n_configs = methyl_pos.shape[0]

    # extended grid so we can find patch indices on edge of patch
    pos_ext = gen_pos_grid(8, z_offset=True, shift_y=-1, shift_z=-1)

    # patch_indices is a list of the (global) indices of points on pos_ext corresponding
    #   to the patch points on positions
    #
    #   (pos_ext[patch_indices[i]] will give position[i], ith patch point)
    d, patch_indices = cKDTree(pos_ext).query(positions, k=1)

    # nn_ext is dictionary of (global) nearest neighbors to each (local) patch point
    #   nn_ext: [local position i] => [list of global pos_ext nearest neighbors]
    #   nn_ext[i]  global idxs of neighbor to local patch i 
    nn, nn_ext, dd, dd_ext = construct_neighbor_dist_lists(positions, pos_ext)

    # edges is a list of (i,j) tuples of (global) indices
    #   only contains nn edges between either patch-patch or patch-external
    # ext_indices contains indices of all patch-external edges
    edges, ext_indices = enumerate_edges(positions, pos_ext, nn_ext, patch_indices)
    n_edges = edges.shape[0]

    int_indices = np.setdiff1d(np.arange(n_edges), ext_indices)


    return (energies, methyl_pos, k_vals, positions, pos_ext, patch_indices, nn, nn_ext, edges, ext_indices, int_indices)


