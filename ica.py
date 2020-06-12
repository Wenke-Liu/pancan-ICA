import numpy as np
import pandas as pd
import h5py
import hdbscan
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import FastICA
from sklearn.metrics import pairwise_distances
from sklearn.manifold import MDS
from sklearn.manifold import TSNE


def multi_ica(data,
              whiten=True,
              n_components=10,
              n_repeats=100,
              seed=None):
    if seed:
        np.random.seed(seed)
        run_seeds = np.random.randint(low=0, high=10000, size=n_repeats)
    else:
        run_seeds = [None] * n_repeats
    comps = [str(i).zfill(3) for i in range(n_components)]
    repeats = [str(i).zfill(3) for i in range(n_repeats)]
    i_repeats = [item for item in repeats for i in range(n_components)]
    i_comps = comps * n_repeats
    s_list = []
    a_list = []
    for i in range(n_repeats):
        np.random.seed(run_seeds[i])
        w_init = np.random.rand(n_components, n_components)
        ica_transformer = FastICA(n_components=n_components,
                                  w_init=w_init,
                                  whiten=whiten)
        s = ica_transformer.fit_transform(data)  # shape: n_genes, n_components
        a = ica_transformer.mixing_  # shape: n_samples, n_components
        s_list.append(s)
        a_list.append(a)
    return s_list, a_list, i_repeats, i_comps


def dis_cal(s_list, name='ics', save_dis=True, metric='cosine'):
    ics = np.hstack(tuple(s_list)).T
    dis = pairwise_distances(ics, metric=metric, njobs=-1)
    if save_dis:
        hf = h5py.File(name + 'dis.h5', 'w')
        hf.create_dataset('dis', data=dis)
        hf.close()
    return dis


def ic_cluster(dis,
               name='ics',
               show_plt=False,
               save_plt=True,
               figsize=5,
               save_output=True):

    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, metric='precomputed')
    clusterer.fit(dis)
    ics_cls = clusterer.labels_.reshape((clusterer.labels_.shape[0], 1))
    tsne_embedding = TSNE(n_components=2, metric='precomputed')  #tSNE
    ics_tsne = tsne_embedding.fit_transform(dis)
    mds_embedding = MDS(dissimilarity='precomputed')  # MDS
    ics_mds = mds_embedding.fit_transform(dis)

    ics_pts = np.hstack((ics_tsne, ics_mds, ics_cls))
    plt.figure(figsize=(figsize, figsize))
    color_palette = sns.color_palette('Paired', 100)
    cluster_colors = [color_palette[x] if x >= 0
                      else (0.5, 0.5, 0.5)
                      for x in clusterer.labels_]
    cluster_member_colors = [sns.desaturate(x, p) for x, p in
                             zip(cluster_colors, clusterer.probabilities_)]
    plt.scatter(ics_pts[:, 0], ics_pts[:, 1], s=50, linewidth=0, c=cluster_member_colors, alpha=0.25)
    if show_plt:
        plt.show()
    if save_plt:
        plt.savefig(fname=name + '.png')
    if save_output:
        np.savetxt(fname=name + '_clustered.txt', X=ics_pts, delimiter='\t', comments='',
                   header = '\t'.join(['tsne_1', 'tsne_2', 'mds_1','mds_2', 'cls_lab']))

