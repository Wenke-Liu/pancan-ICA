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
from scipy import stats


def multi_ica(data,
              whiten=True,
              n_components=None,
              n_repeats=100,
              seed=None):
    if seed:
        np.random.seed(seed)
        run_seeds = np.random.randint(low=0, high=10000, size=n_repeats)
    else:
        run_seeds = [None] * n_repeats

    if not n_components:
        n_components = data.shape[1]

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


def align_tail(s_list, a_list, cutoff=3):
    ics = np.hstack(tuple(s_list)).T  # ics shape: n_components*n_repeats, n_genes
    mixs = np.hstack(tuple(a_list)).T  # mixs shape: n_components*n_repeats, n_samples
    for i in range(ics.shape[0]):
        ic = ics[i, :]
        sd = ic.std()
        ic = ic/sd  # normalize the std of compoments as 1
        w = mixs[i, :]
        w = w*sd
        if sum(ic < (-cutoff)) > sum(ic > cutoff):
            ics[i, :] = -ic
            mixs[i, :] = -w
    return ics, mixs


def dis_cal(ics, metric='cosine', name='ics', save_dis=True, out_dir='.'):
    dis = pairwise_distances(ics, metric=metric, n_jobs=-1)
    if save_dis:
        hf = h5py.File(out_dir + '/' + name + '_dis.h5', 'w')
        hf.create_dataset('dis', data=dis)
        hf.close()
    return dis


def ic_cluster(dis,
               name='ics',
               min_cluster_size=5,
               min_samples=5,
               show_plt=False,
               save_plt=True,
               figsize=5,
               out_dir='.'):

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric='precomputed')
    clusterer.fit(dis)
    ics_cls = clusterer.labels_.reshape((clusterer.labels_.shape[0], 1))
    tsne_embedding = TSNE(n_components=2, metric='precomputed')  # tSNE
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
        plt.savefig(fname=out_dir + '/' + name + '.png')

    return ics_pts


def find_cor(mix, clinical):
    """
    Find p_value of linear regression between mixing scores and clinical variables.
    return a raw p value table.
    """
    mix_dat = mix[clinical.index.values]
    res = []
    for i in range(mix.shape[0]):
        row_res = []
        reg_x = mix.iloc[i, :]
        for variable in clinical.columns.values:
            reg_y = clinical[variable]
            dat = pd.concat([reg_x, reg_y], axis=1, join='inner').dropna()
            x = np.array(dat.iloc[:, 0]).astype(float)
            y = np.array(dat.iloc[:, 1]).astype(float)
            _, _, _, p_value, _ = stats.linregress(x=x, y=y)
            row_res.append(p_value)
        res.append(row_res)

    res = np.array(res)
    res = pd.DataFrame(res, columns=clinical.columns.values, index=mix.index.values)
    res = pd.concat([mix[['i_repeats', 'i_comps', 'cluster']], res], axis=1)
    return res

