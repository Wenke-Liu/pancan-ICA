import numpy as np
import pandas as pd
import hdbscan
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import pairwise_distances
from sklearn.manifold import MDS
from sklearn.manifold import TSNE


def multi_ica(data):
    pass


def ic_cluster(ics, name='ics',
               show_plt=False,
               save_plt=True,
               save_output=True):

    dis = pairwise_distances(ics, metric='cosine')
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, metric='precomputed')
    clusterer.fit(dis)
    ics_cls = clusterer.labels_.reshape((clusterer.labels_.shape[0], 1))
    tsne_embedding = TSNE(n_components=2, metric='precomputed')  #tSNE
    ics_tsne = tsne_embedding.fit_transform(dis)
    mds_embedding = MDS(dissimilarity='precomputed')  # MDS
    ics_mds = mds_embedding.fit_transform(dis)

    ics_pts = np.hstack((ics_tsne, ics_mds, ics_cls))
    plt.figure(figsize=(5, 5))
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

