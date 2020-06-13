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
               show_plt=False,
               save_plt=True,
               figsize=5,
               out_dir='.'):

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='precomputed')
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


def main():
    import argparse
    parser = argparse.ArgumentParser(description='ICA')
    parser.add_argument('--data_dir', type=str, default='.', help='Path to data table.')
    parser.add_argument('--out_dir', type=str, default='.', help='Path to outputs.')
    parser.add_argument('--exp_prefix', type=str, default=None, help='Prefix of outputs files.')
    parser.add_argument('--n_repeats', type=int, default=100, help='Number of randomly initiated runs.')
    parser.add_argument('--n_components', type=int, default=None, help='Number of components extracted.')
    parser.add_argument('--exp_seed', type=int, default=None, help='Seed for the experiment, integer.')

    args = parser.parse_args()
    for arg in vars(args):
        print(arg, getattr(args, arg))

    data = pd.read_csv(args.data_dir, sep='\t', header=0, index_col=0)
    s_list, a_list, i_repeats, i_comps = multi_ica(data=data,
                                                   n_components=args.n_components,
                                                   n_repeats=args.n_repeats,
                                                   seed=args.exp_seed)
    ics = np.hstack(tuple(s_list)).T  # ics shape: n_components*n_repeats, n_genes
    mixs = np.hstack(tuple(a_list)).T  # mixs shape: n_components*n_repeats, n_samples
    dis = dis_cal(ics=ics, name=args.exp_prefix, out_dir=args.out_dir)  # dis shape: symmetrical n_components*n_repeats

    ics_pts = ic_cluster(dis=dis, name=args.exp_prefix,  # cluster assignment
                         min_cluster_size=int(args.n_repeats*0.5),
                         out_dir=args.out_dir)

    hf = h5py.File(args.out_dir + '/' + args.exp_prefix + '_ics.h5', 'w')  # save array data as h5 files
    hf.create_dataset('components', data=ics)
    hf.create_dataset('mixings', data=mixs)

    ics = pd.DataFrame(data=ics, columns=data.index.values)  # convert to dataframe for future operations
    mixs = pd.DataFrame(data=mixs, columns=data.columns.values)
    mixs['i_repeats'] = i_repeats
    mixs['i_comps'] = i_comps
    mixs.to_csv(path_or_buf=args.out_dir + '/' + args.exp_prefix + '_all_mixing.tsv',
                sep='\t', index=False, header=True)

    out_tb = pd.DataFrame(data=ics_pts,
                          columns=['tsne_1', 'tsne_2', 'mds_1', 'mds_2', 'cls_lab'])
    out_tb['i_repeats'] = i_repeats
    out_tb['i_comps'] = i_comps
    out_tb.to_csv(path_or_buf=args.out_dir + '/' + args.exp_prefix + '_ic_cluster.tsv',
                  sep='\t', index=False, header=True)

    ics['cluster'] = out_tb['cls_lab']
    mixs['cluster'] = out_tb['cls_lab']

    mean_mix = pd.pivot_table(mixs, values=data.columns.values, index='cluster', aggfunc=np.mean)
    mean_mix.to_csv(path_or_buf=args.out_dir + '/' + args.exp_prefix + '_mean_mixing.tsv',
                  sep='\t', index=True, header=True)

    mean_ic = pd.pivot_table(ics, values=data.index.values, index='cluster', aggfunc=np.mean)
    mean_ic.T.to_csv(path_or_buf=args.out_dir + '/' + args.exp_prefix + '_mean_components.tsv',
                    sep='\t', index=True, header=True)


if __name__ == "__main__":
    main()
