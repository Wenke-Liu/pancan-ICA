import numpy as np
import pandas as pd
import h5py
import ica
import argparse
import hdbscan
from sklearn.metrics import silhouette_samples


def main():

    parser = argparse.ArgumentParser(description='ICA')
    parser.add_argument('--mode', type=str, default='all', help='analysis mode')
    parser.add_argument('--data_dir', type=str, default='.', help='Path to data table.')
    parser.add_argument('--saved_dir', type=str, default=None, help='Path to saved intermediate output.')
    parser.add_argument('--clinical_dir', type=str, default=None, help='Path to clinical table')
    parser.add_argument('--out_dir', type=str, default='.', help='Path to outputs.')
    parser.add_argument('--exp_prefix', type=str, default=None, help='Prefix of outputs files.')
    parser.add_argument('--n_repeats', type=int, default=100, help='Number of randomly initiated runs.')
    parser.add_argument('--n_components', type=int, default=None, help='Number of components extracted.')
    parser.add_argument('--exp_seed', type=int, default=None, help='Seed for the experiment, integer.')

    args = parser.parse_args()
    for arg in vars(args):
        print(arg, getattr(args, arg))

    if not args.saved_dir:  # ICA from raw data
        data = pd.read_csv(args.data_dir, sep='\t', header=0, index_col=0)
        s_list, a_list, i_repeats, i_comps = ica.multi_ica(data=data,
                                                           n_components=args.n_components,
                                                           n_repeats=args.n_repeats,
                                                           seed=args.exp_seed)

        ics, mixs = ica.align_tail(s_list, a_list)

        hf = h5py.File(args.out_dir + '/' + args.exp_prefix + '_ics.h5', 'w')  # save array data as h5 files
        hf.create_dataset('components', data=ics)
        hf.create_dataset('mixings', data=mixs)
        hf.create_dataset('i_repeats', data=np.asarray(i_repeats))
        hf.create_dataset('i_comps', data=np.asarray(i_comps))
        hf.close()

        dis = ica.dis_cal(ics=ics, name=args.exp_prefix,
                          out_dir=args.out_dir)  # dis shape: symmetrical n_components*n_repeats

    else:
        hf = h5py.File(args.saved_dir + '/' + args.exp_prefix + '_ics.h5', 'r')
        ics = hf.get('components')
        ics = np.array(ics)
        #mixs = hf.get('mixings')
        #mixs = np.array(mixs)
        i_repeats = hf.get('i_repeats')
        i_repeats = np.array(i_repeats)
        i_comps = hf.get('i_comps')
        i_comps = np.array(i_comps)
        hf.close()

        hf2 = h5py.File(args.saved_dir + '/' + args.exp_prefix + '_dis.h5', 'r')
        dis = hf2.get('dis')
        dis = np.array(dis)
        hf2.close()
        mixs = pd.read_csv(args.saved_dir + '/' + args.exp_prefix + '_all_mixing.tsv', sep='\t', header=0, index_col=None)

    if args.mode == 'association':  # only clinical association association
        pass

    else:  # not association only mode

        """grid search for clustering hyper-parameters
        """
        sizes = (np.arange(0.1, 1.0, 0.1) * args.n_repeats).astype(int)
        samples = (np.arange(0.1, 1.0, 0.1) * args.n_repeats).astype(int)

        cls_grid = []
        sizes_grid = []
        samples_grid = []
        sil_grid = []

        n_cls = []
        ave_sil = []
        sd_sil = []
        ave_sil_sample = []

        for min_cluster_size in sizes:
            for min_samples in samples:
                clusterer = hdbscan.HDBSCAN(min_cluster_size=int(min_cluster_size),  # clustering with hdbscan
                                            min_samples=int(min_samples),
                                            metric='precomputed')
                clusterer.fit(dis)
                cls = clusterer.labels_
                s = silhouette_samples(X=dis, labels=cls, metric='precomputed')

                cls_grid.append(cls)  # save grid search results
                sizes_grid.append(min_cluster_size)
                samples_grid.append(min_samples)
                sil_grid.append(s)
                print('Parameter search: min_cluster, {}, min_samples, {}'.format(min_cluster_size, min_samples))

                grid_df = pd.DataFrame()  # summarize grid search results
                grid_df['cls'] = cls
                grid_df['sil'] = s

                grid_df = grid_df[grid_df['cls'] != -1]  # filter out the noise points
                ave_sil_sample.append(np.mean(grid_df['sil']))

                sils = grid_df.groupby('cls').aggregate(np.mean)

                n_cls.append(int(np.max(sils['sil']) + 1))
                ave_sil.append(np.mean(sils['sil']))
                sd_sil.append(np.std(sils['sil']))

        cls_grid = pd.DataFrame(data=np.array(cls_grid).T)
        sil_grid = pd.DataFrame(data=np.array(sil_grid).T)

        param = pd.DataFrame(data={'n_components': args.n_components,
                                   'n_repeats': args.n_repeats,
                                   'min_cluster_size': sizes_grid,
                                   'min_samples': samples_grid,
                                   'ave_sil_sample': ave_sil_sample,
                                   'ave_sil': ave_sil,
                                   'sd_sil': sd_sil})

        cls_grid.to_csv(path_or_buf=args.out_dir + '/' + args.exp_prefix + '_cls_grid.tsv',
                        sep='\t', index=True, header=True)

        sil_grid.to_csv(path_or_buf=args.out_dir + '/' + args.exp_prefix + '_sil_grid.tsv',
                        sep='\t', index=True, header=True)

        param.to_csv(path_or_buf=args.out_dir + '/' + args.exp_prefix + '_param_grid.tsv',
                     sep='\t', index=True, header=True)

        ics = pd.DataFrame(data=ics, columns=data.index.values)  # convert to dataframe for future operations
        mixs = pd.DataFrame(data=mixs, columns=data.columns.values)
        mixs['i_repeats'] = i_repeats
        mixs['i_comps'] = i_comps

        best_param = param[param['ave_sil_sample'] == param['ave_sil_sample'].max()]
        print('Best parameters: min_cluster, {}, min_samples, {}'.format(best_param['min_cluster_size'].iloc[0],
                                                                         best_param['min_samples'].iloc[0]))

        ics_pts = ica.ic_cluster(dis=dis, name=args.exp_prefix,  # cluster assignment
                                 min_cluster_size=int(best_param['min_cluster_size'].iloc[0]),
                                 min_samples=int(best_param['min_samples'].iloc[0]),
                                 out_dir=args.out_dir)

        out_tb = pd.DataFrame(data=ics_pts,
                              columns=['tsne_1', 'tsne_2', 'mds_1', 'mds_2', 'cls_lab'])
        out_tb['i_repeats'] = i_repeats
        out_tb['i_comps'] = i_comps

        silhouettes = silhouette_samples(X=dis, labels=out_tb['cls_lab'].values.astype(int), metric='precomputed')
        out_tb['silhouette'] = silhouettes

        out_tb.to_csv(path_or_buf=args.out_dir + '/' + args.exp_prefix + '_ic_cluster.tsv',
                      sep='\t', index=False, header=True)

        ics['cluster'] = out_tb['cls_lab']
        mixs['cluster'] = out_tb['cls_lab']

        mixs.to_csv(path_or_buf=args.out_dir + '/' + args.exp_prefix + '_all_mixing.tsv',
                    sep='\t', index=False, header=True)

        mean_mix = pd.pivot_table(mixs, values=data.columns.values, index='cluster', aggfunc=np.mean)
        mean_mix.to_csv(path_or_buf=args.out_dir + '/' + args.exp_prefix + '_mean_mixing.tsv',
                        sep='\t', index=True, header=True)

        mean_ic = pd.pivot_table(ics, values=data.index.values, index='cluster', aggfunc=np.mean)
        mean_ic.T.to_csv(path_or_buf=args.out_dir + '/' + args.exp_prefix + '_mean_components.tsv',
                         sep='\t', index=True, header=True)

    if args.clinical_dir:  # clinical association
        clinical_dat = pd.read_csv(args.clinical_dir, sep='\t', header=0, index_col=0)
        raw_p, raw_c = ica.find_cor(mix=mixs, clinical=clinical_dat)
        raw_p.to_csv(path_or_buf=args.out_dir + '/' + args.exp_prefix + '_raw_p_values.tsv',
                     sep='\t', index=True, header=True)
        raw_c.to_csv(path_or_buf=args.out_dir + '/' + args.exp_prefix + '_raw_slopes.tsv',
                     sep='\t', index=True, header=True)


if __name__ == "__main__":
    main()
