import numpy as np
import pandas as pd
import h5py
import ica
import argparse


def main():

    parser = argparse.ArgumentParser(description='ICA')
    parser.add_argument('--data_dir', type=str, default='.', help='Path to data table.')
    parser.add_argument('--clinical_dir', type=str, default=None, help='Path to clinical table')
    parser.add_argument('--out_dir', type=str, default='.', help='Path to outputs.')
    parser.add_argument('--exp_prefix', type=str, default=None, help='Prefix of outputs files.')
    parser.add_argument('--n_repeats', type=int, default=100, help='Number of randomly initiated runs.')
    parser.add_argument('--n_components', type=int, default=None, help='Number of components extracted.')
    parser.add_argument('--exp_seed', type=int, default=None, help='Seed for the experiment, integer.')

    args = parser.parse_args()
    for arg in vars(args):
        print(arg, getattr(args, arg))

    data = pd.read_csv(args.data_dir, sep='\t', header=0, index_col=0)
    s_list, a_list, i_repeats, i_comps = ica.multi_ica(data=data,
                                                       n_components=args.n_components,
                                                       n_repeats=args.n_repeats,
                                                       seed=args.exp_seed)

    ics, mixs = ica.align_tail(s_list, a_list)

    dis = ica.dis_cal(ics=ics, name=args.exp_prefix,
                      out_dir=args.out_dir)  # dis shape: symmetrical n_components*n_repeats

    ics_pts = ica.ic_cluster(dis=dis, name=args.exp_prefix,  # cluster assignment
                             min_cluster_size=int(args.n_repeats*0.5),
                             out_dir=args.out_dir)

    hf = h5py.File(args.out_dir + '/' + args.exp_prefix + '_ics.h5', 'w')  # save array data as h5 files
    hf.create_dataset('components', data=ics)
    hf.create_dataset('mixings', data=mixs)

    ics = pd.DataFrame(data=ics, columns=data.index.values)  # convert to dataframe for future operations
    mixs = pd.DataFrame(data=mixs, columns=data.columns.values)
    mixs['i_repeats'] = i_repeats
    mixs['i_comps'] = i_comps

    out_tb = pd.DataFrame(data=ics_pts,
                          columns=['tsne_1', 'tsne_2', 'mds_1', 'mds_2', 'cls_lab'])
    out_tb['i_repeats'] = i_repeats
    out_tb['i_comps'] = i_comps
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

    if args.clinical_dir:
        clinical_dat = pd.read_csv(args.clinical_dir, sep='\t', header=0, index_col=0)
        raw_p = ica.find_cor(mix=mixs, clinical=clinical_dat)
        raw_p.to_csv(path_or_buf=args.out_dir + '/' + args.exp_prefix + '_raw_p_values.tsv',
                     sep='\t', index=True, header=True)


if __name__ == "__main__":
    main()
