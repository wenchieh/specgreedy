#!/usr/bin/env python3
# -*- coding=utf-8 -*-


# sys
import os
import sys
import time
import argparse

# third-part libs
import numpy as np

# project
from src.ioutils import *
from src.greedy import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="[SPECGREEDY-Bip] Dense Subgraph Detection (Bipartite)",
                                     usage="python run_specgreedy_bip.py --ins --outs --delimiter --weighted --col_wt --topk --alpha")
    parser.add_argument("ins", help="input data path", type=str)
    parser.add_argument("outs", help="result output path", type=str)
    parser.add_argument("delimiter", help="delimiter of input data", type=str, default=' ')
    parser.add_argument("weighted", help="is weighted graph", const=True, nargs='?', type=str2bool)
    parser.add_argument("col_wt", help="column weight type", choices=["even", "sqrt", "log"], default="even", type=str)
    parser.add_argument("topk", help="select largest k svd result", default=1, type=int)
    #parser.add_argument("scale", help="spectral scale", default=1.0, type=float)
    parser.add_argument("alpha", help="smoother for column weight", default=5.0, type=float)
    args = parser.parse_args()
    print(args)

    infn, outfn = args.ins, args.outs
    delm = args.delimiter
    w_g = args.weighted
    topk = args.topk
    #scale = args.scale
    alpha = args.alpha
    print("\n## Dataset: {}".format(infn[infn.rfind('/')+1:]))

    #alpha = 1.0
    greedy_func = None
    if args.col_wt == 'even':
        greedy_func = avgdeg_even
    elif args.col_wt == 'sqrt':
        greedy_func = avgdeg_sqrt
    else:
        greedy_func = avgdeg_log

    t0 = time.time()
    sm, (ms, ns) = loadedge2sm(infn, delimiter=delm, idstartzero=False, weighted=False, issquared=False)
    if not w_g:
        print("max edge weight: {}".format(sm.max()))
        sm = sm > 0
        sm = sm.astype('int')
    es = sm.sum()
    print("load graph time @ {}s".format(time.time() - t0))
    print("graph: #node: {},  #edge: {}".format((ms, ns), es))
    print("matrix max: {}, min: {}, shape: {}\n".format(sm.max(), sm.min(), sm.shape))
    
    opt_k = -1
    opt_density = 0.0
    orgnds, cans = None, None
    fin_pms, fin_pns = 0, 0

    k = 0
    decom_n = 0

    start = 3
    step = 3
    isbreak = False
    t1 = time.time()
    while k < topk:
        print("\ncurrent ks: {}".format(start + decom_n * step))
        U, S, V = linalg.svds(sm.asfptype(), k=start + decom_n * step, which='LM', tol=1e-4)
        U, S, V = U[:, ::-1], S[::-1], V.T[:, ::-1]
        print("lambdas: {}".format(S))
        kth  = k
        while kth < start + decom_n * step - 1 and kth < topk:
            if abs(max(U[:, kth])) < abs(min(U[:, kth])):
                U[:, kth] *= -1
            if abs(max(V[:, kth])) < abs(min(V[:, kth])):
                V[:, kth] *= -1
            row_cans = list(np.where(U[:, kth] >= 1.0 / np.sqrt(ms))[0])
            col_cans = list(np.where(V[:, kth] >= 1.0 / np.sqrt(ns))[0])
            if len(row_cans) <= 1 or len(col_cans) <= 1:
                print("SKIP_ERROR: candidates size: {}".format((len(row_cans), len(col_cans))))
                kth += 1
                k += 1
                continue
            sm_part = sm[row_cans, :][:, col_cans]
            nds_gs, avgsc_gs = greedy_func(sm_part, alpha)
            print("k_cur: {} size: {}, density: {}  @ {}s".format(kth, (len(nds_gs[0]), len(nds_gs[1])), 
																  avgsc_gs, time.time() - t1))
            kth += 1
            k += 1
            if avgsc_gs > opt_density:
                opt_k, opt_density = kth + 1, avgsc_gs
                (sm_pms, sm_pns) = sm_part.shape
                fin_pms, fin_pns = len(nds_gs[0]), len(nds_gs[1])
                print("+++=== svd tops shape (candidates size): {}".format((sm_pms, sm_pns)))
                print("+++=== size: {}, score: {}\n".format((fin_pms, fin_pns), avgsc_gs))

                row_idx = dict(zip(range(sm_pms), sorted(row_cans)))
                col_idx = dict(zip(range(sm_pns), sorted(col_cans)))
                org_rownds = [row_idx[id] for id in nds_gs[0]]
                org_calnds = [col_idx[id] for id in nds_gs[1]]
                cans = [row_cans, col_cans]
                orgnds = [org_rownds, org_calnds]
				
            if 2.0 * opt_density >= S[kth]: # kth < topk and
                print("k_cur = {},  optimal density: {}, compare: {}".format(kth, opt_density, S[kth]))
                isbreak = True
                break
        if isbreak:
            break
        decom_n += 1
    
    print("\noptimal k: {}, density: {}".format(opt_k, opt_density))    
    print("total time @ {}s".format(time.time() - t1))
    save_dictlist({'x': orgnds[0], 'y': orgnds[1]}, outfn)

    print("done")
