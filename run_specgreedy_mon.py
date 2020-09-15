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
    parser = argparse.ArgumentParser(description="[SPECGREEDY] Dense Subgraph Detection",
                                     usage="python run_specgreedy_mono.py ins outs delimiter weighted topk")
    parser.add_argument("ins", help="input tensor path", type=str)
    parser.add_argument("outs", help="result output path", type=str)
    parser.add_argument("delimiter", help="delimiter of input data", type=str, default=' ')
    parser.add_argument("weighted", help="is weighted graph", const=True, nargs='?', type=str2bool)
    parser.add_argument("topk", help="select largest k svd result", type=int, default=int)
    #parser.add_argument("scale", help="spectral scale", type=float, default=1.0)
    args = parser.parse_args()

    infn, outfn = args.ins, args.outs
    delm = args.delimiter
    w_g = args.weighted
    topk = args.topk
    scale = 1.0  # args.scale

    print("## Dataset: {}".format(infn[infn.rfind('/')+1:]))
    greedy_fun = fast_greedy_decreasing_monosym

    t0 = time.time()
    sm, (ms, ns) = loadedge2sm(infn, weighted=False, delimiter=delm, idstartzero=False, issquared=True)
    n = max(ms, ns)
    sm -= sps.diags(sm.diagonal())
    es = sm.sum()
    if (abs(sm-sm.T)>1e-10).nnz > 0:
        sm += sm.T
    if not w_g:
        print("max edge weight: {}".format(sm.max()))
        sm = sm > 0
        sm = sm.astype('int')

    print("load graph @ {}s".format(time.time() - t0))
    print("graph: #node: {}, #edge: {}, # es: {}".format((ms, ns), es, sm.sum()))
    print("matrix max: {}, min: {}, shape: {}\n".format(sm.max(), sm.min(), (n ,n)))

    orgnds, cans = None, None
    opt_density = 0.0
    opt_k = -1

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
        print("lambdas: {} \n".format(S))
        kth  = k
        while kth < start + decom_n * step - 1 and kth < topk:
            if abs(max(U[:, kth])) < abs(min(U[:, kth])): U[:, kth] *= -1
            if abs(max(V[:, kth])) < abs(min(V[:, kth])): V[:, kth] *= -1
            row_cans = list(np.where(U[:, kth] >= 1.0 / np.sqrt(ms))[0])
            # col_cans = list(np.where(V[:, kth] >= 1.0 / np.sqrt(ns))[0])
            col_cans = row_cans
            sm_part = sm[row_cans, :][:, col_cans]
            # print("{}, size: {}".format(kth, sm_part.shape))
            nds_res, avgsc_part = greedy_fun(sm_part)
            print("k_cur:{}, size: {}, density: {}.  @ {}s\n".format(kth, len(nds_res),  avgsc_part / 2.0, time.time() - t1))
            kth += 1
            k += 1
            if avgsc_part > opt_density:
                opt_k, opt_density = kth, avgsc_part
                sm_pms = max(len(row_cans), len(col_cans))
                cans = row_cans
                fin_pms = len(nds_res)
                print("++== svd init shape (candidates size): {}".format((sm_pms, sm_pms)))
                print("++== size: {}, score: {}\n".format((fin_pms, fin_pms), avgsc_part / 2.0))
                nd_idx = dict(zip(range(sm_pms), sorted(cans)))
                orgnds = [nd_idx[id] for id in nds_res]

            if 2.0*opt_density >= S[kth]: # kth < topk and
                print("k_cur = {},  optimal density: {}, compare: {}".format(kth, opt_density / 2.0, S[kth]))
                isbreak = True
                break
        if isbreak:
            break
        decom_n += 1

    print("\noptimals: k:{}, size:{}, density:{}".format(opt_k, fin_pms, opt_density / 2.0))
    print("total time @ {}s".format(time.time() - t1))
    save_dictlist({'x': orgnds}, outfn)

    print("done")
