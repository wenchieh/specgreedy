#!/usr/bin/env python3
# -*- coding=utf-8 -*-


# sys
import time
import argparse

# third-part libs
import numpy as np
import scipy.io as sio

# project
from src.ioutils import *
from src.greedy import *

greedy = False #True


def contrast_dense(matP, matQ, topk, dscore, lamb=1.0, idstartszero=True, outfn=None):
    assert matP.shape == matQ.shape
    (m, n) = matP.shape
    # matQ -= sps.diags(matQ.diagonal())
    sm_diff = ((matP - lamb * matQ) > 0).astype(float)

    opt_k = -1
    opt_sz, opt_density = None, 0.0
    cans, org_nodes = None, None
    greedy_fun = fast_greedy_sym2gs_diff

    t1 = time.time()
    row_cans, col_cans, lambs = spectral_levels(sm_diff, topk)
    print("init score: {}".format(matP.sum() * 1.0 / matQ.sum()))
    print("spectral de-compos @ {} s".format(time.time() - t1))
    print("lambdas: {}\n".format(lambs))
    for kth in range(topk):
        sz = (len(row_cans[kth]), len(col_cans[kth]))
        print("top {} spectral truncated result: size:{} ".format(kth + 1, sz))

        P_kth = matP[sorted(list(row_cans[kth])), :][:, sorted(list(col_cans[kth]))]
        Q_kth = matQ[sorted(list(row_cans[kth])), :][:, sorted(list(col_cans[kth]))]

        init_psc, init_qsc = P_kth.sum(), Q_kth.sum()
        init_sc = init_psc * 1.0 / init_qsc
        sz_inits = len(row_cans[kth])
        print("initial: size: {}, density:{} - {}".format(sz_inits, (init_psc, init_qsc - sz_inits), init_sc))

        nodes_kth, avgsc_kth = greedy_fun(P_kth, Q_kth, dscore=dscore)
        psc, qsc = c2score(P_kth, nodes_kth[0], nodes_kth[1]), c2score(Q_kth, nodes_kth[0], nodes_kth[1])
        sc = psc * 1.0 / qsc
        sz_kths = len(nodes_kth[0])
        print("score: {}".format(avgsc_kth))
        print("result: size: {}, density:{} - {}".format(sz_kths, (psc, qsc - sz_kths), sc))
        print("nodes: {}".format(nodes_kth[0]))

        if avgsc_kth > opt_density:
            opt_k, opt_sz  = kth + 1, sz_kths
            opt_density = avgsc_kth
            max_sz = max(len(row_cans[kth]), len(col_cans[kth]))
            cans = np.asarray(row_cans[kth])
            if not idstartszero: cans += 1
            nd_idx = dict(zip(range(max_sz), sorted(cans)))
            org_nodes = [nd_idx[id] for id in nodes_kth[0]]
            print("\t ++++ optimal result update!")
        print("\n")
        if kth + 1 < topk and opt_density > lambs[kth + 1]:
            print("The best density: {}, k={}".format(opt_density / 2.0, opt_k))
            break

    print("\ntotal time @ {}s".format(time.time() - t1))
    print("optimal: k:{}, size:{}, density:{}".format(opt_k, opt_sz, opt_density))
    if outfn:
        save_dictlist({'x': org_nodes}, outfn + '-%d' % opt_k)


def greedy_contrast(matP, matQ, dscore, outfn=None):
    greedy_fun = fast_greedy_sym2gs_diff
    t1 = time.time()
    nodes, score = greedy_fun(matP, matQ, dscore=dscore)
    print("greedy total time @ {}s".format(time.time() - t1))
    print("score: {}".format(score)) #  / 2.0
    sz = (len(nodes[0]), len(nodes[1]))
    p_score, q_score = c2score(matP, nodes[0], nodes[1]), c2score(matQ, nodes[0], nodes[1])
    sc = p_score * 1.0 / q_score
    print("size: {}, density: {} - {}".format(sz, (p_score, q_score - sz[0]), sc))
    if outfn:
        save_dictlist({'x': nodes}, outfn)


if __name__ == '__main__':
    outfn = None
    infn = '../data/ca-DBLP_Aminer.edgelist'   ###  author author year  

    t0 = time.time()
    idstartswithzero = False
    dat = np.loadtxt(infn, int, '%', ' ')
    max_n = np.max(dat[:, :2].flatten())
    print("load data @ {} s".format(time.time() - t0))

    if not idstartswithzero:
        dat[:, :2] -= 1

    dscore = ''
    topk = 10
    prev, nex = None, None
    year = sorted(list(set(dat[:, -1])))

    for k in range(1, len(year)):
        print("YEAR: P:{} vs. Q:{}".format(year[k - 1], year[k]))
        # print("YEAR: P:{} vs. Q:{}".format(year[k], year[k - 1]))
        if nex is None:
            prev = dat[dat[:, -1] == year[k - 1], :]
        else:
            prev = nex
        nex = dat[dat[:, -1] == year[k], :]
        P = list2sm_mono(prev[:, 0], prev[:, 1], n = max_n)  # previous year graph
        Q = list2sm_mono(nex[:, 0], nex[:, 1], n = max_n)  # next year graph
        print("P:{}, Q:{}".format(P.shape, Q.shape))
        # P += sps.diags([1.0] * max_n)
        Q += sps.diags([1.0] * max_n)
        contrast_dense(P, Q, topk, dscore, idstartszero=idstartswithzero, outfn=outfn)
        # contrast_dense(Q, P, topk, dscore, idstartszero=idstartswithzero, outfn=outfn)
        # print("shape: P-{}, Q-{}".format(P.shape, Q.shape))
        print("\n\n")
