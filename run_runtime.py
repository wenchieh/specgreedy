#!/usr/bin/env python3
# -*- coding=utf-8 -*-


# sys
import os
import sys
import time
import argparse
from random import sample

# third-part libs
import numpy as np

# project
from src.ioutils import *
from src.greedy import *


def array_squezee(dat):
    nids = sorted(set(dat.flatten()))
    # print(len(nids))
    if dat.max() > len(nids):
        idmp = dict(zip(nids, range(len(nids))))
        map_dat = list()
        for k in range(len(dat)):
            map_dat.append([idmp[dat[k, 0]], idmp[dat[k, 1]]])
        return np.asarray(map_dat)
    else:
        return dat


if __name__ == '__main__':
    startszero = True
    issquare = True
    topk = 3
    greedy_fun = fast_greedy_decreasing_monosym

    infn = '../data/soc-Twitter_ASU.edgelist'
    t0 = time.time()
    dat = np.loadtxt(infn, int, '%', ' ', usecols=[0, 1])
    if not startszero: dat -=1
    print("load data @ {} s".format(time.time() - t0))

    ns, es = dat.max() + 1, len(dat)
    print("INFO: #node: {}, #edge:{}\n".format(ns, es))

    rho_es = [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]
    
    for rho in rho_es:
        k_nes = int(es * rho)
        
        if rho < 1:
            nidx = sample(range(es), k_nes)
            subdat = array_squezee(dat[nidx, :2])
            sm = list2sm_mono(subdat[:, 0], subdat[:, 1], issquare)
        else:
        	sm = list2sm_mono(dat[:, 0], dat[:, 1], issquare)
       
        print("\nrho:{}, shape:{}, nes:{}, density:{}".format(rho, sm.shape, sm.sum(), sm.sum()*1.0/np.prod(sm.shape, dtype=float)))
        orgnds, cans = None, None
        opt_density = 0.0
        opt_k = -1
        t_k = time.time()
        row_cans, col_cans, lambs = spectral_levels(sm, topk)
        print("lambdas: {}".format(lambs))
        for kth in range(topk):
            #print("top {} svd approximate result:".format(kth + 1))
            cans = set(row_cans[kth]).union(col_cans[kth])
            if len(cans) < 2:
                continue
            sm_part = sm[sorted(list(cans)), :][:, sorted(list(cans))]
            nds_part_den, avgsc_part = greedy_fun(sm_part)

            if avgsc_part > opt_density:
                opt_k = kth + 1
                opt_density = avgsc_part

            if kth + 1 < topk and avgsc_part > lambs[kth + 1]:
                print("best density: {}, k={}".format(avgsc_part / 2.0, kth))

        print("{}: {}, {} {}, {}\n".format(rho, opt_k, opt_density, k_nes, time.time()- t_k))

    print("done!")

