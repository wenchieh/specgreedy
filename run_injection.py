#!/usr/bin/env python3
# -*- coding=utf-8 -*-


# sys
import sys
import time

# third-part libs
import numpy as np

# project
from src.greedy import *
from src.ioutils import *
from src.performance import *
from src.injections import injectCliqueCamo


def FScore(prec, rec):
    return 0 if (prec + rec == 0) else (2 * prec * rec / (prec + rec))


if __name__ == '__main__':
    greedy_func = avgdeg_even
    issquared = True

    infn = '../data/amazon_user_art_rate_time.edgelist'

    print("## Dataset: {}".format(infn[infn.rfind('/')+1:]))
    t0 = time.time()
    M0, (ms, ns) = loadedge2sm(infn, delimiter=' ', idstartzero=True, issquared=issquared)
    M0 = (M0 > 0).astype('int')

    stx, sty = 0, 0 #10000, 100  #10000, 0 #10000, 1000 #
    szx, szy = M0.shape #4000, 4000
    M = M0[stx:stx + szx, sty:sty + szy]
    print("load graph time @ {}s".format(time.time() - t0))
    print("original graph: #node: {}, #edge: {}, shape:{}, density:{}, range: {}".format(
                  (ms, ns), M0.sum(), M0.shape, M0.sum() *1.0/np.prod(M0.shape, dtype=float), (M0.max(), M0.min())))
    print("select subgraph: #nodes: {}, #edges: {}, density:{}, range: {}".format(
                   M.shape, M.sum(),  M.sum() *1.0/np.prod(M.shape), (M.max(), M.min())))

    grd_pred, grd_sc = greedy_func(M)
    print("greedy detect: size:{}, density:{}, nes:{}".format((len(grd_pred[0]), (len(grd_pred[1]))),
                                                              grd_sc, c2score(M, grd_pred[0], grd_pred[1])))
    #print(np.asarray(grd_pred[0]))
    #print(np.asarray(grd_pred[1]))
	
    topk = 3
    m0, n0 = 600, 600 #200, 200 # 200, 200 # 50, 50 #
    print("original total sum: {}, part sum: {}".format(np.sum(M), np.sum(M[:m0, :n0])))
    print("graph density: {}, part density: {}\n\n".format(
                       np.sum(M)*1.0/np.sum(M.shape), np.sum(M[:m0, :n0])*1.0/(m0+n0)))

    # grd_optfs, spec_optfs, spoken_optfs = list(), list(), list()
    densities = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12]  # 0.02,
    grd_res, spoken_res, spgd_res = list(), list(), list()
    for p in densities:
        # print("++++++++++++++++ {} ++++++++++++++++++".format(p))
        for testIdx in [0, 1, 3]: #[1]: #[0, 1, 2, 3, 4]:
            ns, injes, injus, injvs = injectCliqueCamo(M, m0, n0, p, testIdx)
            actual = [injus, injvs]

            nes = 0
            M1 = M.copy()
            M1 = M1.tolil()
            for (s, t) in injes:
                if M1[s, t] == 0:
                    M1[s, t] = 1
                    nes += 1
            # print("inject p: {}, ns: {}, nes: {}, subgraph-density: {}\n".format(p, ns, nes, M1[:m0,:n0].sum() * 1.0 / (m0 + n0)))

            grd_pred, grd_sc = greedy_func(M1)

            grd_rm, grd_cn = len(grd_pred[0]), len(grd_pred[1])
            grd_res.append([getPrecision(grd_pred, actual), getRecall(grd_pred, actual), getFMeasure(grd_pred, actual)])
                                                              (tol_p, tol_r, tol_f)))

            row_cps, col_cps, sigmas = spectral_levels(M1, topk=topk)
            max_rf, max_cf, max_tf = list(), list(), list()
            spgd_maxscore, spgd_ms, argmaxk = 0, None, 0
            spoken_maxscore, spoken_ms = 0, None,
            # spgd_res, spoken_res = list(), list()

            for k in range(topk):
                M1_ps = M1[sorted(list(row_cps[k])), :][:, sorted(list(col_cps[k]))]
                if M1_ps.sum() <= 0:
                    # print("****** No edges selected ******")
                    break

                rm, cn = len(row_cps[k]), len(col_cps[k])
                pred_spec = [row_cps[k], col_cps[k]]
                spec_dens = M1_ps.sum() * 1.0 / (rm + cn)

                if spec_dens > spoken_maxscore:
                    spoken_maxscore = spec_dens
                    spoken_ms = [getPrecision(pred_spec, actual), getRecall(pred_spec, actual), getFMeasure(pred_spec, actual)]

                if rm < 2 or cn < 2:
                    sc_inj = M1_ps.sum() * 1.0 / (rm + cn)
                    # print("\tWarning: top:{}:: row_cans: {}, col_cans: {}".format(k, rm, cn))
                    row_ndcf, col_ndcf = row_cps[k], col_cps[k]
                else:
                    nd_inj, sc_inj = greedy_func(M1_ps)
                    row_ndcf, col_ndcf = np.asarray(sorted(row_cps[k]))[nd_inj[0]], np.asarray(sorted(col_cps[k]))[nd_inj[1]]

                if sc_inj > spgd_maxscore:
                    spgd_maxscore, argmaxk = sc_inj, k
                    pred = [row_ndcf, col_ndcf]
                    spgd_ms = [getPrecision(pred, actual), getRecall(pred, actual), getFMeasure(pred, actual)]

            spgd_res.append(spgd_ms)
            spoken_res.append(spoken_ms)

    print(grd_res)
    print(spoken_res)
    print(spgd_res)