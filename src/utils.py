from __future__ import print_function, division
import numpy as np
import scipy.linalg as la

def solve(a, b):
    return la.solve(a, b, assume_a='pos')


def whiten(cov, dm, dx, dy, ret_channel_params=False):

    sig_mxy = cov.copy()
    sig_m = cov[:dm, :dm]
    sig_mxy[:, :dm] = solve(la.sqrtm(sig_m).real, sig_mxy[:, :dm].T).T
    sig_mxy[:dm, :] = solve(la.sqrtm(sig_m).real, sig_mxy[:dm, :])
    sig_m = sig_mxy[:dm, :dm]

    sig_x = sig_mxy[dm:dm+dx, dm:dm+dx]
    sig_y = sig_mxy[dm+dx:, dm+dx:]
    sig_x_m = sig_mxy[dm:dm+dx, :dm] 
    sig_y_m = sig_mxy[dm+dx:, :dm]

    sig_x__m = sig_x - sig_x_m @ solve(sig_m, sig_x_m.T)
    sig_y__m = sig_y - sig_y_m @ solve(sig_m, sig_y_m.T)

    sig_mxy[:, dm:dm+dx] = solve(la.sqrtm(sig_x__m).real, sig_mxy[:, dm:dm+dx].T).T
    sig_mxy[dm:dm+dx, :] = solve(la.sqrtm(sig_x__m).real, sig_mxy[dm:dm+dx, :])

    sig_mxy[:, dm+dx:] = solve(la.sqrtm(sig_y__m).real, sig_mxy[:, dm+dx:].T).T
    sig_mxy[dm+dx:, :] = solve(la.sqrtm(sig_y__m).real, sig_mxy[dm+dx:, :])

    sig_xy = sig_mxy[dm:, dm:]
    sig_xy_m = sig_mxy[dm:, :dm]
    sig_xy__m = sig_xy - sig_xy_m @ solve(sig_m, sig_xy_m.T) 

    if ret_channel_params:
        return sig_mxy, sig_x_m, sig_y_m, sig_xy_m, sig_xy__m
    return sig_mxy


def recondition(x, max_cond=1e10, return_tf=False):

    cov = np.cov(x)

    w, v = la.eigh(cov)
    w = w.real
    v = v.real 

    bad_indices = np.where(w < w[-1] / max_cond)[0]
    start_index = bad_indices.max() + 1

    wsub = w[start_index:]
    vsub = v[:, start_index:]

    x_new = vsub.T @ x

    if return_tf:
        return x_new, vsub.T
    return x_new


def lin_tf_params_from_cov(cov, dm, dx, dy):

    covm = cov[:dm, :dm]
    covx = cov[dm:dm+dx, dm:dm+dx]
    covy = cov[dm+dx:, dm+dx:]
    covxy = cov[dm:, dm:]
    covx_m = cov[:dm, dm:dm+dx].T
    covy_m = cov[:dm, dm+dx:].T
    covxy_m = cov[:dm, dm:].T
    hx = covx_m.dot(la.inv(covm))
    hy = covy_m.dot(la.inv(covm))
    hxy = covxy_m.dot(la.inv(covm))
    sigm = covm
    sigx = covx - covx_m.dot(la.inv(covm)).dot(covx_m.T)
    sigy = covy - covy_m.dot(la.inv(covm)).dot(covy_m.T)
    sigxy = covxy - covxy_m.dot(la.inv(covm)).dot(covxy_m.T)

    hx = la.sqrtm(la.inv(sigx)).dot(hx)
    sigx = np.eye(dx)
    hy = la.sqrtm(la.inv(sigy)).dot(hy)
    sigy = np.eye(dy)

    return hx, hy, hxy, sigx, sigy, sigxy, covxy, sigm


def lin_tf_params_ip_dfncy(cov, dm, dx, dy):
    covm = cov[:dm, :dm]
    covx = cov[dm:dm+dx, dm:dm+dx]
    covy = cov[dm+dx:, dm+dx:]
    covxy = cov[dm:, dm:]
    covx_m = cov[:dm, dm:dm+dx].T
    covy_m = cov[:dm, dm+dx:].T
    covxy_m = cov[:dm, dm:].T

    sigm = covm
    gx = covx_m @ la.sqrtm(la.inv(covy))
    gy = covy_m @ la.inv(covy)
    sigx = covx

    return gx, gy, sigm, sigx, sigy, sigx_m, sigy_m


def lin_tf_params_bert(cov, dm, dx, dy):

    covm = cov[:dm, :dm]
    covx = cov[dm:dm+dx, dm:dm+dx]
    covy = cov[dm+dx:, dm+dx:]
    covxy = cov[dm:, dm:]
    covx_m = cov[:dm, dm:dm+dx].T
    covy_m = cov[:dm, dm+dx:].T
    covxy_m = cov[:dm, dm:].T
    covx_y = cov[dm:dm+dx, dm+dx:]

    covm_sqrt = la.sqrtm(covm)
    covx_sqrt = la.sqrtm(covx)
    covy_sqrt = la.sqrtm(covy)

    hx = la.solve(covx_sqrt, la.solve(covm_sqrt, covx_m.T).T)
    hy = la.solve(covy_sqrt, la.solve(covm_sqrt, covy_m.T).T)
    sigx_y = la.solve(covx_sqrt, la.solve(covy_sqrt, covx_y.T).T)

    sigm = np.eye(dm)
    sigx = np.eye(dx)
    sigy = np.eye(dy)

    hxy = np.vstack((hx, hy))
    sigxy = np.block([[sigx, sigx_y], [sigx_y.T, sigy]])

    return hx, hy, sigx_y, sigm, sigx, sigy, hxy, sigxy


def remove_lin_dep_comps(cov, dm, dx, dy):

    covm = cov[:dm, :dm]
    wm, vm = la.eigh(covm)
    wm = wm[::-1]
    vm = vm[:, ::-1]

    wm_thresholded_mask = (wm > 1e-10)
    wm_cumsum = np.cumsum(wm) / wm.sum()
    wm_var_capture_index = np.where(wm_cumsum < 1 - 1e-3)[0].max() + 1
    wm_var_mask = (np.arange(wm.size) <= wm_var_capture_index)
    mask = wm_thresholded_mask & wm_var_mask

    wm_new = wm[mask]
    vm_new = vm[:, mask]
    dm_new = wm_new.size

    transform = np.tile([[vm_new.T, np.zeros((dm, dx + dy))],
                         [np.zeros((dx + dy, dm)), np.eye(dx + dy)]])
    cov_new = transform @ cov @ transform.T

    return cov_new, dm_new


def make_cov_m_equals_xy(covxy, dx, dy, epsilon=0.0):

    covx = covxy[:dx, :dx]
    covy = covxy[dx:, dx:]
    covx_y = covxy[:dx, dx:]

    covm_x = np.vstack((covx, covx_y.T))
    covm_y = np.vstack((covx_y, covy))
    cov = np.block([[covxy, covm_x, covm_y],
                    [covm_x.T, covx + epsilon * np.eye(dx), covx_y],
                    [covm_y.T, covx_y.T, covy + epsilon * np.eye(dy)]])

    return cov
