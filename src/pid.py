from __future__ import print_function, division
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import numpy.linalg as npla
from .utils import whiten

def objective(sig, hx, hy, dm, dx, dy, reg):
    S = (1 + reg) * np.eye(dx) - sig @ sig.T
    B = hx - sig @ hy
    obj = 0.5 / np.log(2) * npla.slogdet(
        np.eye(dm) + hy.T @ hy + B.T @ la.solve(S, B)
    )[1]
    return obj

def gradient(sig, hx, hy, dm, dx, dy, reg):
    S = (1 + reg) * np.eye(dx) - sig @ sig.T
    B = hx - sig @ hy
    S_inv_B = la.solve(S, B)
    g_sig = S_inv_B @ la.solve(np.eye(dm) + hy.T @ hy + B.T @ S_inv_B,
                               S_inv_B.T @ sig - hy.T)
    return g_sig


def project(sig_temp):
    dx, dy = sig_temp.shape
    proj_once = False
    covxy__m = np.block([[np.eye(dx), sig_temp],
                         [sig_temp.T, np.eye(dy)]])
    lamda, V = la.eigh(covxy__m)
    lamda = lamda.real
    if (lamda > 0).all():
        return sig_temp, False

    V = V.real
    lamda[lamda < 0] = 0
    covxy__m = V @ np.diag(lamda) @ V.T

    covx__m = covxy__m[:dx, :dx]
    covy__m = covxy__m[dx:, dx:]
    reg = 1e-12

    while reg < 1.1:
        sig_temp_proj = la.solve(
            reg * np.eye(dx) + la.sqrtm(covx__m).real,
            la.solve(
                reg * np.eye(dy) + la.sqrtm(covy__m).real, covxy__m[:dx, dx:].T
            ).T
        )

        new_covxy__m = np.block([[np.eye(dx), sig_temp_proj],
                                 [sig_temp_proj.T, np.eye(dy)]])
        new_lamda = la.eigvalsh(new_covxy__m)

        if (new_lamda > 0).all():
            break
        else:
            reg *= 10

    if reg >= 1.1:
        warnings.warn('Projection failed: could not find a feasible point')

    return sig_temp_proj, True


def pinv(a):
    u, s, vh = la.svd(a, lapack_driver='gesvd')
    t = u.dtype.char.lower()
    maxS = np.max(s)

    atol = 0.
    rtol = max(a.shape) * np.finfo(t).eps

    val = atol + maxS * rtol
    rank = np.sum(s > val)

    u = u[:, :rank]
    u /= s[:rank]
    B = (u @ vh[:rank]).conj().T

    return B


def exact_tilde_union_info_minimizer(hx, hy, plot=False, ret_obj=False, reg=1e-7):
    dx, dm = hx.shape
    dy, dm_ = hy.shape
    if dm != dm_:
        raise ValueError('Incompatible shapes for Hx and Hy')
    
    sig_temp = hx @ pinv(hy)
    sig_temp_proj = project(sig_temp)[0]
    sig = sig_temp_proj.copy()

    eta_sig = 1e-3 * np.ones((dx, dy))
    beta = 0.9
    alpha = 0.999

    noise_std = 0 
    stop_threshold = 1e-6
    max_iterations = 10000
    patience = 20
    extra_iters = 0

    minima = None
    g_sig_prev = None
    running_obj = []
    if plot:
        running_sig_pre_proj = [sig_temp,]
        running_sig_post_proj = [sig_temp_proj,]
        running_grad = []
        running_eta = []
    i = 1
    extra = 0
    while True:
        obj = objective(sig, hx, hy, dm, dx, dy, reg)

        if minima is None or obj < min(running_obj):
            minima = (sig.copy(), obj)

        if len(running_obj) >= patience:
            if extra == 0:
                if (np.abs(np.array(running_obj[-patience:]) - obj) < stop_threshold).all() or i >= max_iterations:
                    if i >= max_iterations:
                        warnings.warn('Exceeded maximum number of iterations. May not have converged.')
                    if extra_iters == 0: break
                    extra += 1
            elif extra > extra_iters:
                break
            else:
                extra += 1

        if np.isnan(obj):
            running_obj.append(np.inf)
        else:
            running_obj.append(obj)
        i += 1

        g_sig = gradient(sig, hx, hy, dm, dx, dy, reg)
        g_sig = np.sign(g_sig).astype(int)

        # Vanilla gradient descent
        sig_plus = sig - alpha**i * eta_sig * g_sig
        # Project sig back onto the PSD cone
        sig_proj, _ = project(sig_plus)
        if g_sig_prev is not None:
            sign_changed = - g_sig * g_sig_prev
            eta_sig *= beta**sign_changed

        g_sig_prev = g_sig

        if plot:
            running_eta.append(eta_sig)
            running_grad.append(g_sig)
            running_sig_pre_proj.append(sig_plus)
            running_sig_post_proj.append(sig_proj)

        sig[:, :] = sig_proj

    if plot:
        running_sig_pre_proj = np.array(running_sig_pre_proj).squeeze()
        running_sig_post_proj = np.array(running_sig_post_proj).squeeze()
        running_grad = np.array(running_grad).squeeze()

        nrows = 2
        ncols = 2
        plt.figure(figsize=(10, 7))
        plt.subplot(nrows, ncols, 1)
        plt.semilogy(running_obj)
        plt.title('Convergence of objective')
        plt.ylabel('Objective')
        plt.xlabel('Iteration')

        if dx == 1 or dy == 1:
            plt.subplot(nrows, ncols, 2)
            plt.plot(running_sig_pre_proj)
            plt.plot(running_sig_post_proj)
            plt.plot(running_grad)
            plt.title('Convergence of minimizer')
            plt.ylabel('$\Sigma_i$')
            plt.xlabel('Iteration')
        elif dx == 2 and dy == 2:
            plt.subplot(nrows, ncols, 2)
            pre_proj = running_sig_pre_proj.reshape((running_sig_pre_proj.shape[0], -1))
            post_proj = running_sig_post_proj.reshape((running_sig_post_proj.shape[0], -1))
            plt.plot(post_proj)
            plt.title('Convergence of minimizer')
            plt.ylabel('$\Sigma_i$')
            plt.xlabel('Iteration')

        if dx == 1 and dy == 1:
            plt.subplot(nrows, ncols, 3)
            x_ = np.linspace(-1, 1, 100)
            x = 0.5 * (x_[1:] + x_[:-1])
            t = np.arange(len(running_obj) + 1)
            objs = []
            for xi in x:
                sig = np.array([[xi]])
                if np.any(np.eye(dx) - sig @ sig.T <= 0):
                    objs.append(np.nan)
                    continue
                obj = objective(sig, hx, hy, dm, dx, dy, reg)
                objs.append(obj)
            objs = np.repeat(np.array(objs).reshape((1, -1)), len(running_obj), axis=0)
            plt.pcolormesh(t, x_, objs.T, cmap='jet')
            plt.colorbar()
            plt.plot(running_sig_post_proj, 'w-')

        if (dx == 2 and dy == 1) or (dx == 1 and dy == 2):
            plt.subplot(nrows, ncols, 3)
            x, y = np.mgrid[-1:1:100j, -1:1:100j]
            sigs = np.moveaxis(np.array((x, y)), 0, 2)
            objs = []
            for sig in sigs.reshape((-1, 2)):
                sig = sig.reshape((dx, dy))
                if np.any(npla.eigvals(np.eye(dx) - sig @ sig.T) < 0):
                    objs.append(np.nan)
                    continue
                obj = objective(sig, hx, hy, dm, dx, dy, reg)
                objs.append(obj)
            objs = np.array(objs).reshape(sigs.shape[:2])
            plt.pcolormesh(x, y, objs[:-1, :-1], cmap='jet')
            plt.colorbar()
            plt.plot(running_sig_post_proj[:, 0], running_sig_post_proj[:, 1], 'w-')
            plt.plot(running_sig_post_proj[0, 0], running_sig_post_proj[0, 1], 'ko')

        if dx == 2 and dy == 2:
            plt.subplot(nrows, ncols, 3)
            x = np.linspace(-1, 1, 100)
            for i in range(2):
                for j in range(2):
                    post_proj = running_sig_post_proj[-1]
                    sigs = post_proj * np.ones((100, 2, 2))
                    sigs[:, i, j] = x
                    objs = []
                    for sig in sigs:
                        if np.any(npla.eigvals(np.eye(dx) - sig @ sig.T) < 0):
                            objs.append(np.nan)
                            continue
                        obj = objective(sig, hx, hy, dm, dx, dy, reg)
                        objs.append(obj)
                    plt.plot(x, objs, label=('$\Sigma_{%d%d}$' % (i, j)))
            plt.title('Objective around optima')
            plt.xlabel('$\Sigma_{ij}$')
            plt.ylabel('Objective')
            plt.legend()

        plt.show()

    sig, obj = minima

    if ret_obj:
        return sig, obj
    return sig


def bias(d, n):
    return sum(np.log(1 - k / n) for k in range(1, d+1)) / np.log(2) / 2


def compute_bias(du, dv, n):
    return bias(du, n) + bias(dv, n) - bias(du + dv, n)


def debias(imxy, bias_):
    return np.maximum(imxy - bias_, 0)


def exact_gauss_tilde_pid(cov, dm, dx, dy, verbose=False, ret_t_sigt=False,
                          plot=False, unbiased=False, sample_size=None):
    reg = 1e-7

    if unbiased == True and sample_size is None:
        raise ValueError('Must supply sample_size when requesting unbiased estimates')

    ret = whiten(cov, dm, dx, dy, ret_channel_params=True)
    sig_mxy, hx, hy, hxy, sigxy = ret

    imx = 0.5 * npla.slogdet(np.eye(dm) + hx.T @ hx)[1] / np.log(2)
    imy = 0.5 * npla.slogdet(np.eye(dm) + hy.T @ hy)[1] / np.log(2)
    imxy = 0.5 * npla.slogdet(np.eye(dm) + hxy.T @ la.solve(sigxy + reg * np.eye(*sigxy.shape), hxy))[1] / np.log(2)

    if unbiased:
        imx = debias(imx, compute_bias(dm, dx, sample_size))
        imy = debias(imy, compute_bias(dm, dy, sample_size))
        imxy_debiased = debias(imxy, compute_bias(dm, dx + dy, sample_size))
        imxy_debiased = max(imxy_debiased, imx, imy)
    else:
        imxy_debiased = imxy

    debias_factor = imxy_debiased / imxy

    sig, obj = exact_tilde_union_info_minimizer(hx, hy, plot=plot, ret_obj=True, reg=reg)
    covxy__m = np.block([[np.eye(dx), sig], [sig.T, np.eye(dy)]])
    union_info = objective(sig, hx, hy, dm, dx, dy, reg=reg)

    union_info *= debias_factor
    union_info = max(union_info, imx, imy)
    union_info = min(union_info, imx + imy, imxy_debiased)

    uix = union_info - imy
    uiy = union_info - imx
    ri = imx + imy - union_info
    si = imxy_debiased - union_info

    ret = (imx, imy, imxy_debiased, union_info, obj, uix, uiy, ri, si)
    if ret_t_sigt:
        ret = (*ret, None, None, None, sig)

    return ret
