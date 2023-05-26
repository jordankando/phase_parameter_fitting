import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys 
from pyhapke import *
from pyhapke.hapkeFuncs import *



def compute_phase_parameters(filename):

    df = pd.read_csv(filename)

    g_deg = df["Phase Angle"].to_numpy()
    i_deg = np.ones_like(g_deg) * -45
    e_deg = g_deg - 45

    R_700 = df["700"].to_numpy()

    print("g: ", g_deg)
    print("i: ", i_deg)
    print("e: ", e_deg)


    g_rad =  g_deg * np.pi/180
    i_rad =  i_deg * np.pi/180
    e_rad =  e_deg * np.pi/180

    def obj_func(params):
        b = params[0]
        c = params[1]
        w = params[2]
        hs = params[3]
        Bs0 = params[4]


        gs = g_rad
        i = i_rad
        e = e_rad
        R = R_700
        #P = compute_P(gs, b, c, format = "HenyeyGreenstein")
        R_model = np.zeros_like(gs)


        for idx, g in enumerate(gs):

            P_sim = compute_P(g, b, c, format = "HenyeyGreenstein")

            model = HapkeRTM( i = i_rad[idx], e = e_rad[idx], g = g, P = P_sim, wl = None, poros = 0.41, Bs0 = Bs0, Bc0 = 0, hs = hs)

            R_model[idx] = model.hapke_function_REFF(w)

        print(R - R_model)

        return R_model - R
    
    import scipy.optimize

    result = scipy.optimize.least_squares(fun = obj_func, x0 = (0., -0., 0.8, 0, 1),  bounds = ([0,-2, 0., 0., 0.], [1.,2, 1., 1., 1.]), verbose = 2, max_nfev=1e15, ftol = 1e-15, gtol = None, xtol = None)

    print(result.x)

    fit_params = result.x

    print("Nonlinear LSQ result: ", result.x)

    gs = np.linspace(0, np.pi, 1000)
    P_fit = compute_P(gs, fit_params[0], fit_params[1], format = "HenyeyGreenstein")

    plt.figure()
    plt.plot(gs * 180/np.pi, P_fit)
    plt.show()


    gs = np.linspace(0, np.pi, 1000)
    i_rad = np.ones_like(gs) * -np.pi/4
    e_rad = gs - np.pi/4

    R_mod = []


    #print(e_rad * 180/np.pi)

    for idx, g in enumerate(gs):
        P_sim = compute_P(g, fit_params[0], fit_params[1], format = "HenyeyGreenstein" )

        model = HapkeRTM( i = i_rad[idx], e = e_rad[idx], g = g, P = P_sim, wl = None, poros = 0.41, Bs0 = fit_params[4], Bc0 = 0, hs = fit_params[3])

        R_mod.append(model.hapke_function_REFF(fit_params[2]))

    plt.figure()
    plt.plot(gs * 180/np.pi, R_mod)
    plt.plot(g_deg, R_700, '.')
    #plt.ylim(0, 0.7)
    plt.show()

    import emcee

    df_mean = pd.read_csv(filename)


    def log_likelihood(theta, x, y, yerr):
        b, c, w, hs, Bs0, logf = theta
        
        #print(b, c)

        gs = x * np.pi/180
        i_rad = np.ones_like(gs) * -np.pi/4
        e_rad = gs - np.pi/4
        R = y
        R_model = np.zeros_like(gs)



        for idx, g in enumerate(gs):

            P_sim = compute_P(g, b, c, format = "HenyeyGreenstein")

            model = HapkeRTM( i = i_rad[idx], e = e_rad[idx], g = g, P = P_sim, wl = None, poros = 0.41, Bs0 = Bs0, Bc0 = 0, hs = hs)

            R_model[idx] = model.hapke_function_REFF(w)

        sigma2 = R_model **2 * np.exp(2 * logf) + yerr**2

        return -0.5 * np.sum((R - R_model) ** 2 / sigma2+ np.log(sigma2))

    def log_prior(theta):
        b, c, w, hs, Bs0, logf = theta
        if 0. <= b <= 1. and -5 < c < 5 and 0. <= w <= 1. and -10. < logf < 1.0 and 0. <= hs <= 2 and 0. <= Bs0 <= 1.:
            return 0.0
        return -np.inf

    def log_probablility(theta, x, y, yerr):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(theta, x, y, yerr)
    

    x = df_mean["Phase Angle"].to_numpy()
    y = df_mean["700"].to_numpy()
    yerr = df_mean["std"].to_numpy()

    #nonlsq_soln = np.append(result.x, np.array([1]))
    nonlsq_soln = np.append(result.x, np.array([1]))

    pos = nonlsq_soln + 2e-4 * np.random.randn(32, 6)

    nwalkers, ndim = pos.shape

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probablility, args = (x, y, yerr))

    sampler.run_mcmc(pos, 300000, progress=True)

    tau = sampler.get_autocorr_time()
    print(tau)

    fig, axes = plt.subplots(6, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()
    labels = ["b", "c", "w", "hs", "Bs0", "log(f)"]
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number")

    flat_samples = sampler.get_chain(discard=10000, thin=100, flat=True)
    print(flat_samples.shape)

    import corner


    fig = corner.corner(
        flat_samples, labels=labels
    );

    plt.rc('axes', labelsize=12)

    from IPython.display import display, Math

    err_low = []
    err_up = []

    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
        txt = txt.format(mcmc[1], q[0], q[1], labels[i])

        err_low.append(q[0])
        err_up.append(q[1])

        print(txt)
        display(Math(txt))



    result_mcmc = [np.percentile(flat_samples[:,i], 50) for i in range(ndim)]

    gs = np.linspace(0, np.pi, 1000)
    i_rad = np.ones_like(gs) * -np.pi/4
    e_rad = gs - np.pi/4

    R_mod = []

    P_arr = []


    #print(e_rad * 180/np.pi)

    for idx, g in enumerate(gs):
        P_sim = compute_P(g, result_mcmc[0],result_mcmc[1], format = "HenyeyGreenstein")
        P_arr.append(P_sim)

        model = HapkeRTM( i = i_rad[idx], e = e_rad[idx], g = g, P = P_sim, wl = None, poros = 0.41, Bs0 = result_mcmc[4], Bc0 = 0, hs = result_mcmc[3])

        R_mod.append(model.hapke_function_REFF(result_mcmc[2]))


    plt.figure()
    plt.plot(gs * 180/np.pi, R_mod)
    plt.errorbar(x, y, yerr = yerr, marker = '.')
    plt.ylim(0, 0.6)
    plt.xlabel("Phase Angle (°)")
    plt.ylabel("Reflectance")

    ax_lim = plt.gca()

    plt.show()

    """
    plt.figure()
    #plt.plot(gs * 180/np.pi, R_mod)
    plt.errorbar(x, y, yerr = yerr, marker = '.', color = 'orange')
    plt.xlabel("Phase Angle (°)")
    plt.ylabel("Reflectance")
    plt.xlim(ax_lim.get_xlim())
    plt.ylim(ax_lim.get_ylim())

    plt.show()

    plt.figure()
    plt.plot(gs * 180/np.pi, R_mod)
    plt.errorbar(x, y, yerr = yerr, marker = '.', color = 'orange')
    plt.xlabel("Phase Angle (°)")
    plt.ylabel("Reflectance")

    plt.show()
    """
    plt.figure()
    plt.plot(gs * 180/np.pi, P_arr)
    plt.xlabel("Phase Angle")
    plt.ylabel("Single Particle Scattering Phase Function")
    plt.show()


    return (result_mcmc, err_low, err_up)



if __name__ == "__main__":
    (result_mcmc, err_low, err_up) = compute_phase_parameters(sys.argv[1])
    print(out)