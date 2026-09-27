"""
A line-by-line NumPy mirror of cma/core.py (TensorFlow), used only to visualize
the 2D example in this workspace. Every formula below is copied 1:1 from
cma/core.py::init() and cma/core.py::search(), with tf.* -> np.*.

Row-vector convention is kept exactly as in the original:
    tf.matmul(z, tf.matmul(B, D))  ==  z @ (B @ D)

-------------------------------------------------------------------------------
EQUATION MAP -- Hansen 2023, "The CMA Evolution Strategy: A Tutorial"
Every code line carries a trailing tag naming the paper object it implements:
  eq.(NN)        numbered equation of the paper
  eqs.(NN),(NN)  several numbered equations
  eq. none --    no counterpart equation (plumbing, bookkeeping, numerical guard)
  fig6:          pseudo-code of Figure 6 (Appendix A, unnumbered)
  table1:        Table 1 of Appendix A (unnumbered)
  chiN:          E||N(0,I)|| (Appendix A, unnumbered)
  input:         problem-dependent input of the strategy
  appendix ...:  unnumbered bullet of Appendix B.3 / B.5
The same map is printed as a table in examples/eq_code_map.md, generated from
this file by examples/check_eq_annotation_map.py.
-------------------------------------------------------------------------------
"""
import numpy as np  # eq. none -- plumbing (the tf.* -> np.* import of the mirror)


def cma_es_numpy(  # eq. none -- the mirror of core.py::init() + core.py::search()
    initial_solution,  # input: m^(0) -- 'Choose distribution mean m in R^n ... problem dependent' (Fig. 6)
    initial_step_size,  # input: sigma^(0) -- 'Choose ... step-size sigma in R_>0' (Fig. 6)
    fitness_function,  # input: f -- the objective; its ranking defines x_{i:lambda} of eq.(6)
    max_generations=100,  # fig6: the outer loop 'Until termination criterion met, g <- g+1'
    population_size=None,  # eq.(48): lambda; eq.(48) is only the default, 'lambda can be increased'
    enforce_bounds=None,  # eq. none -- appendix: boundaries and constraints (B.5)
    seed=42,  # eq. none -- plumbing (RNG seed)
    dtype=np.float32,  # eq. none -- precision of the state (appendix B.2/B.4 concerns)
    # ---- opt-in replica of core.py::should_terminate (lines 353-391) ----
    # Defaults keep the behaviour of the previously validated mirror.
    terminate=False,
    termination_no_effect=1e-8,  # eq. none -- appendix B.3 bullet, unnumbered (threshold of NoEffectAxis/NoEffectCoord)
    termination_condition=1e14,  # eq. none -- appendix B.3 bullet, unnumbered (ConditionCov: cond(C) > 1e14)
    termination_tol_x_up=1e4,  # eq. none -- appendix B.3 bullet, unnumbered (TolXUp: increase by more than 1e4)
    # Fitness is evaluated in float64 by default (numerically clearer).
    # Pass fitness_dtype=dtype to reproduce core.py's float32 evaluation exactly,
    # which matters when f overflows (|f| > 3.4e38).
    fitness_dtype=np.float64,  # eq. none -- float64 f improves the numerics over core.py's float32 (appendix B.4)
):
    rng = np.random.default_rng(seed)  # eq. none -- plumbing (RNG; tf.random.normal in core.py)

    # ---------------- init() ----------------
    dimension = len(initial_solution)  # eq.(5): n = dimension of R^n (also the n of eqs.(48)-(58))
    N = float(dimension)  # eq.(5): n as float -- feeds every learning rate of eqs.(8),(55)-(58)

    if population_size is not None:  # eq. none -- plumbing (lambda may be overridden by the caller)
        lam = float(population_size)  # eq.(48): lambda is the default 4 + floor(3 ln n); here it comes from the caller
    else:  # eq. none -- plumbing
        lam = float(np.floor(np.log(N) * 3 + 8))  # eq.(48): lambda = 4 + floor(3 ln n)
    lam_i = int(lam)  # eq. none -- plumbing (lambda as int, for indexing)
    N_i = int(N)  # eq. none -- plumbing (n as int, for indexing)

    mu = float(np.floor(lam / 2))  # table1: mu = |{w_i > 0}| = floor(lambda/2)
    mu_i = int(mu)  # eq. none -- plumbing (mu as int, for indexing)

    # weights: log(mu + 0.5) - log(1..mu), zero-padded to lambda, normalized
    # NOTE ON FIDELITY: np.log(<python float>) returns a *float64* numpy scalar, which
    # would silently promote `w` (and then m, x, C, ...) to float64 from generation 2 on.
    # core.py keeps everything in self.dtype, so the cast below is required.
    w = (np.log(dtype(mu + 0.5)) - np.log(np.arange(1, mu_i + 1, dtype=dtype))).astype(dtype)  # eq.(49) with (lambda+1)/2 == mu+0.5, truncated to i = 1..mu
    w = np.concatenate([w, np.zeros(lam_i - mu_i, dtype=dtype)])  # eq.(53) negative branch -> 0 (this repo keeps only the positive weights)
    w = (w / w.sum())[:, None]  # eq.(53) positive branch: w_i = w'_i / sum|w'_j|+ , hence sum_i w_i = 1 (eq.(7)); (lambda, 1) column

    mu_eff = float(w.sum() ** 2 / (w ** 2).sum())  # eq.(8): mu_eff = (sum|w_i|)^2 / sum w_i^2 = 1 / sum w_i^2

    cc = (4 + mu_eff / N) / (N + 4 + 2 * mu_eff / N)  # eq.(56): c_c = (4 + mu_eff/n) / (n + 4 + 2 mu_eff/n)
    c_sigma = (mu_eff + 2) / (N + mu_eff + 5)  # eq.(55) 1st line: c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
    c1 = 2 / ((N + 1.3) ** 2 + mu_eff)  # eq.(57): c_1 = alpha_cov / ((n + 1.3)^2 + mu_eff) with alpha_cov = 2
    c_mu = 2 * (mu_eff - 2 + 1 / mu_eff) / ((N + 2) ** 2 + 2 * mu_eff / 2)  # eq.(58) with alpha_cov = 2 (the min(1-c1, .) clamp of eq.(58) is omitted here)
    damps = 1 + 2 * max(0.0, np.sqrt((mu_eff - 1) / (N + 1)) - 1) + c_sigma  # eq.(55) 2nd line: d_sigma = 1 + 2 max(0, sqrt((mu_eff-1)/(n+1)) - 1) + c_sigma
    chiN = np.sqrt(N) * (1 - 1 / (4 * N) + 1 / (21 * N ** 2))  # chiN: E||N(0,I)|| ~ sqrt(n) (1 - 1/(4n) + 1/(21 n^2)), appendix A (unnumbered)

    m = np.array(initial_solution, dtype=dtype)  # fig6: initialization -- distribution mean m in R^n (problem dependent)
    sigma = dtype(initial_step_size)  # fig6: initialization -- step-size sigma in R_>0 (problem dependent)
    C = np.eye(N_i, dtype=dtype)  # fig6: initialization -- C = I  (eq.(2) with B = D = I)
    p_sigma = np.zeros(N_i, dtype=dtype)  # fig6: initialization -- p_sigma = 0 (used by eq.(43))
    p_C = np.zeros(N_i, dtype=dtype)  # fig6: initialization -- p_c = 0 (used by eq.(45))
    B = np.eye(N_i, dtype=dtype)  # eq.(2): B = I is the eigenvector basis of C = I
    D = np.eye(N_i, dtype=dtype)  # eq.(2): D = I, D = diag(d_1..d_n) with d_i = sqrt(eigenvalue_i of C)

    trace = []  # eq. none -- bookkeeping (state trace, cf. core.py::_store_trace)

    params = dict(  # eq. none -- bookkeeping (the strategy parameters of eqs.(8),(48)-(58))
        lam=lam_i, mu=mu_i, weights=w.ravel(), mu_eff=mu_eff, cc=cc, c_sigma=c_sigma,
        c1=c1, c_mu=c_mu, damps=damps, chiN=chiN,  # eq. none -- bookkeeping
    )

    # ---------------- search() ----------------
    for generation in range(1, max_generations + 1):  # fig6: 'Until termination criterion met, g <- g+1'
        trace.append(dict(  # eq. none -- bookkeeping (state *before* this generation is sampled)
            generation=generation, m=m.copy(), sigma=float(sigma), C=C.copy(),
            B=B.copy(), D=D.copy(), p_sigma=p_sigma.copy(), p_C=p_C.copy(),  # eq. none -- bookkeeping
            population=None,  # eq. none -- bookkeeping
        ))  # eq. none -- bookkeeping

        # (1) sample ~ N(m, sigma^2 C)
        z = rng.standard_normal((lam_i, N_i)).astype(dtype)  # eq.(38): z_k ~ N(0, I), k = 1..lambda
        y = z @ (B @ D)  # eq.(39): y_k = B D z_k, row-vector form (C = B D^2 B^T of eq.(2))
        x = m + sigma * y  # eq.(40): x_k = m + sigma y_k ~ N(m, sigma^2 C); identical to eq.(5), cf. eq.(4)

        penalty = 0.0  # eq. none -- appendix B.5 (not part of the paper's pseudo-code)
        if enforce_bounds is not None:  # eq. none -- appendix B.5
            lo = np.asarray(enforce_bounds, dtype=dtype)[:, 0]  # eq. none -- appendix B.5
            hi = np.asarray(enforce_bounds, dtype=dtype)[:, 1]  # eq. none -- appendix B.5
            x_corr = np.clip(x, lo, hi)  # eq. none -- appendix B.5
            penalty = float(np.sum((x.astype(np.float64) - x_corr.astype(np.float64)) ** 2))  # eq. none -- appendix B.5 (core.py adds one scalar penalty to the whole batch)
            x = x_corr  # eq. none -- appendix B.5

        # (2) selection & recombination
        f_x = np.asarray(fitness_function(x.astype(fitness_dtype))).astype(np.float64) + penalty  # eq. none -- f evaluation; the ranking below defines x_{i:lambda} of eq.(6)
        order = np.argsort(f_x, kind='stable')  # eq.(6): x_{1:lambda} <= x_{2:lambda} <= ... ranked by f (truncation selection)
        x_sorted = x[order]  # eq.(6),(7): the selected points x_{i:lambda}, i = 1..mu

        x_diff = x_sorted - m  # eq.(9),(15): x_{i:lambda} - m^(g), the numerator of y_{i:lambda}
        x_mean = (x_diff * w).sum(axis=0)  # eq.(9) with c_m = 1 (eq.(54)): sum_i w_i (x_{i:lambda} - m^(g))
        m = (m + x_mean).astype(dtype)  # eq.(9),(42): m <- m + c_m sigma <y>_w == sum_i w_i x_{i:lambda}; tf.Variable.assign casts to dtype

        # (3) covariance matrix adaptation
        y_mean = x_mean / sigma  # eq.(41): <y>_w = (m^(g+1) - m^(g)) / (c_m sigma^(g)) = x_mean / sigma, c_m = 1
        p_C = ((1 - cc) * p_C + np.sqrt(cc * (2 - cc) * mu_eff) * y_mean).astype(dtype)  # eq.(24), eq.(45): p_c <- (1-c_c) p_c + sqrt(c_c (2-c_c) mu_eff) <y>_w, i.e. eq.(45) with h_sigma = 1

        C_m = (x_diff / sigma)[:, :, None] * (x_diff / sigma)[:, None, :]  # eq.(15),(16): the outer products y_{i:lambda} y_{i:lambda}^T, shape (lambda, N, N)
        y_s = (C_m * w[:, :, None]).sum(axis=0)  # eq.(16): rank-mu update C_mu = sum_i w_i y_{i:lambda} y_{i:lambda}^T

        C = (1 - c1 - c_mu) * C + c1 * np.outer(p_C, p_C) + c_mu * y_s  # eq.(30): C <- (1 - c_1 - c_mu sum w_j) C + c_1 p_c p_c^T + c_mu sum_i w_i y_{i:lambda} y_{i:lambda}^T

        # enforce symmetry (copy upper triangle onto lower)
        C_upper = np.triu(C)  # eq. none -- numerical hygiene: C stays symmetric analytically, not in floating point
        C = (C_upper + np.triu(C_upper, 1).T).astype(dtype)  # eq. none -- numerical hygiene: copy the upper triangle onto the lower one

        # (4) step-size control
        D_inv = np.diag(1.0 / np.diag(D))  # eq.(2),(3): D^-1 = diag(1/d_1 ... 1/d_n)
        C_inv_squared = B @ D_inv @ B.T  # eq.(31): C^(-1/2) = B D^-1 B^T, the definition used by the conjugate path
        prev_sigma = float(sigma)  # eq. none -- needed only by the TolXUp criterion (appendix B.3)
        prev_D = D.copy()  # eq. none -- needed only by the TolXUp criterion (appendix B.3)

        p_sigma = ((1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff)  # eq.(43): p_sigma <- (1-c_sigma) p_sigma + sqrt(c_sigma (2-c_sigma) mu_eff) C^(-1/2) <y>_w
                   * (C_inv_squared @ y_mean)).astype(dtype)  # eq.(43) = eq.(31) with c_m = 1: the conjugate transformation of the mean step

        sigma = dtype(sigma * np.exp((c_sigma / damps) * (np.linalg.norm(p_sigma) / chiN - 1)))  # eq.(44) = eq.(37), the log form of eq.(32): sigma <- sigma exp((c_sigma/d_sigma)(||p_sigma||/chiN - 1))

        # (5) eigen decomposition
        # NOTE: tf.linalg.svd(C) returns (s, u, v): singular values FIRST.
        # np.linalg.svd(C) returns (u, s, vh): singular values SECOND.
        if not np.all(np.isfinite(C)):  # eq. none -- numerical guard (a NaN C would make every comparison below False)
            # C has blown up: this is the "numerical meltdown" branch. TF's svd would
            # return NaN here too, after which every comparison below is False and NO
            # termination criterion can ever fire (see the report).
            trace[-1]['population'] = x_sorted  # eq. none -- bookkeeping of the melt-down branch
            trace[-1]['fitness'] = f_x[order]  # eq. none -- bookkeeping
            trace[-1]['diagnostics'] = dict(
                nonfinite=True, condition_number=np.nan, max_D=np.nan, min_D=np.nan,
                sigma_max_d=np.nan, norm_p_sigma=float(np.linalg.norm(p_sigma)),  # eq.(43): ||p_sigma||, the driver of the step-size change of eq.(44)
                no_effect_axis=False, no_effect_coord=False, condition_cov=False,
                tol_x_up=False, f_mean=float('nan'),
            )
            return dict(trace=trace, m=m, sigma=float(sigma), C=C, params=params,  # eq. none -- early return of the melt-down branch
                        termination=dict(generation=generation, nonfinite=True,
                                         which=dict(nonfinite=True)))

        U, s, _ = np.linalg.svd(C)  # eq.(2): eigendecomposition C = B D^2 B^T (SVD of the symmetric C)
        B = U  # eq.(2): the columns of B are the orthonormal eigenvectors of C
        diag_D = np.sqrt(s)  # eq.(2): d_i = sqrt(eigenvalue_i), the diagonal of D
        D = np.diag(diag_D)  # eq.(2): D = diag(d_1 ... d_n)

        trace[-1]['population'] = x_sorted  # eq. none -- bookkeeping
        trace[-1]['fitness'] = f_x[order]  # eq. none -- bookkeeping

        # ---- core.py::should_terminate (lines 353-391) ----
        # Faithful detail: core.py uses B[i, :] (ROW i), not column i.
        i = generation % N_i  # appendix B.3, footnote 31: i = (g mod n) + 1, one principal axis per generation
        m_nea_diff = np.abs(0.1 * sigma * diag_D[i] * B[i, :])  # appendix B.3 NoEffectAxis: the vector 0.1 sigma d_ii b_i (core.py uses the row B[i,:])
        no_effect_axis = bool(np.all(m_nea_diff < termination_no_effect))  # appendix B.3 NoEffectAxis: stop if adding it does not change m
        m_nec_diff = np.abs(0.2 * sigma * np.diag(C))  # appendix B.3 NoEffectCoord: the vector 0.2 sigma c_ii e_i, c_ii = diag(C)
        no_effect_coord = bool(np.any(m_nec_diff < termination_no_effect))  # appendix B.3 NoEffectCoord: stop if m_i + 0.2 sigma c_ii == m_i (reduce_any)
        condition_number = float(np.max(diag_D) ** 2 / np.min(diag_D) ** 2)  # appendix B.3 ConditionCov: cond(C) = lambda_max/lambda_min = max(D)^2/min(D)^2
        condition_cov = bool(condition_number > termination_condition)  # appendix B.3 ConditionCov: stop if cond(C) > 1e14
        tol_x_up = bool(abs(float(sigma) * np.max(diag_D)  # appendix B.3 TolXUp: sigma x max(diag(D)) (core.py takes an absolute difference)
                            - prev_sigma * float(np.max(np.diag(prev_D)))) > termination_tol_x_up)  # appendix B.3 TolXUp: stop if it increased by more than 1e4

        trace[-1]['diagnostics'] = dict(  # eq. none -- bookkeeping (core.py keeps no such diagnostics)
            nonfinite=False,  # eq. none -- bookkeeping
            no_effect_axis=no_effect_axis, no_effect_coord=no_effect_coord,  # eq. none -- bookkeeping
            condition_cov=condition_cov, tol_x_up=tol_x_up,  # eq. none -- bookkeeping
            condition_number=condition_number,  # eq. none -- bookkeeping
            max_D=float(np.max(diag_D)), min_D=float(np.min(diag_D)),  # eq. none -- bookkeeping
            sigma_max_d=float(sigma) * float(np.max(diag_D)),  # eq. none -- bookkeeping
            norm_p_sigma=float(np.linalg.norm(p_sigma)),  # eq. none -- bookkeeping
            f_mean=float(np.asarray(fitness_function(m[None, :].astype(fitness_dtype)))[0]),  # eq. none -- f(m^(g)) of the current mean, for the trace
        )  # eq. none -- bookkeeping

        if terminate and (no_effect_axis or no_effect_coord or condition_cov or tol_x_up):  # eq. none -- appendix B.3: evaluated once per generation, after the update (as in core.py)
            return dict(trace=trace, m=m, sigma=float(sigma), C=C, params=params,  # eq. none -- early return when a criterion of appendix B.3 fires
                        termination=dict(generation=generation, nonfinite=False, which=dict(  # eq. none -- bookkeeping
                            no_effect_axis=no_effect_axis, no_effect_coord=no_effect_coord,  # eq. none -- bookkeeping
                            condition_cov=condition_cov, tol_x_up=tol_x_up)))  # eq. none -- bookkeeping

    return dict(trace=trace, m=m, sigma=float(sigma), C=C, params=params,  # eq. none -- normal return after max_generations
                termination=None)  # eq. none -- bookkeeping


if __name__ == '__main__':  # eq. none -- demo driver (the rotated anisotropic quadratic of Figures 1-5)
    # ---- 2D anisotropic, rotated quadratic: f(x) = 0.5 x^T A x, optimum at 0 ----
    A = np.array([[1.0, 0.9], [0.9, 1.0]])  # eq.(2): A plays the role of the inverse covariance that C must adapt to

    def quadratic(x):  # eq. none -- demo objective (the ellipsoid model of Sect. 0.4)
        return 0.5 * np.einsum('ij,jk,ik->i', x, A, x)  # eq. none -- demo objective

    res = cma_es_numpy([4.0, 0.5], 0.5, quadratic, max_generations=40, seed=444)  # fig6: problem-dependent m^(0) = [4.0, 0.5] and sigma^(0) = 0.5
    p = res['params']  # eq. none -- demo plumbing
    print('lambda=%d mu=%d mu_eff=%.3f cc=%.4f c_sigma=%.4f c1=%.5f c_mu=%.5f damps=%.4f chiN=%.4f'  # eqs.(8),(48),(49),(50),(51),(52),(53),(54),(55),(56),(57),(58): the Table 1 defaults
          % (p['lam'], p['mu'], p['mu_eff'], p['cc'], p['c_sigma'], p['c1'], p['c_mu'], p['damps'], p['chiN']))  # eq. none -- demo plumbing
    print('weights =', np.round(p['weights'], 4))  # eq. none -- demo plumbing
    print()  # eq. none -- demo plumbing
    print('gen |        m         | sigma  |   C11    C12    C22  |  d1     d2   | angle(deg)')  # eq. none -- demo plumbing
    for t in res['trace'][:14]:  # eq. none -- demo plumbing
        C = t['C']  # eq. none -- demo plumbing
        vals, vecs = np.linalg.eigh(C)  # eq.(2): eigh(C) gives the eigenvectors/eigenvalues used by eqs.(2),(4),(31),(39)
        ang = np.degrees(np.arctan2(vecs[1, -1], vecs[0, -1]))  # eq.(2): the eigenvector directions, here reported as an angle
        d = np.sqrt(vals)  # eq.(2): d_i = sqrt(eigenvalue_i) -- the same D as in the sampling eq.(39)
        print('%3d | (%7.4f,%7.4f) | %6.4f | %6.4f %6.4f %6.4f | %6.3f %6.3f | %8.2f'  # eq. none -- demo plumbing
              % (t['generation'], t['m'][0], t['m'][1], t['sigma'],  # eq. none -- demo plumbing
                 C[0, 0], C[0, 1], C[1, 1], d[1], d[0], ang))  # eq. none -- demo plumbing
