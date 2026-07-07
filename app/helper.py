from __future__ import annotations

from __main__ import (
    b1_pdf,
    b2_pdf,
    b_pdf,
    bin_centers,
    bin_edges,
    data,
    minimize,
    np,
    plt,
    s_pdf,
)

fit_no_shape = None
fit_with_shape = None


# Background morphing function

def morph_background(alpha):
    """
    Return the background template for a given shape-nuisance value alpha.

    Parameters
    ----------
    alpha : float
        Background shape nuisance parameter.
        alpha = 0 uses the nominal background shape.
        alpha = +1 uses the +1 sigma background variation.
        alpha = -1 uses the -1 sigma background variation.

    Returns
    -------
    h : numpy array
        A normalized background template whose entries sum to 1.
    """

    # Convert alpha to a plain Python float so comparisons like alpha >= 0
    # behave exactly as expected.
    alpha = float(alpha)

    # For positive alpha, move linearly from the nominal shape to the +1 sigma shape.
    if alpha >= 0:
        h = b_pdf + alpha * (b1_pdf - b_pdf)

    # For negative alpha, move linearly from the nominal shape to the -1 sigma shape.
    # If alpha = -1, then (-alpha) = 1, so this gives b2_pdf.
    else:
        h = b_pdf + (-alpha) * (b2_pdf - b_pdf)

    # Linear extrapolation beyond +/-1 can sometimes create tiny negative bins.
    # A probability template cannot have negative entries, so clip to a tiny positive value.
    h = np.clip(h, 1e-12, None)

    # Renormalize so the background template remains a probability distribution.
    # This keeps B interpretable as the total background yield.
    h = h / np.sum(h)

    return h


# Expected-count model

def expected_counts(S, B, alpha=0.0):
    """
    Compute the expected number of events in each bin.

    Parameters
    ----------
    S : float
        Total signal yield. If S = 20, the model predicts 20 total signal events.

    B : float
        Total background yield. If B = 705, the model predicts 705 total background events.

    alpha : float
        Background shape nuisance parameter.

    Returns
    -------
    mu : numpy array
        Expected event counts in each bin.
    """

    # Convert the normalized signal template into absolute expected signal counts.
    signal_counts = S * s_pdf

    # Convert the morphed normalized background template into absolute background counts.
    background_counts = B * morph_background(alpha)

    # The observed data are assumed to be signal plus background.
    mu = signal_counts + background_counts

    return mu


# Extended binned Poisson NLL

def nll_ex1(parameters, include_shape_systematic=True, fixed_S=None):
    """
    Extended binned negative log-likelihood for Exercise 1.

    This function is written in a flexible way so it can be used for:
      1. a full fit, where S is free;
      2. a profile scan, where S is fixed and nuisance parameters are refit;
      3. fits with or without the background shape systematic.

    Parameters
    ----------
    parameters : array-like
        If fixed_S is None and include_shape_systematic is True:
            parameters = [S, B, alpha]
        If fixed_S is None and include_shape_systematic is False:
            parameters = [S, B]
        If fixed_S is not None and include_shape_systematic is True:
            parameters = [B, alpha]
        If fixed_S is not None and include_shape_systematic is False:
            parameters = [B]

    include_shape_systematic : bool
        If True, alpha is fitted and the Gaussian alpha^2/2 constraint is included.
        If False, alpha is fixed to 0.

    fixed_S : float or None
        If None, S is a fit parameter.
        If a number, S is fixed to that value. This is used in profile scans.

    Returns
    -------
    nll : float
        Negative log-likelihood value, up to constants independent of parameters.
    """

    # ------------------------------------------------------------
    # Decode the parameter vector depending on what kind of fit we are doing.
    # ------------------------------------------------------------

    if fixed_S is None:
        # Full fit: S is one of the parameters.
        if include_shape_systematic:
            S, B, alpha = parameters
        else:
            S, B = parameters
            alpha = 0.0
    else:
        # Profile scan: S is fixed by the scan; only nuisance parameters are fitted.
        S = fixed_S
        if include_shape_systematic:
            B, alpha = parameters
        else:
            (B,) = parameters
            alpha = 0.0

    # ------------------------------------------------------------
    # Enforce physical parameter regions.
    # ------------------------------------------------------------

    # Negative signal/background yields are not physically meaningful here.
    # Returning a huge NLL tells the minimizer: "do not go here".
    if S < 0 or B < 0:
        return 1e100

    # ------------------------------------------------------------
    # Compute model prediction mu_i for all bins.
    # ------------------------------------------------------------

    mu = expected_counts(S, B, alpha)

    # The logarithm requires positive expected counts.
    # This should be guaranteed by clipping in morph_background, but this check
    # makes the function robust.
    if np.any(mu <= 0) or not np.all(np.isfinite(mu)):
        return 1e100

    # ------------------------------------------------------------
    # Poisson NLL, dropping constants ln(d_i!) that do not depend on parameters.
    # ------------------------------------------------------------

    nll = np.sum(mu - data * np.log(mu))

    # ------------------------------------------------------------
    # Gaussian constraint for the shape nuisance alpha.
    # ------------------------------------------------------------

    if include_shape_systematic:
        # alpha is measured in units of its own 1-sigma systematic uncertainty.
        # A standard-normal prior contributes alpha^2/2 to the NLL.
        nll += 0.5 * alpha**2

    return float(nll)


# Backwards-compatible wrappers and helpers for notebook usage
def _mode_to_bool(mode):
    if isinstance(mode, str):
        mode = mode.lower()
        # Accept both the old short names and the new verbose names.
        if mode in (
            "with_shape",
            "withshape",
            "shape",
            "with_systematic_shape",
            "withsystematicshape",
            "with-systematic-shape",
            "with systematic shape",
        ):
            return True
        if mode in (
            "no_shape",
            "noshape",
            "no-shape",
            "no shape",
            "no_systematic_shape",
            "nosystematicshape",
            "no-systematic-shape",
            "no systematic shape",
        ):
            return False
        raise ValueError(f"Unknown mode string: {mode}")
    return bool(mode)


def nll_extended(parameters, mode="with_systematic_shape", fixed_S=None):
    include = _mode_to_bool(mode)
    return nll_ex1(parameters, include_shape_systematic=include, fixed_S=fixed_S)


# NLL minimization

def fit_ex1(include_shape_systematic):
    """
    Fit Exercise 1 by minimizing the NLL.

    Parameters
    ----------
    include_shape_systematic : bool
        If False, fit S and B with alpha fixed to zero.
        If True, fit S, B, and alpha.

    Returns
    -------
    result : scipy OptimizeResult
        Object containing the best-fit parameters and NLL minimum.
    """

    # The total observed event count is a good scale for initial guesses.
    total_data = np.sum(data)

    # Start with a small signal and the rest background.
    # The exact starting point should not matter much for this simple problem.
    initial_S = 20.0
    initial_B = total_data - initial_S

    if include_shape_systematic:
        # Parameter order: [S, B, alpha].
        x0 = np.array([initial_S, initial_B, 0.0])

        # Bounds:
        #   S >= 0 and B >= 0 are physical yield constraints.
        #   alpha is allowed to move over several sigma.
        bounds = [(0.0, None), (0.0, None), (-5.0, 5.0)]
    else:
        # Parameter order: [S, B]. Alpha is fixed to zero inside nll_ex1.
        x0 = np.array([initial_S, initial_B])
        bounds = [(0.0, None), (0.0, None)]

    # L-BFGS-B is a standard minimizer that supports simple bounds.
    result = minimize(
        fun=lambda p: nll_ex1(p, include_shape_systematic=include_shape_systematic),
        x0=x0,
        method="L-BFGS-B",
        bounds=bounds,
    )

    return result


# Stacked fit plots

def plot_fit_result(fit_result, include_shape_systematic=None, mode=None, title=None):
    """
    Plot the fitted signal + background model as a stacked histogram.
    """

    # Decode best-fit parameters.
    if mode is not None:
        include_shape_systematic = _mode_to_bool(mode)
    if include_shape_systematic is None:
        include_shape_systematic = True

    if include_shape_systematic:
        S_hat, B_hat, alpha_hat = fit_result.x
    else:
        S_hat, B_hat = fit_result.x
        alpha_hat = 0.0

    # Convert normalized templates to fitted absolute event counts.
    fitted_signal = S_hat * s_pdf
    fitted_background = B_hat * morph_background(alpha_hat)
    fitted_total = fitted_signal + fitted_background

    # Bin widths are needed for a proper bar plot.
    bin_widths = np.diff(bin_edges)

    fig, ax = plt.subplots()

    # Draw background first.
    ax.bar(
        bin_centers,
        fitted_background,
        width=bin_widths,
        align="center",
        label="Fitted background",
        alpha=0.7,
    )

    # Draw signal stacked on top of background.
    ax.bar(
        bin_centers,
        fitted_signal,
        bottom=fitted_background,
        width=bin_widths,
        align="center",
        label="Fitted signal",
        alpha=0.7,
    )

    # Overlay the observed data points.
    ax.errorbar(
        bin_centers,
        data,
        yerr=np.sqrt(np.maximum(data, 1.0)),
        fmt="o",
        color="black",
        label="Observed data",
    )

    # Overlay the total fitted prediction as a line.
    ax.step(bin_centers, fitted_total, where="mid", color="black", linestyle="--", label="Total fit")

    ax.set_xlabel("Multivariate discriminant score")
    ax.set_ylabel("Events per bin")
    ax.set_title(title)
    ax.legend()
    plt.show()

    # Return fitted components in case we want to inspect them later.
    return fitted_signal, fitted_background, fitted_total


# Profile likelihood scan in S

def profile_scan_in_S(include_shape_systematic, S_values):
    """
    Compute the profile NLL as a function of fixed S.

    Parameters
    ----------
    include_shape_systematic : bool
        Whether to include/profile the background shape nuisance alpha.

    S_values : numpy array
        Values of S at which to evaluate the profile NLL.

    Returns
    -------
    nll_values : numpy array
        Profile NLL value at each S.

    profiled_nuisance_values : numpy array
        Best-fit nuisance parameters at each fixed S.
        Columns are [B] without shape systematic or [B, alpha] with it.
    """

    # Choose the global best fit as the first guess for nuisance parameters.
    if isinstance(include_shape_systematic, str):
        include = _mode_to_bool(include_shape_systematic)
    else:
        include = bool(include_shape_systematic)

    if include:
        # Nuisance parameters are [B, alpha].
        current_guess = fit_with_shape.x[1:].copy()
        bounds = [(0.0, None), (-5.0, 5.0)]
    else:
        # Nuisance parameter is [B].
        current_guess = np.array([fit_no_shape.x[1]])
        bounds = [(0.0, None)]

    nll_values = []
    profiled_nuisance_values = []

    # Loop over fixed S values.
    for S_fixed in S_values:
        # Minimize the NLL over nuisance parameters only, with S held fixed.
        result = minimize(
            fun=lambda nuisance: nll_ex1(
                nuisance,
                include_shape_systematic=include_shape_systematic,
                fixed_S=S_fixed,
            ),
            x0=current_guess,
            method="L-BFGS-B",
            bounds=bounds,
        )

        # Store the minimized NLL and the best nuisance parameters.
        nll_values.append(result.fun)
        profiled_nuisance_values.append(result.x.copy())

        # Use the current solution as the next starting point.
        # This makes the scan smooth and fast because nearby S values have nearby minima.
        current_guess = result.x

    return np.array(nll_values), np.array(profiled_nuisance_values)


def find_interval_from_delta_nll(x_values, delta_nll, level=0.5):
    """
    Find left and right crossings of delta_nll = level by linear interpolation.
    """

    # Index of the smallest Delta NLL value on the grid.
    i_min = int(np.argmin(delta_nll))

    # Defaults in case a crossing is outside the scan range.
    left_crossing = np.nan
    right_crossing = np.nan

    # Search left of the minimum for a crossing.
    for i in range(i_min, 0, -1):
        y1 = delta_nll[i]
        y0 = delta_nll[i - 1]
        if (y1 - level) * (y0 - level) <= 0:
            left_crossing = np.interp(level, [y1, y0], [x_values[i], x_values[i - 1]])
            break

    # Search right of the minimum for a crossing.
    for i in range(i_min, len(x_values) - 1):
        y0 = delta_nll[i]
        y1 = delta_nll[i + 1]
        if (y0 - level) * (y1 - level) <= 0:
            right_crossing = np.interp(level, [y0, y1], [x_values[i], x_values[i + 1]])
            break

    return left_crossing, right_crossing


# Parabolic approximation near the NLL minimum

def fit_local_parabola(S_values, delta_nll_values, max_delta_for_fit=1.5):
    """
    Fit a quadratic polynomial to the region near the minimum.

    The fitted polynomial is:
        y = a*S^2 + b*S + c

    From this, the parabola minimum is at:
        S_hat_parabola = -b/(2a)

    If DeltaNLL = (S-S_hat)^2/(2 sigma^2), then:
        a = 1/(2 sigma^2)
        sigma = sqrt(1/(2a))
    """

    # Use only points close to the minimum, where the Taylor expansion should work.
    mask = delta_nll_values < max_delta_for_fit

    # Fit a second-degree polynomial to the selected points.
    a, b, c = np.polyfit(S_values[mask], delta_nll_values[mask], deg=2)

    # Extract the minimum and curvature-based uncertainty.
    S_hat_parabola = -b / (2.0 * a)
    sigma_parabola = np.sqrt(1.0 / (2.0 * a))

    # Evaluate the fitted parabola over the full scan range for plotting.
    fitted_curve = a * S_values**2 + b * S_values + c

    return S_hat_parabola, sigma_parabola, fitted_curve


# Optional: compare the NLL fit to a Pearson chi-square value

def pearson_chi2(mu):
    """
    Pearson chi-square statistic for Poisson counts.

    This is not the minimized objective above. It is shown for intuition.
    """
    return np.sum((data - mu)**2 / mu)
