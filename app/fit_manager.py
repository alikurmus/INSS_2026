"""Fit manager class wrapping the helper fit functions for notebook use.

This small class centralizes running the two fit modes, printing
results, and producing stacked-fit plots, while keeping the low-level
service functions in `app.helper`.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import app.helper as helper


class FitManager:
    """Manage and run Exercise 1 fits in both modes.

    Attributes
    ----------
    fit_no_shape : OptimizeResult | None
    fit_with_shape : OptimizeResult | None
    """

    def __init__(self) -> None:
        self.fit_no_shape = None
        self.fit_with_shape = None

    def run(self, mode="with_systematic_shape"):
        """Run a single fit given a mode string or boolean.

        Returns the OptimizeResult.
        """
        result = helper.fit_extended(mode=mode)
        return result

    def run_both(self):
        """Run both fits and store results on the instance.

        Returns (fit_no_shape, fit_with_shape)
        """
        self.fit_no_shape = helper.fit_extended(mode="no_systematic_shape")
        self.fit_with_shape = helper.fit_extended(mode="with_systematic_shape")
        return self.fit_no_shape, self.fit_with_shape

    def profile_scan_for_mode(self, S_values, mode="no_systematic_shape"):
        """Run a profile scan in S for a single mode.

        Returns (S_values, nll_scan, delta_nll, profiled_nuisance)
        """
        # Ensure the corresponding global fit exists for the baseline NLL_min
        if mode in ("no_systematic_shape", "no_shape", False):
            if self.fit_no_shape is None:
                self.run_both()
            baseline = self.fit_no_shape
        else:
            if self.fit_with_shape is None:
                self.run_both()
            baseline = self.fit_with_shape

        # Compute the profile NLL scan using the helper function.
        nll_scan, profiled_nuisance = helper.profile_scan_in_S(mode, S_values)

        # Delta NLL relative to the baseline NLL minimum.
        delta_nll = nll_scan - baseline.fun

        return S_values, nll_scan, delta_nll, profiled_nuisance

    def run_profile_scans(self, S_values=None):
        """Run both profile scans (no-shape and with-shape).

        Returns a dict with keys 'no' and 'with' mapping to the tuples
        returned by `profile_scan_for_mode`.
        """
        if S_values is None:
            S_values = np.linspace(0.0, 120.0, 601)

        no_scan = self.profile_scan_for_mode(S_values, mode="no_systematic_shape")
        with_scan = self.profile_scan_for_mode(S_values, mode="with_systematic_shape")

        # Use verbose, self-descriptive keys for clarity in notebooks.
        return {
            "no_systematic_shape": no_scan,
            "with_systematic_shape": with_scan,
        }

    def find_interval(self, S_values, delta_nll, level=0.5):
        """Convenience wrapper around helper.find_interval_from_delta_nll."""
        return helper.find_interval_from_delta_nll(S_values, delta_nll, level=level)

    def plot_profile_scans(self, S_values, delta_nll_no, delta_nll_with, level=0.5, **plot_kwargs):
        """Plot two delta-NLL scans on the same axes and annotate intervals.

        Returns a dict with interval endpoints for both scans.
        """
        fig, ax = plt.subplots()
        ax.plot(S_values, delta_nll_no, label=plot_kwargs.get("label_no", "No shape systematic"))
        ax.plot(S_values, delta_nll_with, label=plot_kwargs.get("label_with", "With profiled shape systematic"))
        ax.axhline(level, linestyle="--", color="black", label=f"Delta NLL = {level}")
        ax.set_xlabel(plot_kwargs.get("xlabel", "Fixed signal yield S"))
        ax.set_ylabel(plot_kwargs.get("ylabel", r"$\Delta\mathrm{NLL}(S)$"))
        ax.set_title(plot_kwargs.get("title", "Profile likelihood scan in S"))
        ax.set_ylim(plot_kwargs.get("ylim", (0, 6)))
        ax.legend()
        plt.show()

        lo_no, hi_no = self.find_interval(S_values, delta_nll_no, level=level)
        lo_shape, hi_shape = self.find_interval(S_values, delta_nll_with, level=level)

        return {
            "no_systematic_shape": (lo_no, hi_no),
            "with_systematic_shape": (lo_shape, hi_shape),
        }

    def print_results(self):
        """Pretty-print the stored fit results.

        Requires that `run_both()` has been called.
        """
        if self.fit_no_shape is None or self.fit_with_shape is None:
            raise RuntimeError("Call run_both() before print_results().")

        fit_no = self.fit_no_shape
        fit_sh = self.fit_with_shape

        print("Fit WITHOUT background shape systematic")
        print("--------------------------------------")
        print(f"converged: {fit_no.success}")
        print(f"S_hat    = {fit_no.x[0]:.3f} events")
        print(f"B_hat    = {fit_no.x[1]:.3f} events")
        print(f"NLL_min  = {fit_no.fun:.3f}")
        print()

        print("Fit WITH profiled background shape systematic")
        print("---------------------------------------------")
        print(f"converged:  {fit_sh.success}")
        print(f"S_hat     = {fit_sh.x[0]:.3f} events")
        print(f"B_hat     = {fit_sh.x[1]:.3f} events")
        print(f"alpha_hat = {fit_sh.x[2]:.3f} sigma")
        print(f"NLL_min   = {fit_sh.fun:.3f}")

    def plot_results(self, title_no=None, title_with=None):
        """Create stacked fit plots for both stored fits and return components.

        Returns a tuple: ((sig_no, bkg_no, total_no), (sig_shape, bkg_shape, total_shape)).
        """
        if self.fit_no_shape is None or self.fit_with_shape is None:
            raise RuntimeError("Call run_both() before plot_results().")

        sig_no, bkg_no, total_no = helper.plot_fit_result(
            self.fit_no_shape,
            include_shape_systematic=False,
            title=title_no,
        )

        sig_shape, bkg_shape, total_shape = helper.plot_fit_result(
            self.fit_with_shape,
            include_shape_systematic=True,
            title=title_with,
        )

        return (sig_no, bkg_no, total_no), (sig_shape, bkg_shape, total_shape)
