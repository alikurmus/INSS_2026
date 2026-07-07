"""Fit manager class wrapping the helper fit functions for notebook use.

This small class centralizes running the two fit modes, printing
results, and producing stacked-fit plots, while keeping the low-level
service functions in `app.helper`.
"""
from __future__ import annotations

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
