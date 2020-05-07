# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
# MDAnalysis --- https://www.mdanalysis.org
# Copyright (c) 2006-2017 The MDAnalysis Development Team and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v21 or any higher version
#
# Please cite your use of MDAnalysis in published work:
#
# R. J. Gowers, M. Linke, J. Barnoud, T. J. E. Reddy, M. N. Melo, S. L. Seyler,
# D. L. Dotson, J. Domanski, S. Buchoux, I. M. Kenney, and O. Beckstein.
# MDAnalysis: A Python package for the rapid analysis of molecular dynamics
# simulations. In S. Benthall and S. Rostrup editors, Proceedings of the 15th
# Python in Science Conference, pages 102-109, Austin, TX, 2016. SciPy.
# doi: 10.25080/majora-629e541a-00e
#
# N. Michaud-Agrawal, E. J. Denning, T. B. Woolf, and O. Beckstein.
# MDAnalysis: A Toolkit for the Analysis of Molecular Dynamics Simulations.
# J. Comput. Chem. 32 (2011), 2319--2327, doi:10.1002/jcc.21787
#
r"""Model for external degrees of freedom --- :mod:`MDAnalysis.analysis.encore.partial_models.external`
===========================================================================

:Author: David Minh
:Year: 2020
:Copyright: GNU Public License, v2 or any higher version

.. versionadded:: N/A

This module contains two classes for modelling external degrees of freedom
with a nonparameteric density estimate, as described in [Menzer2018]_.
Translational degrees of freedom are modelled by a Gaussian KDE in principal
components space. Rotational degrees of freedom are modelled by using a FFT
to convolute the histogram with a Gaussian kernel.


References
----------

.. [Menzer2018] Menzer, William, Chen Li, Wenji Sun, Bing Xie, and David D L
   Minh. “Simple Entropy Terms for End-Point Binding Free Energy Calculations.”
   *Journal of Chemical Theory and Computation* 14: 6035–49.
   doi:`10.1021/acs.jctc.8b00418<https://doi.org/10.1021/acs.jctc.8b00418>`_

"""

import numpy as np
from scipy.stats import gaussian_kde
from scipy.stats import rv_discrete

from .base import PartialModelBase
from .gaussiankde import PrincipalComponentsKDE

class Translational(PrincipalComponentsKDE):
    """Models translational degrees of freedom

    Translational degrees of freedom are modelled by a Gaussian KDE on a
    principal components projection. The procedure is described in [Menzer2018]_.

    """
    def __init__(self, X):
        """Parameters
        ----------
        X : np.array
            The translational degrees of freedom.
            An array with dimensions (N, 3), where N is the number of samples.
        """
        super(Translational, self).__init__(X, coordinate_type='translation')

        # Log volume of the binding site
        box = [(self._edges[dim][-1] - self._edges[dim][0]) \
            for dim in range(3)]
        self.lnV_site = np.sum([np.log(box) for dim in range(3)])
        # Standard state correction for confining the system into a box
        # The standard state volume for a single molecule
        # in a box of size 1 L is 1.66053928 nanometers**3
        self.DeltaG_xi = -self.lnV_site + np.log(1660.53928)

class Rotational(PartialModelBase):
    """Models rotational degrees of freedom

    Rotational degrees of freedom are modelled by a discretized distribution.
    The FFT is used to convolute the histogram with a Gaussian kernel.
    The procedure is described in [Menzer2018]_.

    """
    def __init__(self, rot, nbins=100):
        """Parameters
        ----------
        rot : np.array
            The rotational degrees of freedom.
            An array with dimensions (N, 3), where N is the number of samples.
        nbins : int
            The number of bins in the histogram.
        """
        edges_2pi = np.linspace(-np.pi, np.pi, nbins)
        edges_pi = np.linspace(0, np.pi, nbins)

        scotts_factor = np.power(rot.shape[0], (-1. / 5))

        # Define parameters and estimate log partition function
        self._rv_discrete = []
        self._edges = []
        self._centers = []
        self._delta = []

        self.lnZ = 0
        for dim in range(3):
            name = {0: 'phi', 1: 'theta', 2: 'omega'}[dim]
            if dim in [0, 2]:
                edges = edges_2pi
            else:
                edges = edges_pi
            delta = edges[1] - edges[0]
            centers = edges[:-1] + delta / 2

            H = np.histogram(rot[:, dim], edges, density=True)[0]
            # Gaussian kernel
            sigma = scotts_factor * 10. * delta
            ker = np.exp(-(centers-centers[int(len(centers)/2)])**2/\
                         2/sigma**2)/sigma/np.sqrt(2*np.pi)
            # Convolution
            rho = np.abs(np.fft.fftshift(np.fft.ifft(\
                         np.fft.fft(H)*np.fft.fft(ker))).real)
            # For rv_discrete, the sum of probabilities is required to be one
            rho /= np.sum(rho)
            # Initiat discretized random variable
            rv = rv_discrete(name=name, values=(range(len(rho)), rho))
            # Because a numerical integral is np.sum(rho*delta),
            # probabilities from rv_discrete will be divided by delta
            # so that they are normalized

            # These are numerical integrals of $I(\xi)J(\xi)$
            # over rotational dofs
            if dim in [0, 2]:
                # $Z = \sum_i^n_b \rho(x_i) \delta = 1$
                # \sum_i \delta = n_b \delta = 2 \pi
                self.lnZ += 0.
            else:
                # The second Euler angle has a Jacobian
                # $Z = \sum_i^n_b \rho(x_i) sin(x_i) \delta = 1$
                # \sum_i^n_b sin(x_i) \delta = 2$
                self.lnZ += np.log(np.sum(np.sin(centers)*rho/delta))

            # There is no lnZ_J because the Jacobian is not assumed to be
            # a constant, but part of the integral over theta
            self._rv_discrete.append(rv)
            self._edges.append(edges)
            self._centers.append(centers)
            self._delta.append(delta)

    def rvs(self, N):
        """Generate random samples

        Parameters
        ----------
        N : int
            number of samples to generate

        Returns
        -------
        X : np.array
            an array of coordinates with dimensions (N, 3), where N is the
            number of samples. Samples are at the bin centers.

        """
        samples = [self._centers[dim][self._rv_discrete[dim].rvs(size=N)] \
            for dim in range(3)]
        return np.array(samples).transpose()

    def logpdf(self, X):
        """Calculate the log probability density

        Parameters
        ----------
        X : np.array
            an array of coordinates with dimensions (N, 3), where N is the
            number of samples

        Returns
        -------
        logpdf : np.array
            an array with dimensions (N,), with the log probability density.
            The density is approximated as constant within the bin.

        """
        return np.sum([\
            self._rv_discrete[dim].logpmf(\
                  np.floor((X[:,dim] - self._edges[dim][0])/self._delta[dim])
              ) - np.log(self._delta[dim]) for dim in range(3)],0)
