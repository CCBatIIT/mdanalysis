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
r"""Model for external degrees of freedom --- :mod:`MDAnalysis.analysis.tem.partial_models.external`
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

from MDAnalysis.analysis.encore.covariance import shrinkage_covariance_estimator

class Translational(PrincipalComponentsKDE):
    """Models translational degrees of freedom

    Translational degrees of freedom are modelled by a Gaussian KDE on a
    principal components projection. The procedure is described in [Menzer2018]_.

    """
    # _param_keys inherits from PrincipalComponentsKDE
    _allowed_coordinate_types = ['translation']

class Rotational(PartialModelBase):
    """Models rotational degrees of freedom

    Rotational degrees of freedom are modelled by a discretized distribution.
    The FFT is used to convolute the histogram with a Gaussian kernel.
    The procedure is described in [Menzer2018]_.

    """
    _param_keys = ['rho', 'edges']
    _allowed_coordinate_types = ['rotation']

    def __init__(self, rho, edges):
        """Parameters
        ----------
        rho : numpy.ndarray
            the normalized density, with (K, nbins) dimensions
        edges : numpy.ndarray
            bin edges, with (K, nbins+1) dimensions
        """
        super(Rotational, self).__init__('rotation')

        delta = [e[1]-e[0] for e in edges]
        centers = [edges[dim, :-1] + delta[dim] / 2 for dim in range(3)]

        rv = []
        for dim in range(3):
            name = {0: 'phi', 1: 'theta', 2: 'omega'}[dim]
            rv.append(rv_discrete(name=name, \
                values=(range(rho.shape[1]), rho[dim])))

        lnZ = 0.
        for dim in range(3):
            # Because a numerical integral is np.sum(rho*delta),
            # probabilities from rv_discrete will be divided by delta
            # so that they are normalized

            # These are numerical integrals of $I(\xi)J(\xi)$
            # over rotational dofs
            if dim in [0, 2]:
                # $Z = \sum_i^n_b \rho(x_i) \delta = 1$
                # \sum_i \delta = n_b \delta = 2 \pi
                lnZ += 0.
            else:
                # The second Euler angle has a Jacobian
                # $Z = \sum_i^n_b \rho(x_i) sin(x_i) \delta = 1$
                # \sum_i^n_b sin(x_i) \delta = 2$
                lnZ += np.log(np.sum(np.sin(centers)*rho[dim]/delta[dim]))
            # There is no lnZ_J because the Jacobian is not assumed to be
            # a constant, but part of the integral over theta

        self._rho = rho
        self._edges = edges
        self._centers = centers
        self._delta = delta
        self._rv_discrete = rv
        self.lnZ = lnZ

    @classmethod
    def from_data(cls, rot, nbins=1000):
        """Parameters
        ----------
        rot : numpy.ndarray
            The rotational degrees of freedom.
            An array with dimensions (N, 3), where N is the number of samples.
        """
        scotts_factor = np.power(rot.shape[0], (-1. / 5))

        # Define parameters and estimate log partition function
        rho = []
        edges = []
        for dim in range(3):
            if dim in [0, 2]:
                e = np.linspace(-np.pi, np.pi, nbins)
            else:
                e = np.linspace(0, np.pi, nbins)
            delta = e[1] - e[0]
            centers = e[:-1] + delta / 2

            H = np.histogram(rot[:, dim], e, density=True)[0]
            # Gaussian kernel
            sigma = scotts_factor * 10. * (delta)
            ker = np.exp(-(centers-centers[int(len(centers)/2)])**2/\
                         2/sigma**2)/sigma/np.sqrt(2*np.pi)
            # Convolution
            r = np.abs(np.fft.fftshift(np.fft.ifft(\
                         np.fft.fft(H)*np.fft.fft(ker))).real)
            # For rv_discrete, the sum of probabilities
            # is required to be one
            r /= np.sum(r)

            rho.append(r)
            edges.append(e)

        return cls(np.array(rho), np.array(edges))

    def rvs(self, N):
        """Generate random samples

        Parameters
        ----------
        N : int
            number of samples to generate

        Returns
        -------
        X : numpy.ndarray
            an array of coordinates with dimensions (N, 3), where N is the
            number of samples. Samples are at the bin centers.

        """
        samples = [self._centers[dim][self._rv_discrete[dim].rvs(size=N)] \
            for dim in range(3)]
        return numpy.ndarray(samples).transpose()

    def logpdf(self, X):
        """Calculate the log probability density

        Parameters
        ----------
        X : numpy.ndarray
            an array of coordinates with dimensions (N, 3), where N is the
            number of samples

        Returns
        -------
        logpdf : numpy.ndarray
            an array with dimensions (N,), with the log probability density.
            The density is approximated as constant within the bin.

        """
        return np.sum([\
            self._rv_discrete[dim].logpmf(\
                  np.floor((X[:,dim] - self._edges[dim][0])/self._delta[dim])
              ) - np.log(self._delta[dim]) for dim in range(3)],0)
