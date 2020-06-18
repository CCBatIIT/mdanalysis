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
r"""Base class for partial models --- :mod:`MDAnalysis.analysis.tem.partial_models.base`
===========================================================================

:Author: David Minh
:Year: 2020
:Copyright: GNU Public License, v2 or any higher version

.. versionadded:: N/A

This module contains a base class for modelling a subset of the degrees of
freedom.

"""

import numpy as np
tau = 2 * np.pi


class PartialModelBase:
    """Models a subset of the degrees of freedom

    """
    def __init__(self, X):
        """Parameters
        ----------
        X : numpy.ndarray
            an array of coordinates with dimensions (N, K), where N is the
            number of samples and K is the number of degrees of freedom
        """
        self.lnZ = None

    def rvs(self, N):
        """Generate random samples

        Parameters
        ----------
        N : int
            number of samples to generate

        Returns
        -------
        X : numpy.ndarray
            an array of coordinates with dimensions (N, K), where N is the
            number of samples and K is the number of degrees of freedom

        """
        raise NotImplementedError

    def logpdf(self, X):
        """Calculate the log probability density

        Parameters
        ----------
        X : numpy.ndarray
            an array of coordinates with dimensions (N, K), where N is the
            number of samples and K is the number of degrees of freedom

        Returns
        -------
        logpdf : numpy.ndarray
            an array with dimensions (N,), with the log probability density

        """
        raise NotImplementedError
