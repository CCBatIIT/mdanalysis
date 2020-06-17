# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
# MDAnalysis --- https://www.mdanalysis.org
# Copyright (c) 2006-2017 The MDAnalysis Development Team and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version
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
r"""Bond-Angle-Torsion coordinates analysis --- :mod:`MDAnalysis.analysis.bat`
===========================================================================

:Author: David Minh
:Year: 2020
:Copyright: GNU Public License, v2 or any higher version

.. versionadded:: N/A

This module contains classes for computing the free energy difference between
a pair of thermodynamic states using readily integrable models of the
probability distributions. The thermodynamic cycle for the free energy
difference between states A and B includes A --> A_m --> B_m --> B,
where A_m and B_m are the states corresponding to modeled distributions of
states A and B, respectively. States A and B must have the same number of
coordinates and differ by the potential energy function. States A_m and B_m
are models for which the normalizing constant/absolute configurational integral
are readily available. Model distributions for the torsions could include
independent Gaussians, independent Gaussian kernel density estimates,
multivariate Gaussians, and Gaussian mixtures.

The method is under development and has not been published. If you make use of
it before publication, please cite this github repository. Similar methods are
described in [Ytreberg2006]_ and [Grigoryan2013]_.


References
----------

 .. [Grigoryan2013] Grigoryan, Gevorg. “Absolute Free Energies of Biomolecules
    from Unperturbed Ensembles.” *Journal of Computational Chemistry* 34(31):
    2726–41. doi:`10.1002/jcc.23448<https://doi.org/10.1002/jcc.23448>.`

 .. [Ytrebreg2006] Ytreberg, F Marty, and Daniel M Zuckerman. “Simple
    Estimation of Absolute Free Energies for Biomolecules.” *Journal of
    Chemical Physics* 124(10): 104105.
    doi:`10.1063/1.2174008<https://doi.org/10.1063/1.2174008>`

"""

import pymbar

def FEP_and_BAR(w_F, w_R=None):
    """Calculates free energy differences based on work

    The free energy difference between states A and B is estimated
    based on the uni- or bi-directional work of transitions between them.

    Parameters
    ----------
    w_F : np.array
        The work of going from A to B
    w_R : np.array
        The work of going from B to A

    Returns
    -------
    f_EXP_F : float
        Unidirectional free energy estimate based on w_F. If w_F is None,
        None will be returned.
    f_EXP_R : float
        Unidirectional free energy estimate based on w_R. If w_R is None,
        None will be returned.
    f_BAR : float
        Bidirectional free energy estimate based on w_F and w_R. If either
        w_F or w_R is None, None will be returned.
    """
    f_EXP_F = None
    f_EXP_R = None
    f_BAR = None

    if w_F is not None:
        min_w_F = min(w_F)
        f_EXP_F = -np.log(np.mean(np.exp(-w_F+min_w_F))) + min_w_F

    if w_R is not None:
        min_w_R = min(w_R)
        f_EXP_R = np.log(np.mean(np.exp(-w_R+min_w_R))) - min_w_R

    if (w_F is not None) and (w_R is not None):
        f_BAR = pymbar.BAR(w_F, w_R, \
            relative_tolerance=1.0E-5, \
            verbose=False, \
            compute_uncertainty=False)

    return (f_EXP_F, f_EXP_R, f_BAR)


class FreeEnergy():
    """Class to evaluate the free energy difference using ensemble models

    """
    def __init__(self, A_m, B_m):
        r"""Parameters
        ----------
        A_m : class in encore.bat_models
            Ensemble model for state A
        B_m : class in encore.bat_models
            Ensemble model for state B
        """
        self._A_m = A_m
        self._B_m = B_m

        self.free_energies = {}
        self.free_energies['A_mB_m'] = - self._B_m.lnZ['total'] \
                                       + self._A_m.lnZ['total']

    def Forward(self):
        r"""Free energy estimates based on forward work

        Forward work means going from the MM force field into the ensemble model

        """
        for enmodel_name in ['A_m','B_m']:
            enmodel = getattr(self,'_'+enmodel_name)
            # w_F = u_f(x) - u_i(x) = -logpdf_f(x) + logpdf_i(x)
            if hasattr(enmodel,'logpdf_MM'):
                w_F = -enmodel.logpdf() + enmodel.logpdf_MM
                (f_EXP_F, ~, ~) = FEP_and_BAR(w_F, None)
            else:
                raise ValueError('Log probability in MM force field unavailable')
            self.free_energies[enmodel_name[0]+enmodel_name] = f_EXP_F
        self.free_energies['total_F'] = \
              self.free_energies['AA_m'] \
            + self.free_energies['A_mB_m'] \
            - self.free_energies['BB_m']

    def Reverse(self):
        r"""Free energy estimates based on reverse work

        Reverse work means going from the ensemble model into the MM force field

        """
        raise NotImplementedError

    def Bidirectional(self):
        r"""Free energy estimates based on forward and reverse work

        """
        raise NotImplementedError
