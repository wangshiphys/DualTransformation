"""
Systematic construction of dual transformation for J-K-G0-G1 model on
triangle and/or honeycomb lattice.

By a dual transformation we mean a prescription for site-dependent rotations
in the spin space which transforms a spin Hamiltonian $H(S)$ in to a formally
new Hamiltonian $H^'(S^')$. We are interested in self-dual transformations
that map the model onto itself, preserving all its symmetry properties. That
is, the rotated partner has the same terms albeit with different parameters
J'-K'-G0'-G1' and it respects the C3 rotation rules hence preserving the
original distribution of the three types of bond-dependent interactions on a
lattice.

See also:
    J. Chaloupka and G. Khaliullin, Phys. Rev. B 92, 024413 (2015).
"""


__all__ = [
    "Searcher",
]


import matplotlib.pyplot as plt
import numpy as np

from HamiltonianPy import lattice_generator, SiteID


class Searcher:
    """
    Implementation of the algorithm for a systematic search for the dual
    transformations.
    """

    def __init__(self, which, num0=10, num1=10):
        """
        Customize the newly created instance.

        Parameters
        ----------
        which : ["triangle" or "honeycomb"]
            On which lattice the J-K-G0-G1 model is defined.
        num0, num1 : int, optional
            The number of unit-cell along the 1st and 2nd translation vector.
            Default: 10.
        """

        assert which in ("triangle", "honeycomb"), "Invalid `which` parameter!"
        assert isinstance(num0, int) and num0 > 0, "Invalid `num0` parameter!"
        assert isinstance(num1, int) and num1 > 0, "Invalid `num1` parameter!"

        cluster = lattice_generator(which, num0=num0, num1=num1)
        bonds, trash = cluster.bonds(nth=1)

        all_bonds = {}
        all_azimuths = set()
        for bond in bonds:
            azimuth = bond.getAzimuth(ndigits=0)
            # 180 degree is equivalent to -180 degree
            # Make all translation equivalent bonds pointing the same direction
            if azimuth < 0 or azimuth == 180:
                bond = bond.flip()
                azimuth = bond.getAzimuth(ndigits=0)
            # Only distinct azimuths are saved
            all_azimuths.add(azimuth)
            if azimuth in all_bonds:
                all_bonds[azimuth].append((azimuth, bond))
            else:
                all_bonds[azimuth] = [(azimuth, bond)]

        # For triangle and honeycomb lattice, there are only three distinct
        # azimuths. However, numerical errors may cause different results.
        if len(all_azimuths) != 3:
            raise RuntimeError("Wrong azimuths: {0}".format(all_azimuths))

        self._all_bonds = all_bonds
        self._all_azimuths = tuple(sorted(all_azimuths))
        # First tuple is row indices and second tuple is column indices
        self._JTermIndices = {
            "x": ((1, 2), (1, 2)),
            "y": ((0, 2), (0, 2)),
            "z": ((0, 1), (0, 1)),
        }
        self._KTermIndices = {
            "x": ((0, ), (0, )),
            "y": ((1, ), (1, )),
            "z": ((2, ), (2, )),
        }
        self._G0TermIndices = {
            "x": ((1, 2), (2, 1)),
            "y": ((0, 2), (2, 0)),
            "z": ((0, 1), (1, 0)),
        }
        self._G1TermIndices = {
            "x": ((0, 0, 1, 2), (1, 2, 0, 0)),
            "y": ((0, 1, 1, 2), (1, 0, 2, 1)),
            "z": ((0, 1, 2, 2), (2, 2, 0, 1)),
        }

    # Extract new model parameters from the given `hij` if `hij` is of the
    # right form.
    def _new_parameters(self, hij, which, rtol=1e-05, atol=1e-08):
        Js = hij[self._JTermIndices[which]]
        G0s = hij[self._G0TermIndices[which]]
        G1s = hij[self._G1TermIndices[which]]
        judge0 = np.allclose(Js[0], Js[1], rtol=rtol, atol=atol)
        judge1 = np.allclose(G0s[0], G0s[1], rtol=rtol, atol=atol)
        judge2 = np.allclose(G1s[0], G1s[1:], rtol=rtol, atol=atol)
        if judge0 and judge1 and judge2:
            J = Js[0]
            K = hij[self._KTermIndices[which]][0] - J
            G0 = G0s[0]
            G1 = G1s[0]
            return J, K, G0, G1
        else:
            return None

    def __call__(
            self, R0, R1,
            J=None, K=None, G0=None, G1=None, rtol=1e-05, atol=1e-08
    ):
        """
        Search for the dual transformation starting from the R0 and R1.

        Parameters
        ----------
        R0, R1 : (3, 3) array
            The starting rotations on an arbitrary nearest-neighbor bond.
        J, K, G0, G1 : int or float or None, optional
            The coefficients of the Heisenberg, Kitaev, Gamma and GammaPrime
            terms.If set to None, then the coefficient is generated randomly
            in the range [1, 2).
            Default: None.
        rtol : float, optional
            The relative tolerance parameter.
            Default: 1e-05.
        atol : float, optional
            The absolute tolerance parameters.
            Default: 1e-08.

        Returns
        -------
        res : None or list
            The corresponding dual-transformation.
        """

        J = np.random.random() + 1 if J is None else J
        K = np.random.random() + 1 if K is None else K
        G0 = np.random.random() + 1 if G0 is None else G0
        G1 = np.random.random() + 1 if G1 is None else G1

        hijx = np.array([[J + K, G1, G1], [G1, J, G0], [G1, G0, J]])
        hijy = np.array([[J, G1, G0], [G1, J + K, G1], [G0, G1, J]])
        hijz = np.array([[J, G0, G1], [G0, J, G1], [G1, G1, J + K]])
        hijs = dict(zip(self._all_azimuths, [hijx, hijy, hijz]))
        azimuth_to_bond_type = dict(zip(self._all_azimuths, ["x", "y", "z"]))

        for azimuth in self._all_azimuths:
            which = azimuth_to_bond_type[azimuth]
            hij_tmp = np.linalg.multi_dot([R0, hijs[azimuth], R1.T])
            new_parameters = self._new_parameters(
                hij_tmp, which, rtol=rtol/100, atol=atol/100
            )
            if new_parameters is not None:
                JNew, KNew, G0New, G1New = new_parameters
                break
        else:
            print("The given R0 and R1 cannot generate a dual transformation!")
            return None

        azimuth, bond = self._all_bonds[azimuth][0]
        p0, p1 = bond.endpoints
        rotation_records = {SiteID(p0): R0, SiteID(p1): R1}

        hijx_new = np.array(
            [
                [JNew + KNew, G1New, G1New],
                [G1New, JNew, G0New],
                [G1New, G0New, JNew],
            ]
        )
        hijy_new = np.array(
            [
                [JNew, G1New, G0New],
                [G1New, JNew + KNew, G1New],
                [G0New, G1New, JNew],
            ]
        )
        hijz_new = np.array(
            [
                [JNew, G0New, G1New],
                [G0New, JNew, G1New],
                [G1New, G1New, JNew + KNew],
            ]
        )
        hijs_new = dict(zip(self._all_azimuths, [hijx_new, hijy_new, hijz_new]))

        all_bonds = set()
        for azimuth in self._all_azimuths:
            all_bonds.update(self._all_bonds[azimuth])

        while all_bonds:
            azimuth, bond = all_bonds.pop()
            hij = hijs[azimuth]
            hij_new = hijs_new[azimuth]

            p0, p1 = bond.endpoints
            id0 = SiteID(p0)
            id1 = SiteID(p1)
            if (id0 not in rotation_records) and (id1 not in rotation_records):
                all_bonds.add((azimuth, bond))
            elif (id0 not in rotation_records) and (id1 in rotation_records):
                R1 = rotation_records[id1]
                R0 = np.dot(hij_new, np.linalg.inv(np.dot(hij, R1.T)))
                rotation_records[id0] = R0
            elif (id0 in rotation_records) and (id1 not in rotation_records):
                R0 = rotation_records[id0]
                R1 = np.dot(np.linalg.inv(np.dot(R0, hij)), hij_new).T
                rotation_records[id1] = R1
            else:
                R0 = rotation_records[id0]
                R1 = rotation_records[id1]
                hij_tmp = np.linalg.multi_dot([R0, hij, R1.T])
                match = np.allclose(hij_tmp, hij_new, rtol=rtol, atol=atol)
                if not match:
                    print("Rotated hij does not match!")
                    return None

        site_id, R = rotation_records.popitem()
        containers = [[R, site_id.site]]
        while rotation_records:
            site_id, R = rotation_records.popitem()
            for container in containers:
                if np.allclose(R, container[0], rtol=rtol, atol=atol):
                    container.append(site_id.site)
                    break
            else:
                containers.append([R, site_id.site])
        return containers


if __name__ == "__main__":
    from HamiltonianPy.rotation3d import *
    from itertools import product

    solver = Searcher("triangle")
    Rs = [
        E, INVERSION,
        RX30, RX45, RX60, RX90, RX180, RX270,
        RY30, RY45, RY60, RY90, RY180, RY270,
        RZ30, RZ45, RZ60, RZ90, RZ180, RZ270,
        R111_60, R111_120, R111_180, R111_240, R111_300,
        AP0, AP1, AP2,
    ]
    TMP = [E, INVERSION]


    for A, B, C, D in product(TMP, Rs, TMP, Rs):
        R0 = np.dot(A, B)
        R1 = np.dot(C, D)
        rotations = solver(
            R0, R1,
            # J=1, K=0, G0=0, G1=0,
            rtol=1e-2, atol=1e-4,
        )

        if rotations is not None:
            fig, ax = plt.subplots()
            ax.set_axis_off()
            ax.set_aspect("equal")
            lines = []
            labels = []
            for index, sub_rotations in enumerate(rotations):
                print("SiteIndex={0}".format(index))
                for row in sub_rotations[0]:
                    print(row)
                sites = np.stack(sub_rotations[1:])
                line, = ax.plot(sites[:, 0], sites[:, 1], marker="o", ls="")
                lines.append(line)
                labels.append("SiteIndex={0}".format(index))
            print("=" * 80)
            ax.legend(lines, labels, loc="upper left", fontsize="xx-large")
            plt.get_current_fig_manager().window.showMaximized()
            plt.show()
            plt.close("all")
