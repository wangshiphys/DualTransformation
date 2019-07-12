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
    "DualTransformationForBond",
    "DualTransformationForCluster",
]


from itertools import combinations, product
from time import strftime

import matplotlib.pyplot as plt
import numpy as np
from HamiltonianPy import lattice_generator, SiteID

from core import *


def DualTransformationForBond(
        which, axes, thetas,
        J=None, K=None, G0=None, G1=None, rtol=1e-05, atol=1e-08
):
    """
    Find all dual-transformations for a specific bond from the allowed
    rotations.

    All dual-transformations are write to a text file with six column.
    The 1st, 2nd abd 3rd columns correspond to alpha0, beta0, theta0 and the
    remaining columns correspond to alpha1, beta1 and theta1.

    Parameters
    ----------
    which : ["x", "y" or "z"]
        The type of the Bond.
    axes : sequences
        The allowed rotation axes.
    thetas : sequence
        The allowed rotation angles.
    J, K, G0, G1 : int or float or None, optional
        The coefficients of the Heisenberg, Kitaev, Gamma and GammaPrime terms.
        If set to None, then the coefficient is integer generated randomly
        in the range [1, 10).
        Default: None.
    rtol : float, optional
        The relative tolerance parameter.
        Default: 1e-05.
    atol : float, optional
        The absolute tolerance parameters.
        Default: 1e-08.
    """
    J = np.random.randint(1, 10) if J is None else J
    K = np.random.randint(1, 10) if K is None else K
    G0 = np.random.randint(1, 10) if G0 is None else G0
    G1 = np.random.randint(1, 10) if G1 is None else G1

    if which == "x":
        hij = np.array(
            [[J + K, G1, G1], [G1, J, G0], [G1, G0, J]], dtype=np.float64
        )
        ValidRotation = ValidRotationForXBond
        file_name = "ValidRotationForXBond"
    elif which == "y":
        hij = np.array(
            [[J, G1, G0], [G1, J + K, G1], [G0, G1, J]], dtype=np.float64
        )
        ValidRotation = ValidRotationForYBond
        file_name = "ValidRotationForYBond"
    elif which == "z":
        hij = np.array(
            [[J, G0, G1], [G0, J, G1], [G1, G1, J + K]], dtype=np.float64
        )
        ValidRotation = ValidRotationForZBond
        file_name = "ValidRotationForZBond"
    else:
        raise ValueError("Invalid `which` parameter: {0}".format(which))
    file_name += " with J={0},K={1},G0={2},G1={3}.txt".format(J, K, G0, G1)

    with open(file_name, "w", buffering=1) as fp:
        time_fmt = "%Y-%m-%d %H:%M:%S"
        fp.write("# Start running at: {0}\n".format(strftime(time_fmt)))
        fp.write("# alpha0,   beta0,  theta0,  alpha1,   beta1,  theta1\n")
        msg = "{0:>8.2f},{1:>8.2f},{2:>8.2f},{3:>8.2f},{4:>8.2f},{5:>8.2f}\n"

        for r0, r1 in combinations(product(axes, thetas), r=2):
            (alpha0, beta0), theta0 = r0
            (alpha1, beta1), theta1 = r1
            R0 = Rotation(alpha0 * np.pi, beta0 * np.pi, theta0 * np.pi)
            R1 = Rotation(alpha1 * np.pi, beta1 * np.pi, theta1 * np.pi)
            if ValidRotation(R0, R1, hij, rtol=rtol, atol=atol):
                fp.write(
                    msg.format(alpha0, beta0, theta0, alpha1, beta1, theta1)
                )
        fp.write("# Stop running at: {0}\n".format(strftime(time_fmt)))


class DualTransformationForCluster:
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
            terms.If set to None, then the coefficient is integer generated
            randomly in the range [1, 10).
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

        J = np.random.randint(1, 10) if J is None else J
        K = np.random.randint(1, 10) if K is None else K
        G0 = np.random.randint(1, 10) if G0 is None else G0
        G1 = np.random.randint(1, 10) if G1 is None else G1
        hijx = np.array([[J + K, G1, G1], [G1, J, G0], [G1, G0, J]], np.float64)
        hijy = np.array([[J, G1, G0], [G1, J + K, G1], [G0, G1, J]], np.float64)
        hijz = np.array([[J, G0, G1], [G0, J, G1], [G1, G1, J + K]], np.float64)
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
            ], dtype=np.float64
        )
        hijy_new = np.array(
            [
                [JNew, G1New, G0New],
                [G1New, JNew + KNew, G1New],
                [G0New, G1New, JNew],
            ], dtype=np.float64
        )
        hijz_new = np.array(
            [
                [JNew, G0New, G1New],
                [G0New, JNew, G1New],
                [G1New, G1New, JNew + KNew],
            ], dtype=np.float64
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
    from pathlib import Path

    solver = DualTransformationForCluster("triangle")

    J = 1
    K = G0 = G1 = 0
    rotations_name = "ValidRotationX with J={0},K={1},G0={2},G1={3}.txt".format(
        J, K, G0, G1
    )
    fig_name = "alpha0={0:.2f},beta0={1:.2f},theta0={2:.2f}"
    fig_name += "alpha1={3:.2f},beta1={4:.2f},theta1={5:.2f}.jpg"
    fig_path_template = "{0}-sublattice-transformation/"

    bond_rotations = np.loadtxt(rotations_name, comments="#", delimiter=",")
    for r in bond_rotations:
        R0 = Rotation(r[0] * np.pi, r[1] * np.pi, r[2] * np.pi)
        R1 = Rotation(r[3] * np.pi, r[4] * np.pi, r[5] * np.pi)
        cluster_rotations = solver(
            R0, R1, J=J, K=K, G0=G0, G1=G1, rtol=1e-2, atol=1e-4
        )

        if cluster_rotations is not None:
            fig_path = fig_path_template.format(len(cluster_rotations))
            Path(fig_path).mkdir(exist_ok=True, parents=True)

            fig, ax = plt.subplots()
            ax.set_axis_off()
            ax.set_aspect("equal")
            lines = []
            labels = []
            for index, sub_rotations in enumerate(cluster_rotations):
                # print("SiteIndex={0}".format(index))
                # for row in sub_rotations[0]:
                #     print(row)
                sites = np.stack(sub_rotations[1:])
                line, = ax.plot(
                    sites[:, 0], sites[:, 1], marker="o", ls="", ms=30
                )
                lines.append(line)
                labels.append("SiteIndex={0}".format(index))
            # print("=" * 80)
            ax.legend(lines, labels, loc="upper left", fontsize="xx-large")
            fig.set_size_inches(19.2, 9.3)
            fig.savefig(fig_path + fig_name.format(*r))
            # plt.get_current_fig_manager().window.showMaximized()
            # plt.show()
            plt.close("all")
