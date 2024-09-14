import numpy as np
import numpy.typing as npt


def _xy_to_barycentric_coords(points: npt.NDArray[np.float32],
                              vertices: npt.NDArray[np.float32],
                              triangles: list[npt.NDArray[np.int32]]
                              ) -> tuple[list[tuple[tuple[np.int32, np.float32], tuple[np.int32, np.float32], tuple[np.int32, np.float32]]],
                                         npt.NDArray[np.bool]]:
    """
    Given and array containing xy locations and the vertices & triangles making up a mesh,
    find the triangle that each points in within and return it's representation using barycentric coordinates.
    points: ndarray [N,2] of point xy coords
    vertices: ndarray of vertex locations, row position is index id
    triangles: ndarraywith ordered vertex ids of vertices that make up each mesh triangle

    Is point inside triangle? : https://mathworld.wolfram.com/TriangleInterior.html

    Returns a list of barycentric coords for points inside the mesh,
    and a list of True/False values indicating whether a given pin was inside the mesh or not.
    Needed for removing pins during subsequent solve steps.

    """
    def det(u: npt.NDArray[np.float32], v: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """ helper function returns determinents of two [N,2] arrays"""
        ux, uy = u[:, 0], u[:, 1]
        vx, vy = v[:, 0], v[:, 1]
        return ux*vy - uy*vx

    tv_locs: npt.NDArray[np.float32] = np.asarray([vertices[t].flatten() for t in triangles])  # triangle->vertex locations, [T x 6] array

    v0 = tv_locs[:, :2]
    v1 = np.subtract(tv_locs[:, 2:4], v0)
    v2 = np.subtract(tv_locs[:, 4: ], v0)

    b_coords: list[tuple[tuple[np.int32, np.float32], tuple[np.int32, np.float32], tuple[np.int32, np.float32]]] = []
    pin_mask: list[bool] = []

    for p_xy in points:

        p_xy = np.expand_dims(p_xy, axis=0)
        a =  (det(p_xy, v2) - det(v0, v2)) / det(v1, v2)
        b = -(det(p_xy, v1) - det(v0, v1)) / det(v1, v2)

        # find the indices of triangle containing
        in_triangle = np.bitwise_and(np.bitwise_and(a > 0, b > 0), a + b < 1)
        containing_t_idxs = np.argwhere(in_triangle)

        # if length is zero, check if on triangle(s) perimeters
        if not len(containing_t_idxs):
            on_triangle_perimeter = np.bitwise_and(np.bitwise_and(a >= 0, b >= 0), a + b <= 1)
            containing_t_idxs = np.argwhere(on_triangle_perimeter)

        # point is outside mesh. Log a warning and continue
        if not len(containing_t_idxs):
            msg = f'point {p_xy} not inside or on edge of any triangle in mesh. Skipping it'
            print(msg)
            pin_mask.append(False)
            continue

        # grab the id of first triangle the point is in or on
        t_idx = int(containing_t_idxs[0])

        vertex_ids = triangles[t_idx]                               # get ids of verts in triangle
        a_xy, b_xy, c_xy = vertices[vertex_ids]                     # get xy coords of verts
        uvw = _get_barycentric_coords(p_xy, a_xy, b_xy, c_xy)  # get barycentric coords
        b_coords.append(list(zip(vertex_ids, uvw)))                 # append to our list  # pyright: ignore[reportGeneralTypeIssues]
        pin_mask.append(True)

    return (b_coords, np.array(pin_mask, dtype=np.bool))


def _get_barycentric_coords(p: npt.NDArray[np.float32],
                            a: npt.NDArray[np.float32],
                            b: npt.NDArray[np.float32],
                            c: npt.NDArray[np.float32]
                            ) -> npt.NDArray[np.float32]:
    """
    As described in Christer Ericson's Real-Time Collision Detection.
    p: the input point
    a, b, c: the vertices of the triangle

    Returns ndarray [u, v, w], the barycentric coordinates of p wrt vertices a, b, c
    """
    v0: npt.NDArray[np.float32] = np.subtract(b, a)
    v1: npt.NDArray[np.float32] = np.subtract(c, a)
    v2: npt.NDArray[np.float32] = np.subtract(p, a)
    d00: np.float32 = np.dot(v0, v0)
    d01: np.float32 = np.dot(v0, v1)
    d11: np.float32 = np.dot(v1, v1)
    d20: np.float32 = np.dot(v2, v0)
    d21: np.float32 = np.dot(v2, v1)
    denom = d00 * d11 - d01 * d01
    v: npt.NDArray[np.float32] = (d11 * d20 - d01 * d21) / denom  # pyright: ignore[reportGeneralTypeIssues]
    w: npt.NDArray[np.float32] = (d00 * d21 - d01 * d20) / denom  # pyright: ignore[reportGeneralTypeIssues]
    u: npt.NDArray[np.float32] = 1.0 - v - w

    return np.array([u, v, w]).squeeze()


def get_offsets_from_ma(ma_fpath, parents2d, parents2d_to_kps, names, dim=2):
    all_offsets = None

    with (
        open(ma_fpath) as fin,
    ):
        for line in fin:
            if "animCurveTL" not in line:
                continue

            *_, joint_name = line.split()
            *joint_name, axe = joint_name.split("_")
            axe = axe.strip('";')[-1]
            joint_name = " ".join(joint_name).strip('"')
            line = fin.readline()
            line = fin.readline()
            line = fin.readline()
            line = fin.readline()
            #joint_index = KPS.index(joint_name)
            i = names.index(joint_name)
            seq = line.split()[4:]
            seq = list(map(lambda x: float(x.strip(";")), seq))
            if all_offsets is None:
                all_offsets = np.zeros((len(seq) // 2, len(parents2d), dim), dtype="float64")

            for _j, ai in enumerate(range(0, len(seq), 2)):
                #j = seq[ai] - 1
                c = seq[ai + 1]
                #print(joint_name, 1, x0, y0, 2, x1, y1)
                if axe == "X":
                    all_offsets[_j, i, 0] = c
                    #offsets_b[i, 0] = b
                elif axe == "Y":
                    all_offsets[_j, i, 1] = c
                    #offsets_b[i, 1] = b

    assert all_offsets is not None

    return all_offsets
