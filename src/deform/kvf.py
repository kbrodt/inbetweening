import numpy as np
import scipy.io
import scipy.sparse as sps
import trimesh


class KVF:
    def __init__(self, vertices, triangles, skeleton_inds):
        self.vertices = vertices.copy()
        self.triangles = triangles.copy()
        self.skeleton_inds = skeleton_inds.copy()

        self.inds = np.array(sorted(set(range(len(self.vertices))) - set(self.skeleton_inds)))

        mesh = trimesh.Trimesh(
            np.pad(self.vertices, ((0, 0), (0, 1))),
            self.triangles,
            process=False,
            validate=False,
        )
        neighbours = mesh.vertex_neighbors
        ninds, degrees, idxs = ninds_degrees_idxs(neighbours)
        self.ninds = ninds
        self.degrees = degrees
        self.idxs = idxs

    def solve(self, skeleton_new, vertices=None):
        if vertices is None:
            vertices = self.vertices

        xy_known = skeleton_new - vertices[self.skeleton_inds].copy()
        K = kvf_operator(V=vertices, T=self.triangles)
        xy_unk = solve(K, xy_known, self.inds, self.skeleton_inds, vertices)
        diff = np.zeros_like(vertices)
        diff[self.skeleton_inds] = xy_known
        diff[self.inds] = xy_unk

        vertices = deform(vertices, diff, self.ninds, self.degrees, self.idxs, is_linear=False)

        return vertices


def ninds_degrees_idxs(neighbours):
    degrees = []
    ninds = []
    for i in range(len(neighbours)):
        n = neighbours[i]
        assert i not in n
        degrees.append(len(n))
        ninds.extend(n)
    degrees = np.array(degrees)
    ninds = np.array(ninds)

    idxs = np.cumsum(degrees)[:-1]
    idxs = np.concatenate(([0], idxs))

    return ninds, degrees, idxs


def kvf_operator(V, T):
    # V: [N, 2]
    # T: [T, 3]

    assert V.shape[1] == 2
    assert T.shape[1] == 3

    nv = len(V)
    nt = len(T)

    i, j, k = T.T
    v0 = V[i]
    v1 = V[j]
    v2 = V[k]

    e0 = v1 - v0
    e1 = v2 - v0
    e0 = np.pad(e0, ((0, 0), (0, 1)))
    e1 = np.pad(e1, ((0, 0), (0, 1)))

    cross_prods = np.cross(e0, e1)
    double_areas = np.abs(cross_prods[:, 2])[:, None]
    areas = 0.5 * double_areas[:, 0]

    v0 = v0 / double_areas
    v1 = v1 / double_areas
    v2 = v2 / double_areas

    vtxVal = []
    i_rows = np.arange(nt)
    data = np.ones(nt)
    for j_rows in T.T:
        val = sps.coo_matrix((data, (i_rows, j_rows)), shape=(nt, nv))
        vtxVal.append(val)

    # Polygon Mesh Processing, eq. 5.3
    grad_x = (
        vtxVal[0].multiply((v1[:, 1] - v2[:, 1])[:, None])
        + vtxVal[1].multiply((v2[:, 1] - v0[:, 1])[:, None])
        + vtxVal[2].multiply((v0[:, 1] - v1[:, 1])[:, None])
    )

    grad_y = (
        vtxVal[0].multiply((v2[:, 0] - v1[:, 0])[:, None])
        + vtxVal[1].multiply((v0[:, 0] - v2[:, 0])[:, None])
        + vtxVal[2].multiply((v1[:, 0] - v0[:, 0])[:, None])
    )

    I = sps.eye(2 * nv).tocsr()
    vx = I[:nv]
    vy = I[nv:]

    J11 = grad_x @ vx
    J12 = grad_y @ vx
    J21 = grad_x @ vy
    J22 = grad_y @ vy

    S11 = 2 * J11
    S12 = J12 + J21
    S21 = S12
    S22 = 2 * J22

    Sh = sps.hstack([S11.T, S12.T, S21.T, S22.T])
    D = sps.spdiags(areas, 0, nt, nt)
    B = sps.block_diag([D, D, D, D])
    Sv = sps.vstack([S11, S12, S21, S22])
    K = Sh @ B @ Sv
    K = 0.5 * (K + K.T)

    return K


def solve(K, xy_known, inds, mapping, points, tau=1e-3):
    nv = len(mapping) + len(inds)
    #K11t = sps.hstack(
    #    [
    #        K[np.ix_(mapping, mapping)],
    #        K[np.ix_(mapping, mapping + nv)],
    #    ]
    #)
    #K11b = sps.hstack(
    #    [
    #        K[np.ix_(mapping + nv, mapping)],
    #        K[np.ix_(mapping + nv, mapping + nv)],
    #    ]
    #)
    #K11 = sps.vstack([K11t, K11b])
    #assert np.allclose(K11.toarray(), K11.T.toarray())

    #K12t = sps.hstack(
    #    [
    #        K[np.ix_(mapping, inds)],
    #        K[np.ix_(mapping, inds + nv)],
    #    ]
    #)
    #K12b = sps.hstack(
    #    [
    #        K[np.ix_(mapping + nv, inds)],
    #        K[np.ix_(mapping + nv, inds + nv)],
    #    ]
    #)
    #K12 = sps.vstack([K12t, K12b])

    #K21t = sps.hstack(
    #    [
    #        K[nv_known:nv, :nv_known],
    #        K[nv_known:nv, nv : nv + nv_known],
    #    ]
    #)
    #K21b = sps.hstack(
    #    [
    #        K[nv + nv_known :, :nv_known],
    #        K[nv + nv_known :, nv : nv + nv_known],
    #    ]
    #)
    K21t = sps.hstack(
        [
            K[np.ix_(inds, mapping)],
            K[np.ix_(inds, mapping + nv)],
        ]
    )
    K21b = sps.hstack(
        [
            K[np.ix_(inds + nv, mapping)],
            K[np.ix_(inds + nv, mapping + nv)],
        ]
    )
    K21 = sps.vstack([K21t, K21b])
    #assert np.allclose(K12.toarray(), K21.T.toarray())

    #K22t = sps.hstack(
    #    [
    #        K[nv_known:nv, nv_known:nv],
    #        K[nv_known:nv, nv + nv_known :],
    #    ]
    #)
    #K22b = sps.hstack(
    #    [
    #        K[nv + nv_known :, nv_known:nv],
    #        K[nv + nv_known :, nv + nv_known :],
    #    ]
    #)
    K22t = sps.hstack(
        [
            K[np.ix_(inds, inds)],
            K[np.ix_(inds, inds + nv)],
        ]
    )
    K22b = sps.hstack(
        [
            K[np.ix_(inds + nv, inds)],
            K[np.ix_(inds + nv, inds + nv)],
        ]
    )
    K22 = sps.vstack([K22t, K22b])
    #assert np.allclose(K22.toarray(), K22.T.toarray())

    xy_known = xy_known.T.ravel()

    #xy_unk, istop, itn, normr, *_ = sps.linalg.lsqr(K22, b)
    #e1 = np.zeros((nv - nv_known, 2))
    #e1[:, 0] = 1
    #e1 = e1.ravel()
    #n = np.linalg.norm(e1)
    #e1 /= n
    #e2 = np.ones_like(e1) - e1
    #e2 /= n
    #e3 = points[nv_known:]
    #e3 = np.hstack([e3[:, 1], -e3[:, 0]])
    #e3 /= np.linalg.norm(e3)
    #e123t = np.vstack([e1, e2, e3])
    #print(e123t.shape, xy_unk.shape)
    #xy_unk -= e123t.T @ (e123t @ xy_unk)
    #xy_unk = points[nv_known:].ravel() - xy_unk

    #I = sps.eye(2 * (nv - nv_known), format="csc")
    #K22 += tau * I

    b = -(K21 @ xy_known)
    xy_unk = sps.linalg.spsolve(K22, b)

    #w = 1e-1
    #A = sps.vstack([w * K21.T, K22])
    #B = sps.vstack([w * K11, K21])
    #b = -(B @ xy_known)
    #xy_unk, *_ = sps.linalg.lsqr(A, b)

    # print(K11.shape, K22.shape, K21.shape, xy_known.shape, xy_unk.shape)
    #print(K11 @ xy_known + K12 @ xy_unk)
    #print((K21 @ xy_known + K22 @ xy_unk))

    # K22_inv = np.linalg.pinv(K22.toarray())
    # xy_unk = K22_inv @ xy_known
    xy_unk = xy_unk.reshape(2, -1).T

    return xy_unk


def to_complex(a):
    return a[:, 0] + a[:, 1] * 1j


def from_complex(a):
    return np.vstack([a.real, a.imag]).T


def deform(verts, kvf_field, ninds, degrees, idxs, is_linear=False):
    if is_linear:
        gamma = verts + kvf_field
    else:
        # As-Killing-as-possible
        v = np.repeat(verts, degrees, axis=0)
        vi = verts[ninds]
        kvf_field = -kvf_field
        u = np.repeat(kvf_field, degrees, axis=0)
        ui = kvf_field[ninds]
        v = to_complex(v)
        vi = to_complex(vi)
        u = to_complex(u)
        ui = to_complex(ui)
        si = (u - ui) / (v - vi)
        ci = v + u / si
        gamma = ci + np.exp(si) * (v - ci)
        # https://stackoverflow.com/questions/70322757/numpy-sum-over-1-d-array-split-by-index
        gamma = np.add.reduceat(gamma, idxs, axis=0)
        gamma = gamma / degrees
        gamma = from_complex(gamma)

    return gamma


def simple_test():
    V = np.array(
        [
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1],
        ],
        dtype="float32",
    )
    T = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
        ],
        dtype="int64",
    )
    ANS = np.array(
        [
            [3, -2, 0, -1, 0, 0, -1, 1],
            [-2, 3, -1, 0, 1, -1, 0, 0],
            [0, -1, 3, -2, -1, 1, 0, 0],
            [-1, 0, -2, 3, 0, 0, 1, -1],
            [0, 1, -1, 0, 3, -1, 0, -2],
            [0, -1, 1, 0, -1, 3, -2, 0],
            [-1, 0, 0, 1, 0, -2, 3, -1],
            [1, 0, 0, -1, -2, 0, -1, 3],
        ]
    )

    K = kvf_operator(V, T)

    assert np.allclose(K.toarray(), ANS)


def complex_test():
    data = scipy.io.loadmat("./data/vt.mat")
    V = data["V"]
    T = data["T"]
    K = kvf_operator(V=V, T=T)

    data = scipy.io.loadmat("./data/k.mat")
    K_mat = data["K"]

    assert np.allclose(K.toarray(), K_mat.toarray())


if __name__ == "__main__":
    simple_test()
    complex_test()
