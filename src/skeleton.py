import math

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchmin import minimize

import pose_estimation
import write_utils

import numpy as np


def make_trans(t: torch.Tensor):
    e = torch.eye(2, device=t.device, dtype=t.dtype)
    z = t.new_tensor(0)
    o = t.new_tensor(1)
    trans = torch.cat(
        [
            torch.cat([e, t.unsqueeze(1)], dim=1),
            torch.stack([z, z, o]).unsqueeze(0),
        ],
        dim=0,
    )

    return trans


def make_trans3(t: torch.Tensor):
    e = torch.eye(3, device=t.device, dtype=t.dtype)
    z = t.new_tensor(0)
    o = t.new_tensor(1)
    trans = torch.cat(
        [
            torch.cat([e, t.unsqueeze(1)], dim=1),
            torch.stack([z, z, z, o]).unsqueeze(0),
        ],
        dim=0,
    )

    return trans


def make_rot(phi):
    s = torch.sin(phi)
    c = torch.cos(phi)
    z = torch.zeros_like(phi)
    o = torch.ones_like(phi)
    rot = torch.stack(
        [
            torch.stack([c, -s, z]),
            torch.stack([s, c, z]),
            torch.stack([z, z, o]),
        ],
        dim=0,
    )

    return rot


def make_rot3(phi, order="xyz"):
    assert order in ["xyz"], f"euler to rot is implemented only for xyz order. Given {order}"
    # phi [3]
    s = torch.sin(phi)
    c = torch.cos(phi)
    z = torch.zeros_like(phi)
    o = torch.ones_like(phi)
    rot_z = torch.stack(
        [
            torch.stack([c[2],-s[2], z[2], z[2]]),
            torch.stack([s[2], c[2], z[2], z[2]]),
            torch.stack([z[2], z[2], o[2], z[2]]),
            torch.stack([z[2], z[2], z[2], o[2]]),
        ],
        dim=0,
    )
    rot_y = torch.stack(
        [
            torch.stack([c[1], z[1],s[1], z[1]]),
            torch.stack([z[1], o[1], z[1], z[1]]),
            torch.stack([-s[1], z[1], c[1], z[1]]),
            torch.stack([z[1], z[1], z[1], o[1]]),
        ],
        dim=0,
    )
    rot_x = torch.stack(
        [
            torch.stack([o[0], z[0],  z[0], z[0]]),
            torch.stack([z[0], c[0], -s[0], z[0]]),
            torch.stack([z[0], s[0],  c[0], z[0]]),
            torch.stack([z[0], z[0],  z[0], o[0]]),
        ],
        dim=0,
    )

    rot = rot_z @ rot_y @ rot_x

    return rot


def make_point(x):
    o = x.new_tensor([1.0])
    x = torch.cat([x, o])

    return x


class Skeleton2d:
    def __init__(self, parents, offsets, inds=None, zero_angles_inds=None):
        assert len(parents) == len(offsets)
        assert parents[0] == -1
        assert sum(p == -1 for p in parents) == 1

        self.parents = parents
        self.offsets = offsets
        self.inds = inds
        self.zero_angles_inds = zero_angles_inds

        self.dim = 2
        self.order = "xyz"

    def __str__(self):
        return f"parents={self.parents}\noffsets=\n{self.offsets.cpu().numpy()}"

    def __repr__(self):
        return self.__str__()

    def num_joints(self):
        return len(self.parents)

    def forward(self, theta=None, k=None, root_pos=None, reduce=False, return_rot=False):
        theta = self.offsets.new_zeros(len(self.offsets)) if theta is None else theta
        offsets = self.offsets
        if k is not None:
            offsets = k * offsets

        e = torch.eye(3, device=theta.device, dtype=theta.dtype)

        positions = []
        rots = []
        ts = []
        for parent, o, phi in zip(self.parents, offsets, theta):
            trans = make_trans(o)
            rot = make_rot(phi)
            t = trans @ rot
            if parent == -1:
                p_t = e
                rots.append(e)
            else:
                p_t = ts[parent]
                #abs_rots.append(abs_rots[parent] @ rot)
                rots.append(rot)

            pos = p_t @ make_point(o)

            positions.append(pos[:2])
            ts.append(p_t @ t)

        positions = torch.stack(positions, dim=0)

        if reduce and self.inds is not None:
            positions = positions[self.inds]

        if root_pos is not None:
            positions = positions + root_pos

        if return_rot:
            rots = torch.stack(rots, dim=0)

            return positions, rots

        return positions

    def render(self, pos=None, ax=None, color="orange"):
        if ax is None:
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(1, 1, 1)

        if pos is None:
            pos = self.forward(reduce=False)

        pos = pos.cpu().numpy()
        ax.scatter(pos[:, 0], -pos[:, 1], c=color)
        for j, _ in enumerate(pos):
            p = self.parents[j]
            if p == -1:
                continue

            ax.plot(pos[[p, j], 0], -pos[[p, j], 1], color=color)

        return ax

    def calc_angles_and_offsets_ik(self, kps_b, root_pos=None):
        n = len(self.offsets)

        def f(x):
            theta, ks = torch.split(x, n)
            theta = theta.view(-1)
            ks = ks.view(-1, 1).exp()
            p = self.forward(theta, ks, root_pos=root_pos, reduce=True)
            mse = ((p - kps_b) ** 2).sum(1).mean(0)

            return mse

        x = kps_b.new_zeros(self.dim * n)
        res = minimize(f, x, method="bfgs")
        x = res.x
        theta, ks = torch.split(x, n)
        theta = theta.view(-1)
        ks = ks.view(-1, 1).exp()

        err = ((kps_b[:, :2] - self.forward(theta, reduce=True)[:, :2]) ** 2).sum(1).mean(0)

        return theta, ks, err


class Skeleton3d:
    def __init__(self, parents, offsets, inds=None, zero_angles_inds=None, order="xyz"):
        assert len(parents) == len(offsets)
        assert parents[0] == -1
        assert sum(p == -1 for p in parents) == 1

        self.parents = parents
        self.offsets = torch.tensor(offsets, dtype=torch.float64)
        self.inds = inds
        self.zero_angles_inds = zero_angles_inds
        self.order = order
        self.dim = 3

    def __str__(self):
        return f"3d: parents={self.parents}\noffsets=\n{self.offsets.cpu().numpy()}"

    def __repr__(self):
        return self.__str__()

    def num_joints(self):
        return len(self.parents)

    def forward(self, theta=None, k=None, root_pos=None, reduce=False, return_rot=False):
        theta = self.offsets.new_zeros(len(self.offsets), 3) if theta is None else theta
        assert theta.shape[1] == 3
        offsets = self.offsets
        if k is not None:
            offsets = k * offsets

        e = torch.eye(4, device=theta.device, dtype=theta.dtype)

        positions = []
        rots = []
        ts = []
        for parent, o, phi in zip(self.parents, offsets, theta):
            trans = make_trans3(o)
            rot = make_rot3(phi, order=self.order)
            t = trans @ rot
            if parent == -1:
                p_t = e
                rots.append(e)
            else:
                p_t = ts[parent]
                #abs_rots.append(abs_rots[parent] @ rot)
                rots.append(rot)

            pos = p_t @ make_point(o)

            positions.append(pos[:3])
            ts.append(p_t @ t)

        positions = torch.stack(positions, dim=0)
        if reduce and self.inds is not None:
            positions = positions[self.inds]

        #positions = positions[:, :2]

        if return_rot:
            rots = torch.stack(rots, dim=0)
            rots = rots[:, :3, :3]

            return positions, rots

        return positions

    def render(self, pos=None, ax=None, color="orange"):
        if ax is None:
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(1, 1, 1)

        if pos is None:
            pos = self.forward(reduce=False)

        pos = pos.cpu().numpy()
        ax.scatter(pos[:, 0], -pos[:, 1], c=color)
        for j, _ in enumerate(pos):
            p = self.parents[j]
            if p == -1:
                continue

            ax.plot(pos[[p, j], 0], -pos[[p, j], 1], color=color)

        return ax

    def calc_angles_and_offsets_ik(self, kps_b, theta0=None, ks0=None, use_random=False):
        n = len(self.offsets)

        def f(x):
            theta, ks = x.split([self.dim * n, n])
            theta = theta.view(n, self.dim)
            ks = ks.view(-1, 1).exp()
            p = self.forward(theta, ks, reduce=True)
            #p = p[:, :2]
            mse = ((p[:, :2] - kps_b[:, :2]) ** 2).sum(1).mean(0)
            #l2 = (theta[self.zero_angles_inds] ** 2).sum(1).mean()
            #mse = mse + 1e+1 * l2

            return mse

        x = kps_b.new_zeros((self.dim + 1) * n)
        theta, ks = x.split([self.dim * n, n])
        theta = theta.view(n, self.dim)
        theta.zero_()
        if use_random:
            theta.normal_(0, 1e-2)

        x = torch.cat([theta.view(-1), ks.view(-1)])

        res = minimize(f, x, method="bfgs")
        x = res.x
        theta, ks = x.split([self.dim * n, n])
        theta = theta.view(n, self.dim)
        ks = ks.view(-1, 1).exp()

        return theta, ks

    def calc_angles_ik(self, kps_b, theta0=None, use_random=False):
        n = len(self.offsets)

        def f(x):
            theta = x.view(n, self.dim)
            p = self.forward(theta, reduce=True)
            #p = p[:, :2]
            #mse = ((p - kps_b) ** 2).sum(1).mean(0)
            mse_xy = ((p[:, :2] - kps_b[:, :2]) ** 2).sum(1).mean(0)
            mse_z = ((p[:, 2] - kps_b[:, 2]) ** 2).mean(0)
            mse = mse_xy + 1e-2 * mse_z
            #l2 = (theta[self.zero_angles_inds] ** 2).sum(1).mean()
            #mse = mse + 1e+1 * l2
            l2 = (theta ** 2).sum(1).mean(0)
            mse = mse + 1e-4 * l2

            return mse

        x = kps_b.new_zeros(self.dim * n)
        theta = x.view(n, self.dim)
        theta.zero_()
        if use_random:
            theta.normal_(0, 1e-2)

        x = theta.view(-1)

        res = minimize(f, x, method="bfgs", tol=1e-14)
        x = res.x
        theta = x.view(n, self.dim)

        err = ((kps_b[:, :2] - self.forward(theta, reduce=True)[:, :2]) ** 2).sum(1).mean(0)

        return theta, err


def create_offsets(kps, parents, parents_to_kps):
    offsets = []
    for p, j in zip(parents, parents_to_kps):
        if p == -1:
            offset = kps[j]
        else:
            offset = kps[j] - kps[parents_to_kps[p]]

        offsets.append(offset)

    offsets = np.array(offsets, dtype="float64")

    return offsets


def create_offsets3(k=1):
    offsets = [
        [0, 0, 0],  # root

        [0, 0, 0],  # hips
        [0, -1, 0],  # spine
        [0, -1, 0],  # neck
        [0, 0, 0],   # neck
        [-1, 0, 0],   # l shoulder
        [-1, 0, 0],   # l arm
        [-1, 0, 0],   # l hand

        [0, 0, 0],   # neck
        [0, -1, 0],   # head

        [0, 0, 0],   # neck
        [1, 0, 0],   # r shoulder
        [1, 0, 0],   # r arm
        [1, 0, 0],   # r hand

        [0, 0, 0],  # hips
        [-1, 0.5, 0],   # l hip
        [0, 1, 0],   # l knee
        [0, 1, 0],   # l foot
        [-1, 0, 0],   # l toe
        #[0, 0, 1],   # l toe

        [0, 0, 0],  # hips
        [1, 0.5, 0],   # r hip
        [0, 1, 0],   # r knee
        [0, 1, 0],   # r foot
        [1, 0, 0],   # r toe
        #[0, 0, 1],   # r toe
    ]

    offsets = np.array(offsets, dtype="float64")
    n = np.linalg.norm(offsets, ord=2, axis=-1)
    m = n > 0
    offsets[m] /= n[m, None]
    offsets = k * offsets

    return offsets


def cross(a, b):
    c = a[0] * b[1] - a[1] * b[0]

    return c


def get_fk(skeleton, parents2d, parents2d_to_kps, zero_inds, predicted_keypoints_2d_normalized=None, dim=2):
    offsets2d_1 = create_offsets(
        skeleton,
        parents2d,
        parents2d_to_kps,
    )
    if predicted_keypoints_2d_normalized is not None:
        assert dim == 3
        k1 = np.linalg.norm(offsets2d_1, axis=-1)
        m = k1 > 0
        offsets2d_1[m] /= k1[m, None]

        offsets2d_2 = create_offsets(
            predicted_keypoints_2d_normalized,
            parents2d,
            parents2d_to_kps,
        )
        k2 = np.linalg.norm(offsets2d_2, axis=-1)
        k = np.maximum(k1, k2)
        k = k[:, None]
        #offsets = create_offsets3(k)
        offsets2d_1 = k * offsets2d_1

    offsets = offsets2d_1
    if dim == 3:
        offsets = np.pad(offsets, ((0, 0), (0, 1)))

    offsets = torch.from_numpy(offsets)
    inds = [
        parents2d_to_kps.index(i)
        for i in range(len(set(parents2d_to_kps)))
    ]

    if dim == 2:
        skel = Skeleton2d(parents2d, offsets, inds, zero_inds)
    elif dim == 3:
        skel = Skeleton3d(parents2d, offsets, inds, zero_inds)
    else:
        raise NotImplementedError(f"No skeleton for {dim=}")

    return skel



def calc_angles_and_offsets(kps_a, kps_b, parents, parents_to_kps):
    # phi = atan2(coss, dot)
    theta = np.zeros(len(parents), dtype="float64")
    ks = np.ones_like(theta)[:, None]
    theta2 = np.zeros_like(theta)
    #from ipdb import set_trace; set_trace()
    for i, (p, j) in enumerate(zip(parents, parents_to_kps)):
        if p == -1:
            continue
        else:
            a = kps_a[j] - kps_a[parents_to_kps[p]]
            b = kps_b[j] - kps_b[parents_to_kps[p]]
            an = np.linalg.norm(a)
            bn = np.linalg.norm(b)
            if np.allclose(an, 0) and np.allclose(bn, 0):
                continue

            c = cross(a, b)
            d = (a * b).sum()
            atan2 = math.atan2(c, d)
            phi = atan2 - theta[p]
            theta[i] = phi
            theta2[p] = phi
            ks[i] = bn / an

    return theta2, ks


if __name__ == "__main__":
    #parents = [-1, 0, 1]
    #offsets = torch.tensor(
    #    [
    #        [0., 0],
    #        [1, 0],
    #        [1, 0],
    #    ]
    #)

    #skel = Skeleton2d(parents, offsets)
    #pos = skel.forward()
    #pos_ans = torch.tensor(
    #    [
    #        [0., 0],
    #        [1, 0],
    #        [2, 0],
    #    ]
    #)
    #assert torch.allclose(pos_ans, pos, atol=1e-7), torch.abs(pos_ans - pos).max()

    #theta = torch.tensor([0, math.pi / 2, 0])
    #pos = skel.forward(theta)
    #pos_ans = torch.tensor(
    #    [
    #        [0., 0],
    #        [1, 0],
    #        [1, 1],
    #    ]
    #)
    #assert torch.allclose(pos_ans, pos, atol=1e-7), torch.abs(pos_ans - pos).max()

    #theta = torch.tensor([math.pi / 4, math.pi / 2, 0])
    #pos = skel.forward(theta)
    #pos_ans = torch.tensor(
    #    [
    #        [0., 0],
    #        [math.sqrt(2) / 2, math.sqrt(2) / 2],
    #        [0, math.sqrt(2)],
    #    ]
    #)
    #assert torch.allclose(pos_ans, pos, atol=1e-7), torch.abs(pos_ans - pos).max()

    #theta = torch.tensor([-math.pi / 4, math.pi / 2, 0])
    #pos = skel.forward(theta)
    #pos_ans = torch.tensor(
    #    [
    #        [0., 0],
    #        [math.sqrt(2) / 2, -math.sqrt(2) / 2],
    #        [math.sqrt(2), 0],
    #    ]
    #)
    #assert torch.allclose(pos_ans, pos, atol=1e-7), torch.abs(pos_ans - pos).max()

    #p1 = np.array(
    #    [
    #        [0, 0],
    #        [1, 0],
    #        [2, 0],
    #    ],
    #    dtype="float64",
    #)
    #p2 = np.array(
    #    [
    #        [0., 0],
    #        [2, 0],
    #        [2, 4],
    #    ],
    #    dtype="float64",
    #)
    #theta_ans = torch.tensor([0, math.pi / 2, 0])
    #theta, ks = calc_angles_and_offsets(
    #    p1,
    #    p2,
    #    parents,
    #    [0, 1, 2],
    #)
    #theta = torch.from_numpy(theta)
    #assert torch.allclose(theta_ans, theta)
    #ks = torch.from_numpy(ks)
    #print(theta)
    #print(ks)
    #pos = skel.forward(theta, ks)
    #assert np.allclose(p2, pos.numpy(), atol=1e-7)
    #print(pos)

    parents = [-1, 0, 1]
    offsets = torch.tensor(
        [
            [0., 0, 0],
            [1, 0, 0],
            [1, 0, 0],
        ]
    )

    skel = Skeleton3d(parents, offsets)
    pos = skel.forward()
    pos_ans = torch.tensor(
        [
            [0., 0, 0],
            [1, 0, 0],
            [2, 0, 0],
        ]
    )
    assert torch.allclose(pos_ans, pos, atol=1e-7), torch.abs(pos_ans - pos).max()

    theta = torch.tensor([
        [0, 0, 0],
        [0, 0, math.pi / 2],
        [0, 0, 0],
    ])
    pos = skel.forward(theta)
    pos_ans = torch.tensor(
        [
            [0., 0, 0],
            [1, 0, 0],
            [1, 1, 0],
        ]
    )
    assert torch.allclose(pos_ans, pos, atol=1e-7), torch.abs(pos_ans - pos).max()

    theta = torch.tensor([
        [0, 0, math.pi / 4],
        [0, 0, math.pi / 2],
        [0, 0, 0],
    ])
    pos = skel.forward(theta)
    pos_ans = torch.tensor(
        [
            [0., 0, 0],
            [math.sqrt(2) / 2, math.sqrt(2) / 2, 0],
            [0, math.sqrt(2), 0],
        ]
    )
    assert torch.allclose(pos_ans, pos, atol=1e-7), torch.abs(pos_ans - pos).max()

    theta = torch.tensor([
        [0, 0, -math.pi / 4],
        [0, 0, math.pi / 2],
        [0, 0, 0],
    ])
    pos = skel.forward(theta)
    pos_ans = torch.tensor(
        [
            [0., 0, 0],
            [math.sqrt(2) / 2, -math.sqrt(2) / 2, 0],
            [math.sqrt(2), 0, 0],
        ]
    )
    #print(f"{theta=}")
    #print(f"{pos=}")
    #print(f"{pos_ans=}")
    assert torch.allclose(pos_ans, pos, atol=1e-7), torch.abs(pos_ans - pos).max()

    #theta, k = calc_angles_and_offsets_ik3(
    #theta = calc_angles_ik3(
    #    skel,
    #    np.array(
    #        [
    #            [0, 0, 0],
    #            [1, 0, 1],
    #            [2, 0, 0],
    #        ]
    #    ),
    #)
    #print(theta)
    ##print(skel.forward(theta, k).round())
    #print(skel.forward(theta))
    #raise
    #p1 = np.array(
    #    [
    #        [0, 0, 0],
    #        [1, 0, 0],
    #        [2, 0, 0],
    #    ],
    #    dtype="float64",
    #)
    #p2 = np.array(
    #    [
    #        [0., 0, 0],
    #        [2, 0, 0],
    #        [2, 4, 0],
    #    ],
    #    dtype="float64",
    #)
    #theta_ans = torch.tensor([
    #    [0, 0, 0],
    #    [0, 0, math.pi / 2],
    #    [0, 0, 0],
    #])
    #theta, ks = calc_angles_and_offsets(
    #    p1,
    #    p2,
    #    parents,
    #    [0, 1, 2],
    #)
    #theta = torch.from_numpy(theta)
    #assert torch.allclose(theta_ans, theta)
    #ks = torch.from_numpy(ks)
    #print(theta)
    #print(ks)
    #pos = skel.forward(theta, ks)
    #assert np.allclose(p2, pos.numpy(), atol=1e-7)
    #print(pos)

    k = 1# + np.clip(np.random.randn(24)[:, None], 0, 1)
    offsets = create_offsets3(k)
    #print(offsets)
    offsets = torch.from_numpy(offsets)
    inds = [
        pose_estimation.PARENTS2D_TO_KPS.index(i)
        for i in range(len(pose_estimation.KPS))
    ]
    skel = Skeleton3d(pose_estimation.PARENTS2D, offsets, inds)
    pos = skel.forward(reduce=True)
    pos = pos.clone()
    pos[6] = pos[5]
    pos[6, 2] = 1
    print(pos)

    a = calc_angles_ik3(skel, pos.cpu().numpy())
    pred = skel.forward(a, reduce=True)
    print(pred)
    print(torch.abs(pred - pos).max())
    raise
    #animation = torch.randn_like(offsets) * 180
    #print(skel.forward(animation))

    order = "xyz"
    animation = animation.unsqueeze(0).cpu().numpy()
    # rotate root over x by pi
    # animation[:, 0, 0] = 180
    offset = np.pad(skel.offsets.cpu().numpy(), ((0, 0), (0, 1)))
    write_utils.write_bvh(
        parent=pose_estimation.PARENTS2D,
        offset=offset,
        rotation=animation,
        position=np.zeros((len(animation), 3)),
        names=[f"{i}" for i in pose_estimation.PARENTS2D],
        frametime=1,
        order=order,
        path="skeleton3.bvh",
        endsite=None,
    )
