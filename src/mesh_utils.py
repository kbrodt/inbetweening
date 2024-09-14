import copy
import itertools
from dataclasses import dataclass
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import igl
import triangle as tr
from scipy.spatial.distance import cdist
from shapely import intersects_xy
from shapely.geometry import (
    Polygon,
    Point,
    LineString,
    MultiLineString,
)

import inb
import utils


@dataclass()
class OverlapingArea:
    idx: int
    poly: Polygon
    contour: Optional[np.ndarray] = None
    mapping: Optional[dict] = None
    t_ov_inds: Optional[np.ndarray] = None
    t_ov_inds_f: Optional[np.ndarray] = None
    edges: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.idx < 1:
            raise ValueError(f"Id must be positive. Got {self.idx}")


class OverlapingMesh:
    def __init__(
        self,
        vertices,
        triangles,
        overlaping_areas,
        keypoints_2d,
        bone_to_top,
        t_jun,
        bnd,
        skeleton_data,
        n_sample_pts=1,
    ):
        self.vertices: np.ndarray = vertices
        self.triangles: np.ndarray = triangles
        # TODO: add sort areas <24-11-23 kbrodt> #
        self.overlaping_areas: list[OverlapingArea] = overlaping_areas
        self.bnd = set(bnd)

        self.bone_to_top = bone_to_top
        self.t_jun = t_jun

        self.n_sample_pts = n_sample_pts
        self.keypoints_2d = keypoints_2d
        self.skeleton_data = skeleton_data

        self._post_init(keypoints_2d)

    def _post_init(self, keypoints_2d):
        jinp = set()
        self.poly_to_bones = {}
        for oa in self.overlaping_areas:
            jinp.update(find_joints_inside_poly(keypoints_2d, oa.poly))
            self.poly_to_bones[oa.idx] = find_bones_intersect_poly(
                keypoints_2d,
                oa.poly,
                self.skeleton_data,
            )

        self.bone_to_top = sorted(
            set(self.bone_to_top)
            &
            set(itertools.chain(*list(self.poly_to_bones.values())))
        )

        vertices_to_map = set()
        for bi in self.bone_to_top:
            a, b = self.skeleton_data.skeleton[bi]
            if a in jinp:
                vertices_to_map.add(a)

            if b in jinp:
                vertices_to_map.add(b)

        self.mapping = {}
        self.t_ov_inds = []
        self.t_ov_inds_f = []
        for oa in self.overlaping_areas:
            self.mapping.update(oa.mapping)
            self.t_ov_inds.extend(oa.t_ov_inds)
            self.t_ov_inds_f.extend(oa.t_ov_inds_f)

        self.vert_to_skel = [
            self.mapping[i] if i in vertices_to_map else i
            for i in range(len(self.skeleton_data.joints))
        ]

        sverts_inds = np.arange(
            self.n_sample_pts * len(self.skeleton_data.skeleton)
        ) + len(self.skeleton_data.joints)

        if len(sverts_inds) > 0:
            for oa in self.overlaping_areas:
                m = oa.mapping
                for i, (a, b) in enumerate(self.skeleton_data.skeleton):
                    if not (i in set(self.bone_to_top) & set(self.poly_to_bones[oa.idx])):
                        continue

                    sverts_inds[self.n_sample_pts*i:self.n_sample_pts*(i + 1)] = [
                        m.get(p, p)
                        for p in sverts_inds[self.n_sample_pts*i:self.n_sample_pts*(i + 1)]
                    ]

        self.sverts_inds = list(sverts_inds)

        BE = []
        for x, y in self.skeleton_data.skeleton:
           if x in vertices_to_map:
               x = self.mapping[x]
           if y in vertices_to_map:
               y = self.mapping[y]

           BE.append((x, y))

        self.BE = np.array(BE)
        self.CE = np.array(self.sverts_inds).reshape(len(self.skeleton_data.skeleton), self.n_sample_pts)

    def sample_pts_on_bones(self, kps, is_bary=False):
        n_pts = self.n_sample_pts + 2
        p_bs = []
        xs = []
        for i, (a, b) in enumerate(self.skeleton_data.skeleton):
            u = kps[a]
            v = kps[b]
            x = np.linspace(u, v, n_pts - 1, endpoint=False)[1:]
            xs.extend(x)
            if not is_bary:
                continue

            p_b, _ = utils._xy_to_barycentric_coords(
                x,
                self.vertices,
                self.triangles,
            )
            assert len(x) == len(p_b), (i, len(x), len(p_b))

            if i in self.bone_to_top:
            #if self.vert_to_skel[a] != a or self.vert_to_skel[b] != b:
                p_b = [
                    [
                        (self.mapping.get(i, i), uv)
                        for i, uv in p
                    ]
                    for p in p_b
                ]

            p_bs.extend(p_b)

        xs = np.array(xs, dtype="float64")
        if is_bary:
            B = np.array([[uvw for _, uvw in p] for p in p_bs], dtype="float64")[:, :, None]
            vinds = np.array([[i for i, _ in p] for p in p_bs], dtype="int64")
        else:
            return xs

        return xs, vinds, B

    def _split_node(self, G, node):
        """https://stackoverflow.com/questions/65853641/networkx-how-to-split-nodes-into-two-effectively-disconnecting-edges"""
        edges = G.edges(node, data=True)
        new_edges = []
        new_nodes = []

        H = G.__class__()
        H.add_nodes_from(G.subgraph(node))
        for i, (s, t, data) in enumerate(edges):
            new_node = '{}_{}'.format(node, i)
            I = nx.relabel_nodes(H, {node:new_node})
            new_nodes += list(I.nodes(data=True))
            new_edges.append((new_node, t, data))

        G.remove_node(node)
        G.add_nodes_from(new_nodes)
        G.add_edges_from(new_edges)

        return G

    def _relabel_nodes(self, G):
        mapping = {
            n: int(n.split("_")[0])
            for n in G.nodes if isinstance(n, str)
        }
        print(f"relabel_nodes {mapping}")
        if len(mapping) > 0:
            G = nx.relabel_nodes(G, mapping)

        return G

    def get_paths(self, to_show=False):
        #segment_markers = self.segment_markers.ravel()
        #S = self.segments
        #boundary = S[segment_markers == 1]  # [n, 2]
        #boundary = np.unique(boundary.ravel())
        #boundary = igl.boundary_loop(self.triangles)
        #boundary = self.bnd#igl.boundary_loop(self.triangles)

        idx2paths = []
        for oa in self.overlaping_areas:
            t_ov_inds = oa.t_ov_inds
            t_ov_inds_f = oa.t_ov_inds_f
            poly = oa.poly
            mapping = oa.mapping

            #uc = S[segment_markers == 2 * oa.idx]  # [N, 2]
            uc = oa.edges

            #self._path = np.unique(uc.ravel())
            g = nx.Graph()
            g.add_edges_from(uc)
            #for node in boundary:
            #    if g.has_node(node):
            #        nodes = list(g.edges(node))
            #        if len(nodes) == 2:
            #            g = self._split_node(g, node)
            #            # split graph

            ucc = [
                #self._relabel_nodes(g.subgraph(c).copy())
                g.subgraph(c).copy()
                for c in nx.connected_components(g)
            ]
            contours = []
            for j, ug in enumerate(ucc):
                source_target = [v for v in ug.nodes() if ug.degree[v] == 1]
                is_cycle = False
                split_inds = []
                if len(source_target) == 0:
                    path = [
                        a
                        for a, _ in nx.find_cycle(ug)
                    ]
                    _inds_ee = list(
                        set(self.triangles[t_ov_inds].ravel())
                        &
                        set(self.skeleton_data.end_effectors_inds)
                    )
                    print("cylce?", path)
                    if len(_inds_ee) > 0:
                        _v1 = self.vertices[path]
                        _v2 = self.vertices[_inds_ee]
                        pw = cdist(_v1, _v2, metric="euclidean")
                        v_node, _ = np.unravel_index(pw.argmin(), pw.shape)
                        path = path[v_node:] + path[:v_node]
                        print("new path")
                        print(path)
                        is_cycle = True
                else:
                    # TODO: more than two end vertices <05-12-23 kbrodt> #
                    if len(source_target) == 3:
                        source_target = list(
                            set(source_target)
                            -
                            set(
                                u
                                for v in [
                                    v for v in ug.nodes() if ug.degree[v] == 3
                                ]
                                for u in ug.neighbors(v)
                            )
                        )

                    source, target = source_target
                    path: list[int] = nx.shortest_path(
                        ug,
                        source=source,
                        target=target,
                    )

                if is_cycle:
                    manifold_joints = set(itertools.chain(*[self.skeleton_data.end_effector2par[i] for i in _inds_ee]))
                else:
                    manifold_joints = set(range(len(self.skeleton_data.skeleton)))

                manifold_joints = manifold_joints & self.skeleton_data.manifold_joints
                print(f"0: {oa.idx=} {manifold_joints}")

                binov = [
                    (a, b)
                    for (a, b) in [
                        self.skeleton_data.skeleton[bone]
                        for bone in self.poly_to_bones[oa.idx]
                        if bone in self.skeleton_data.manifold_bones
                    ]
                    if a in manifold_joints or b in manifold_joints
                ]
                print(f"1: {oa.idx=} {binov=}")
                binov = [
                    (a, b)
                    for a, b in binov
                    if MultiLineString(
                        [self.vertices[[a, b]] for a, b in zip(path, path[1:])]
                    ).intersects(LineString(self.vertices[[a, b]]))
                ]

                print(f"2: {oa.idx=} {binov=}")
                for a, b in binov:
                    if a in manifold_joints:
                        v = a
                        print('a', v)
                    else:
                        assert b in manifold_joints
                        v = b
                        print('b', v)
                    _v1 = self.vertices[path]
                    _v2 = self.vertices[[v]]
                    pw = cdist(_v1, _v2, metric="euclidean")
                    v_node, _ = np.unravel_index(pw.argmin(), pw.shape)

                    if to_show:
                        plt.title(f"area: {oa.idx} nearest vertex")
                        plt.triplot(
                            self.vertices[:, 0],
                            -self.vertices[:, 1],
                            self.triangles,
                        )
                        plt.scatter(
                            self.vertices[path, 0],
                            -self.vertices[path, 1],
                            color="blue",
                        )
                        plt.scatter(
                            self.vertices[[v], 0],
                            -self.vertices[[v], 1],
                            color="green",
                            label=f"source {v=} d:{pw[v_node].item():.3f}",
                        )
                        plt.scatter(
                            self.vertices[[path[v_node]], 0],
                            -self.vertices[[path[v_node]], 1],
                            color="red",
                            label=f"target {v_node=} ({len(path) - 1=})",
                        )
                        plt.legend()
                        plt.gca().set_aspect("equal")
                        plt.show()
                        plt.close()

                    print(f"{oa.idx=} {v_node} {len(path) - 1}")
                    if 0 < v_node < len(path) - 1:
                        split_inds.append(
                            (
                                pw[v_node].item(),
                                v_node,
                                a in self.skeleton_data.adjacent or b in self.skeleton_data.adjacent,
                            )
                        )

                if len(split_inds) > 0:
                    print(f"3: {oa.idx=} {split_inds}")
                    #if len(split_inds) > 1:
                    #    split_inds = list(filter(lambda x: x[-1], split_inds))

                    #if len(split_inds) > 0:
                    split_inds_val, v_node, is_adjacent = min(split_inds, key=lambda x: x[0])
                    split_inds = [v_node]
                else:
                    is_adjacent = False
                    split_inds_val = None
                    #if is_adjacent:
                    #    split_inds = [v_node]
                    #else:
                    #    split_inds = []
                        #_, v_node, is_adjacent = min(split_inds, key=lambda x: x[0])
                        #if is_adjacent:
                        #    split_inds = [v_node]
                        #else:
                        #    split_inds = []

                if is_cycle and len(split_inds) == 0:
                    continue

                contours.append(
                    (
                        split_inds_val,
                        is_adjacent,
                        ContourPath(
                            path=path,
                            is_cycle=is_cycle,
                            split_inds=split_inds,
                        )
                    )
                )

            if len(contours) == 0:
                continue

            if any(ia for _, ia, _ in contours):
                assert sum(ia for _, ia, _ in contours) == 1, sum(ia for _, ia, _ in contours)
                _contours = []
                for _, ia, c in contours:
                    if ia and not c.is_cycle:
                        _contours.append(c)
                    else:
                        _contours.append(
                            ContourPath(
                                path=c.path,
                                is_cycle=c.is_cycle,
                                split_inds=[],
                            )
                        )
                contours = _contours
            elif any(v is not None for v, *_ in contours):
                i_f = None
                min_val = float("+inf")
                for i, (val, *_) in enumerate(contours):
                    if val is not None and val < min_val:
                        min_val = val
                        i_f = i

                _contours = []
                for i, (*_, c) in enumerate(contours):
                    if i_f == i:
                        _contours.append(c)
                    else:
                        _contours.append(
                            ContourPath(
                                path=c.path,
                                is_cycle=c.is_cycle,
                                split_inds=[],
                            )
                        )
                contours = _contours
            else:
                contours = [c for *_, c in contours]

            idx2paths.append(
                Contours(
                    idx=oa.idx,
                    contours=contours,
                    n_contours=len(ucc),
                    oa=OverlapingArea(
                        idx=oa.idx,
                        poly=poly,
                        mapping=mapping,
                        t_ov_inds=t_ov_inds,
                        t_ov_inds_f=t_ov_inds_f,
                    ),
                ).sort_contours(self.vertices)
            )

        return idx2paths

    def plot(self, ax=None):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
        else:
            fig = plt.gcf()

        ax.triplot(
            self.vertices[:, 0],
            self.vertices[:, 1],
            self.triangles,
        )
        for oa in self.overlaping_areas:
            ax.triplot(
                self.vertices[:, 0],
                self.vertices[:, 1],
                self.triangles[oa.t_ov_inds],
            )

        ax.scatter(
            self.keypoints_2d[:, 0],
            self.keypoints_2d[:, 1],
            color="red",
        )

        #ax.scatter(
        #    self.vertices[[69], 0],
        #    self.vertices[[69], 1],
        #    color="black",
        #)

        if len(self.t_jun) > 0:
            ax.scatter(self.t_jun[:, 0], self.t_jun[:, 1], c="yellow")

        return fig, ax


@dataclass()
class ContourPath:
    path: list[int]
    left_t: Optional[str] = None
    right_t: Optional[str] = None
    is_cycle: bool = False
    split_inds: Optional[list[int]] = None
    m: Optional[int] = None

    def __post_init__(self):
        assert len(self.path) > 0, "Path must not be empty"
        if self.left_t is not None:
            assert self.right_t is not None
            if self.left_t != self.right_t:
                self.split_inds = None

    def split(self):
        if self.left_t != self.right_t:
            return [self]

        if self.split_inds is None:
            return [self]

        assert self.split_inds is not None
        for i in self.split_inds:
            assert 0 < i < len(self.path) - 1

        paths = np.split(self.path, self.split_inds)
        cs = []
        for i, path in enumerate(paths):
            path = path.tolist()
            if i > 0:
                path = path[1:]

            c = ContourPath(
                path=path,
                left_t=self.left_t,
                right_t=self.right_t,
                is_cycle=self.is_cycle,
            )
            cs.append(c)

        return cs


class Contours:
    def __init__(self, idx, contours, n_contours, oa):
        self.idx: int = idx
        assert len(contours) > 0, "Contours must not be empty"
        self.contours: list[ContourPath] = contours
        self.n_contours: int = n_contours
        self.oa = oa

    def __repr__(self):
        return f"\nidx={self.idx}\nnc={self.n_contours}\nc={self.contours}\n"

    def __str__(self):
        return self.__repr__()

    def sort_contours(self, vertices):
        order = [0]
        vis = set()
        js = list(range(len(self.contours)))
        while len(order) < len(js):
            path0 = self.contours[js[order[-1]]].path
            print(path0)
            vi = path0[-1]
            #if vi in vis:
            #    path0.reverse()
            #    vi = path0[0]

            assert (js[order[-1]], vi) not in vis, (vi, vis)
            vis.add((js[order[-1]], vi))
            v = vertices[vi]

            mind = float("+inf")
            vi_next = None
            j_next = None
            i = None
            for j in js:
                if j == js[order[-1]]:
                    continue

                pathj = self.contours[j].path
                print(j, pathj)

                for _i in [0, -1]:
                    _vi = pathj[_i]
                    if (j, _vi) in vis:
                        continue

                    _u = vertices[_vi]
                    d = ((v - _u) ** 2).sum()
                    if d < mind:
                        mind = d
                        j_next = j
                        vi_next = _vi
                        i = _i

            assert j_next is not None
            assert j_next not in order, (j_next, order)
            order.append(j_next)
            pathj = self.contours[j_next].path
            if i == -1:
                pathj.reverse()
                # TODO: reverse split index <21-11-23 kbrodt> #
                self.contours[j_next].split_inds = [
                    len(pathj) - 1 - s
                    for s in self.contours[j_next].split_inds
                ]

            assert pathj[0] == vi_next

            vis.add(vi_next)

        self.contours = [
            self.contours[o]
            for o in order
        ]

        return self

    def assign_contours(self):
        n = self.n_contours

        if n == 1 and self.contours[0].is_cycle:
            ff = [["ff"], ["bb"]]
        elif n % 2 == 0:
            ff = [["ff", "bb"] * (n // 2), ["bb", "ff"] * (n // 2)]
        else:
            ff = []
            for p in range(n):
                _ff = []
                for i in range(p):
                    if i % 2 == 0:
                        _ff.append("ff")
                    else:
                        _ff.append("bb")

                if p % 2 == 0:
                    _ff.append("fb")
                else:
                    _ff.append("bf")

                for i in range(p + 1, n):
                    if i % 2 == 0:
                        _ff.append("bb")
                    else:
                        _ff.append("ff")
                ff.append(_ff)

            for p in range(n):
                _ff = []
                for i in range(p):
                    if i % 2 == 0:
                        _ff.append("bb")
                    else:
                        _ff.append("ff")

                if p % 2 == 0:
                    _ff.append("bf")
                else:
                    _ff.append("fb")

                for i in range(p + 1, n):
                    if i % 2 == 0:
                        _ff.append("ff")
                    else:
                        _ff.append("bb")
                ff.append(_ff)

        for c_idxs in ff:
            contours = []
            assert len(c_idxs) == len(self.contours)
            for c_idx, c in zip(c_idxs, self.contours):
                c = copy.deepcopy(c)
                contours.append(
                    ContourPath(
                        c.path,
                        left_t=c_idx[0],
                        right_t=c_idx[1],
                        is_cycle=c.is_cycle,
                        split_inds=c.split_inds,
                    )
                )

            c, to_yield = Contours(
                idx=self.idx,
                contours=contours,
                n_contours=self.n_contours,
                oa=self.oa,
            )._set_lr()._split()

            yield c

            if to_yield and n % 2 == 0:
                contours = [
                    ContourPath(
                        c.path,
                        left_t=c.left_t,
                        right_t=c.right_t,
                        is_cycle=c.is_cycle,
                    )
                    for c in contours
                ]
                yield Contours(
                    idx=self.idx,
                    contours=contours,
                    n_contours=self.n_contours,
                    oa=self.oa,
                )._set_lr()


    def has_tri(self):
        if self.n_contours == 1:
            is_any_tri = not self.contours[0].is_cycle
        else:
            is_any_tri = self.n_contours % 2 == 1

        return is_any_tri

    def _set_lr(self):
        new_idx2path = []
        for c in self.contours:
            new_idx2path.append(
                ContourPath(
                    c.path,
                    left_t=c.left_t,
                    right_t=c.right_t,
                    is_cycle=c.is_cycle,
                    split_inds=c.split_inds,
                )
            )

        self.contours = new_idx2path

        return self

    def _split(self):
        is_split = False

        if self.has_tri():
            return self, is_split

        new_contours = []
        for c in self.contours:
            c = c.split()
            is_split = is_split or (len(c) > 1)
            new_contours.extend(c)

        self.contours = new_contours

        return self, is_split


def points(v, dv, dw, P):
    """
    Expand the vertex v to points at distance P
    :param P: expanding points as np.array (4,2)
    :param v: single vertex as np.array (2,)
    :param dv: origin as np.array (2,)
    :param dw: destiation as np.array (2,)
    :return:
    """
    j = int(np.arctan2(*dv) // (np.pi / 2))
    k = -int(-np.arctan2(*dw) // (np.pi / 2))
    return [v + P[(i + j) % 4] for i in range((k - j) % 4 + 1)]


def enhance_contour(U, P):
    """
    Expand a contour found by cv2.findContours with distances P.
    :param P: expanding points as np.array (4,2)
    :param U: single, not-closed contour with vertices as as np.array: (n,2)
    :return:
    """
    if type(U[0]) is np.intc:  # single coordinate
        return U + P

    dU = np.diff(np.vstack((U[-1], U)).T).T
    dV = np.diff(np.vstack((U, U[0])).T).T

    return np.array([p for u, du, dv in zip(U, dU, dV) for p in points(u, du, dv, P)])


def findContours(mask):
    """https://forum.opencv.org/t/findcontours-around-pixels/4702/8"""

    #contours, *_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #return contours

    #P = np.array(((-1, 0), (0, 1), (1, 0), (0, -1))) / 2  # contour through pixel edes (not stable)
    P = np.array(((-1, -1), (-1, 1), (1, 1), (1, -1))) / 2  # contour of the pixels
    U = [
        np.squeeze(c)
        for c in cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]
    ]
    U2 = [
        enhance_contour(u, P)
        for u in U
    ]

    # close contours and display
    #UC = [u if type(u[0]) is np.intc else np.vstack((u, u[0])) for u in U]
    #UC2 = [np.vstack((u, u[0])) for u in U2]

    #plt.imshow(mask)
    #for u in UC:
    #    plt.plot(*u.T, '.-k')

    #for u in UC2:
    #    plt.plot(*u.T, '.-r')

    #plt.show()

    U2 = [u[:, None] for u in U2]

    return U2


def create_segments(n):
    ran = np.arange(n)
    segments = np.stack([ran, np.roll(ran, -1)], axis=1)

    return segments


def is_unique(a):
    ua = np.unique(a, axis=0)

    return a.shape == ua.shape


def check_unique(a):
    assert is_unique(a), f"There are duplicated points ({a.shape}, {np.unique(a, axis=0).shape})"


def fix_boundary_segments(cunvis, hull_points_normalized, v, num_vertices):
    segments = create_segments(len(cunvis)) + num_vertices

    pw = cdist(cunvis, hull_points_normalized, metric="chebychev")
    i, j = np.argwhere(pw == 0).T
    check_unique(i)
    if len(j) > 0:
        segments[i, 0] = j + v

        ni = np.setdiff1d(np.arange(len(segments)), i)
        segments[ni, 0] = np.arange(len(ni)) + num_vertices
        segments[:, 1] = np.roll(segments[:, 0], -1)
        cunvis = cunvis[ni]

    return cunvis, segments


def sample_pts(kps, skeleton, n_pts=10):
    xs = []
    for a, b in skeleton:
        u = kps[a]
        v = kps[b]
        x = np.linspace(u, v, n_pts - 1, endpoint=False)[1:]
        xs.extend(x)

    xs = np.array(xs, dtype="float64")

    return xs


def generate_mesh(
    hull_points_normalized,
    predicted_keypoints_2d_start_normalized,
    overlapping_areas,
    skeleton,
    n_pts=0,
    holes=None,
    triangulate_args="pqa0.005",
    lens=None,
    to_show=False,
):
    #to_show = True
    if n_pts > 0:
        predicted_keypoints_2d_start_normalized = np.concatenate(
            [
                predicted_keypoints_2d_start_normalized,
                sample_pts(predicted_keypoints_2d_start_normalized, skeleton, n_pts=n_pts+2),
            ],
            axis=0,
        )

    points_normalized = [
        predicted_keypoints_2d_start_normalized,
        hull_points_normalized,
    ]

    check_unique(predicted_keypoints_2d_start_normalized)
    check_unique(hull_points_normalized)
    check_unique(np.concatenate(points_normalized, axis=0))

    segments = []
    segment_markers = []
    num_vertices = len(predicted_keypoints_2d_start_normalized)
    if lens is None:
        lens = [len(hull_points_normalized)]

    # boundary
    for N in lens:
        _segments = create_segments(N) + num_vertices
        segments.append(_segments)
        segment_markers.append(np.ones(N))
        num_vertices += len(_segments)

    # overlapping contours
    for overlapping_area in overlapping_areas:
        idx = overlapping_area.idx
        le = np.array([0, len(overlapping_area.contour)])

        cvu = overlapping_area.contour
        #plt.scatter(
        #    cvu[:, 0],
        #    -cvu[:, 1],
        #    s=50,
        #    c="red",
        #)

        pw = cdist(cvu, predicted_keypoints_2d_start_normalized, metric="chebychev")
        i, j = np.nonzero(pw == 0)
        #print(i, j)
        if len(i) > 0:
            predicted_keypoints_2d_start_normalized[j] += 0.05 * np.random.rand(*predicted_keypoints_2d_start_normalized[j].shape)
            #print(cvu[i])
            #print(predicted_keypoints_2d_start_normalized[j])
            #cvu[i] += 1
            #cvu_i, kps_i = np.unravel_index(pw.argmin(), pw.shape)
            #if pw[cvu_i, kps_i] == 0:
            #    raise

        cvu, _segments = fix_boundary_segments(
            cvu,
            hull_points_normalized,
            len(predicted_keypoints_2d_start_normalized),
            num_vertices,
        )

        segments.append(_segments)
        _segment_markers = np.full(le[-1], fill_value=-1)
        assert len(le) == 2
        _segment_markers[le[0]: le[1]] = 2 * idx
        segment_markers.append(_segment_markers)
        points_normalized.append(cvu)
        num_vertices += len(points_normalized[-1])
        check_unique(np.concatenate(points_normalized, axis=0))

    #plt.show()
    #plt.close()

    segments = np.concatenate(segments, axis=0)
    segment_markers = np.concatenate(segment_markers, axis=0)
    points_normalized = np.concatenate(points_normalized, axis=0)

    check_unique(points_normalized)
    assert num_vertices == len(points_normalized)

    if to_show:
        colors = ["orange", "blue", "red"]
        ms = [3, 2, 1]
        for k, s in zip(segment_markers.ravel(), segments):
            k = int(k)
            if k > 1:
                if k % 2 == 0:
                    k = 1
                else:
                    k = 2
            else:
                k = 0
            plt.plot(
                points_normalized[s, 0],
                -points_normalized[s, 1],
                color=colors[k],
            )
            #plt.plot(
            #    points_normalized[s, 0],
            #    -points_normalized[s, 1],
            #    "o",
            #    color=colors[k],
            #    markersize=ms[k],
            #)
        plt.scatter(
                predicted_keypoints_2d_start_normalized[:, 0],
                -predicted_keypoints_2d_start_normalized[:, 1],
                color="red",
        )
        plt.title("contours for triangle")
        plt.gca().set_aspect("equal")
        plt.show()
        plt.close()

    A = dict(
        vertices=points_normalized,
        segments=segments,
        segment_markers=segment_markers,  # 0 inner, 1 boundary, 2k visible, 2k + 1 unvisible
    )
    if holes is not None:
        A["holes"] = holes

    B = tr.triangulate(A, triangulate_args)
    B["triangles"] = B["triangles"][:, [0, 2, 1]]

    if to_show:
        plt.title("mesh")
        plt.triplot(B["vertices"][:, 0], -B["vertices"][:, 1], B["triangles"])
        segment_markers = B["segment_markers"].ravel()
        i = segment_markers == 1  # boundary
        S = B["segments"]
        for s in S[i]:
            plt.plot(
                B["vertices"][s, 0],
                -B["vertices"][s, 1],
                color="orange",
            )
        for idx in range(1, segment_markers.max() + 1):
            i = segment_markers == 2 * idx  # visible
            for s in S[i]:
                plt.plot(
                    B["vertices"][s, 0],
                    -B["vertices"][s, 1],
                    color="red",
                )
        vm = B["vertex_markers"].ravel()
        for s, i in enumerate(np.unique(vm)):
            plt.scatter(
                B["vertices"][vm == i, 0],
                -B["vertices"][vm == i, 1],
                label=f"{i}",
                s=max(5, 50 - 10*s),
            )

        #i = np.unique(S[i].ravel())
        #if len(i) > 0:
        #    plt.scatter(B["vertices"][i, 0], -B["vertices"][i, 1], color="orange", s=25)

        #for idx in range(1, segment_markers.max() + 1):
        #    i = segment_markers == 2 * idx  # visible
        #    i = np.unique(S[i].ravel())
        #    plt.scatter(B["vertices"][i, 0], -B["vertices"][i, 1], color="red", s=50)
        plt.gca().set_aspect("equal")
        plt.legend()
        plt.show()
        plt.close()

    return B


def pldist(point, start, end):
    """
    Calculates the distance from ``point`` to the line given
    by the points ``start`` and ``end``.

    :param point: a point
    :type point: numpy array
    :param start: a point of the line
    :type start: numpy array
    :param end: another point of the line
    :type end: numpy array
    """
    if np.all(np.equal(start, end)):
        return np.linalg.norm(point - start)

    return np.divide(
        np.abs(np.linalg.norm(np.cross(end - start, start - point))),
        np.linalg.norm(end - start),
    )


def rdp_rec(M, epsilon, dist=pldist, include=None):
    """
    Simplifies a given array of points.

    Recursive version.

    :param M: an array
    :type M: numpy array
    :param epsilon: epsilon in the rdp algorithm
    :type epsilon: float
    :param dist: distance function
    :type dist: function with signature ``f(point, start, end)`` -- see :func:`rdp.pldist`
    """
    dmax = 0.0
    index = -1
    assert len(M) > 1

    for i in range(1, M.shape[0] - 1):
        if include is not None:
            if (include == M[i]).all(-1).any(0).item():
                index = i
                dmax = None
                break

        d = dist(M[i], M[0], M[-1])

        if d > dmax:
            index = i
            dmax = d

    if dmax is None:
        rs = []
        if index > 1:
            r1 = rdp_rec(M[:index], epsilon, dist, include=include)
            rs.append(r1)
        else:
            rs.append(M[0])

        rs.append(M[index])

        if index + 2 < M.shape[0]:
            r2 = rdp_rec(M[index + 1:], epsilon, dist, include=include)
            rs.append(r2)
        else:
            rs.append(M[-1])

        return np.vstack(rs)

    if dmax > epsilon:
        r1 = rdp_rec(M[:index + 1], epsilon, dist, include=include)
        r2 = rdp_rec(M[index:], epsilon, dist, include=include)

        return np.vstack((r1[:-1], r2))
    else:
        assert not np.allclose(M[0], M[-1])
        return np.vstack((M[0], M[-1]))


def simplify_contours(contours, epsilon=2, include=None):
    # contours [[N, 1, 2]]
    # inlcude [N, 2]
    # Must be pixel by pixel (i.e. findContours flag cv2.CHAIN_APPROX_NONE)
    #return contours [[N, 1, 2]]
    #epsilon = 0
    # TODO: choose better endpoints

    if epsilon == 0:
        return contours

    ncs = []
    for contour in contours:
        if len(contour) == 1:
            ncs.append(contour)
            continue

        assert len(contour) > 1, len(contour)

        contour = rdp_rec(contour[:, 0], epsilon=epsilon, include=include)

        ncs.append(contour[:, None])

        # TODO fix this
        #pw = cdist(contour[[0, -1]], include, metric="chebychev")
        #i, j = np.argwhere(pw == 0).T
        #if len(i) == 2:
        #    ncs.append(contour[:, None])
        #elif len(i) == 1:
        #    i, = i
        #    if i == 0:
        #        ncs.append(contour[:-1, None])
        #    else:
        #        ncs.append(contour[1:, None])
        #else:
        #    ncs.append(contour[:-1, None])

    return ncs


def fit_skeleton_to_mesh(xy, skeleton):
    return skeleton
    skeleton = skeleton.copy()
    polygon = Polygon(xy).buffer(0)
    for i, pi in pose_estimation.EE_PARENTS.items():
        x, y = skeleton[i]
        px, py = skeleton[pi]
        t = 1
        while not polygon.contains(Point(x, y)):
            raise RuntimeError(f"{i} is not inside mask!")
            x, y = [
                px + t * (x - px),
                py + t * (y - py),
            ]
            #print(t, i, pi)
            t -= 0.01

        skeleton[i] = x, y

    return skeleton


def extract_continuous(i, n):
    gps = []
    gp = []
    for ix in i:
        if len(gp) == 0:
            gp.append(ix)
        elif (gp[-1] + 1) % n == ix:
            gp.append(ix)
        elif len(gp) > 0:
            gps.append(gp)
            gp = [ix]

    if len(gp) > 0:
        gps.append(gp)
        gp = []

    return gps


def assign_cont(gps_i, gps_ni):
    gps_i = [(g, True) for g in gps_i]
    gps_ni = [(g, False) for g in gps_ni]
    gps = sorted(gps_i + gps_ni, key=lambda x: x[0])

    return gps


def fix_unsorted_arr(a, c):
    if (np.sort(a) == a).all():
        return c, False

    for pos in range(len(a) - 1):
        if a[pos + 1] < a[pos]:
            break

    c = np.roll(c, -(len(a) - (pos + 1)), axis=0)

    return c, True


def simplify_common_boundary(contours, occlusion_contours, to_show=False):
    #return contours, occlusion_contours
    # contours: [ [N, 1, 2] ]
    # occlusion_contours: [ [N, 2] ]

    new_contours = []
    for c in contours:
        new_ov_contours = []
        for oc in occlusion_contours:
            pw = cdist(c[:, 0], oc, metric="chebychev")
            i, j = np.argwhere(pw == 0).T
            # TODO improve solution
            c, tc = fix_unsorted_arr(i, c)  # [45, 46, 0, 1] -> [43, 45, 46, 47]
            oc, toc = fix_unsorted_arr(j, oc)
            if tc or toc:
                pw = cdist(c[:, 0], oc, metric="chebychev")
                i, j = np.argwhere(pw == 0).T
            gps_i = extract_continuous(i, len(c))
            gps_j = extract_continuous(j, len(oc))
            #if len(gps_i) == 0:
            #    assert len(gps_j) == 0
            #    print("no contours to simplify")
            #    continue

            for _x, _y in zip(gps_i, gps_j):
                # _x = [0]  # gps_i = [0, 11]
                # _y = [6, 7]  # gps_j = [6, 7]
                if len(_x) != len(_y):
                    print(c.shape)
                    print(oc.shape)
                    print()
                    print(i)
                    print(gps_i)
                    print()
                    print(j)
                    print(gps_j)
                    print()
                    print(c[i, 0])
                    print(oc[j])
                    if to_show:
                        xy1 = np.concatenate(contours, axis=0)[:, 0]
                        xy2 = np.concatenate(occlusion_contours, axis=0)
                        plt.subplot(1, 2, 1)
                        plt.plot(xy1[:, 0], -xy1[:, 1])
                        plt.plot(xy2[:, 0], -xy2[:, 1])
                        #plt.plot(xy2[_b, 0], -xy2[_b, 1])
                        plt.plot(xy1[i, 0], -xy1[i, 1])
                        plt.plot(xy2[j, 0], -xy2[j, 1])
                        plt.show()
                        plt.close()
                    #raise

                    return contours, occlusion_contours

                assert len(_x) == len(_y), (len(_x), len(_y))  # 1, 2
                assert (c[_x, 0] == oc[_y]).all()

            ni = np.setdiff1d(np.arange(len(c)), i)
            nj = np.setdiff1d(np.arange(len(oc)), j)
            gps_ni = extract_continuous(ni, len(c))
            gps_nj = extract_continuous(nj, len(oc))
            gps_io = assign_cont(gps_i, gps_ni)
            gps_jo = assign_cont(gps_j, gps_nj)

            _a = list(range(len(c)))
            _b = list(itertools.chain(*sorted(gps_i + gps_ni, key=lambda k: k[0])))
            if _a != _b:
                print(c.shape)
                print(oc.shape)
                print()
                print(i)
                print(ni)
                print(gps_i)
                print(gps_ni)
                print(gps_io)
                print()
                print(j)
                print(nj)
                print(gps_j)
                print(gps_nj)
                print(gps_jo)
                print()
                print(i, j)
                print(c[i, 0])
                print(oc[j])

                if to_show:
                    xy1 = np.concatenate(contours, axis=0)[:, 0]
                    xy2 = np.concatenate(occlusion_contours, axis=0)
                    plt.subplot(1, 2, 1)
                    plt.plot(xy1[:, 0], -xy1[:, 1])
                    plt.plot(xy2[:, 0], -xy2[:, 1])
                    #plt.plot(xy2[_b, 0], -xy2[_b, 1])
                    plt.plot(xy2[_a, 0], -xy2[_a, 1])
                    plt.subplot(1, 2, 2)

                    plt.plot(xy1[:, 0], -xy1[:, 1])
                    plt.plot(xy2[:, 0], -xy2[:, 1])
                    plt.plot(xy2[_b, 0], -xy2[_b, 1])
                    #plt.plot(xy2[_a, 0], -xy2[_a, 1])
                    plt.show()
                    plt.close()
                #raise

                return contours, occlusion_contours

            assert _a == _b, (_a, _b)
            _a = list(range(len(oc)))
            _b = list(itertools.chain(*sorted(gps_j + gps_nj, key=lambda k: k[0])))
            if _a != _b:
                print(c.shape)
                print(oc.shape)
                print()
                print(i)
                print(ni)
                print(gps_i)
                print(gps_ni)
                print(gps_io)
                print()
                print(j)
                print(nj)
                print(gps_j)
                print(gps_nj)
                print(gps_jo)
                print()
                print(i, j)
                print(c[i, 0])
                print(oc[j])

                if to_show:
                    xy1 = np.concatenate(contours, axis=0)[:, 0]
                    xy2 = np.concatenate(occlusion_contours, axis=0)
                    plt.subplot(1, 2, 1)
                    plt.plot(xy1[:, 0], -xy1[:, 1])
                    plt.plot(xy2[:, 0], -xy2[:, 1])
                    #plt.plot(xy2[_b, 0], -xy2[_b, 1])
                    plt.plot(xy2[_a, 0], -xy2[_a, 1])
                    plt.subplot(1, 2, 2)

                    plt.plot(xy1[:, 0], -xy1[:, 1])
                    plt.plot(xy2[:, 0], -xy2[:, 1])
                    plt.plot(xy2[_b, 0], -xy2[_b, 1])
                    #plt.plot(xy2[_a, 0], -xy2[_a, 1])
                    plt.show()
                    plt.close()
                #raise

                return contours, occlusion_contours

            assert _a == _b, (_a, _b)

            new_c = []
            for g, ok in gps_io:
                if ok:
                    _c, = simplify_contours([c[g]])
                else:
                    _c = c[g]
                new_c.append(_c)

            c = np.concatenate(new_c)

            new_oc = []
            for g, ok in gps_jo:
                if ok:
                    _c, = simplify_contours([oc[g, None]])
                    _c = _c[:, 0]
                else:
                    _c = oc[g]

                new_oc.append(_c)
            oc = np.concatenate(new_oc)
            new_ov_contours.append(oc)

        new_contours.append(c)
        occlusion_contours = new_ov_contours

    return new_contours, occlusion_contours


def get_mesh_from_mask(mask, o_mask, skeleton_un, skeleton, n_pts=0, epsilon=2, buffer_eps=1e-3, to_show=False):
    #contours, *_ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = findContours(mask)
    mask_contours = np.concatenate(contours, axis=0)[:, 0]

    overlapping_areas = get_visible_unvisible_contours(
        o_mask,
        mask_contours=mask_contours,
        epsilon=epsilon,
        to_show=to_show,
    )

    if to_show:
        cont = np.zeros_like(mask)
        cv2.drawContours(cont, [c.astype("int") for c in contours], -1, (255, ))
        plt.title("simplified contours")
        plt.imshow(cont, cmap="gray")
        plt.axis("off")

    wh = max(mask.shape[:2])
    if len(overlapping_areas) == 0:
        contours = simplify_contours(contours, epsilon=epsilon)
    else:
        contours = simplify_contours(
            contours,
            epsilon=epsilon,
            include=np.concatenate(
                [
                    oa.contour  # [N, 2]
                    for oa in overlapping_areas
                ],
                axis=0,
            ),
        )

        # TODO: fix bug with extract continuous <18-11-23 kbrodt> #  <29-01-24> fixed?
        contours, occlusion_contours = simplify_common_boundary(
            contours,
            [oa.contour for oa in overlapping_areas],
        )
        assert len(occlusion_contours) == len(overlapping_areas)
        for oc, oa in zip(occlusion_contours, overlapping_areas):
            oa.contour = oc
            oa.poly = Polygon(oc).buffer(buffer_eps)

    overlapping_areas_normalized = [
        OverlapingArea(
            idx=oa.idx,
            poly=oa.poly,
            contour=inb.normalize_keypoints(oa.contour, wh),
        )#.set_contour(inb.normalize_keypoints(oa.contour, wh)
        for oa in overlapping_areas
    ]

    hull_points = np.concatenate([c[:, 0] for c in contours], axis=0)

    lens = [len(c) for c in contours]
    if to_show:
        plt.plot(hull_points[:, 0], hull_points[:, 1])
        plt.scatter(hull_points[:, 0], hull_points[:, 1])
        for oa in overlapping_areas:
            plt.plot(oa.contour[:, 0], oa.contour[:, 1])
            plt.scatter(oa.contour[:, 0], oa.contour[:, 1])

        plt.scatter(skeleton_un[:, 0], skeleton_un[:, 1])
        plt.show()
        plt.close()

    skeleton_un = fit_skeleton_to_mesh(
        sorted(contours, key=lambda x: cv2.contourArea(x.astype("int")), reverse=True)[0][:, 0],
        skeleton_un,
    )
    skeleton_normalized = inb.normalize_keypoints(skeleton_un, wh)

    hull_points_normalized = inb.normalize_keypoints(hull_points, wh)

    if len(contours) > 1:
        holes = []
        for c in sorted(contours, key=lambda x: cv2.contourArea(x.astype("int")), reverse=True)[1:]:
            pc = Polygon(c[:, 0])
            xmin, ymin, xmax, ymax = pc.bounds
            i = 0
            while True:
                i += 1
                px = np.random.uniform(xmin, xmax)
                py = np.random.uniform(ymin, ymax)
                p = Point(px, py)
                if pc.covers(p):
                    print(i, p)
                    holes.append((px, py))
                    break

        holes = np.array(holes)
        holes = inb.normalize_keypoints(holes, wh)
    else:
        holes = None

    # CHANGE BOUNDARY

    B = generate_mesh(
        hull_points_normalized,
        skeleton_normalized,
        overlapping_areas=overlapping_areas_normalized,
        skeleton=skeleton,
        n_pts=n_pts,
        holes=holes,
        #triangulate_args="pqa0.005",
        #triangulate_args="pqa0.05",
        triangulate_args="pqa1",
        lens=lens,
        to_show=to_show,
    )

    B["vertices"] = inb.denormalize_keypoints(B["vertices"], wh)
    B["overlaping_areas"] = overlapping_areas

    return B


def get_poly_cv(cv, eps=1e-3):
    # [N, 1, 2]
    cv = cv[:, 0]

    check_unique(cv)

    poly = Polygon(cv).buffer(eps)

    return poly, cv


def get_visible_unvisible_contours(
    o_mask,
    mask_contours=None,
    epsilon=2,
    to_show=False,
):
    #contours, *_ = cv2.findContours(o_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = findContours(o_mask)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x.astype("int")))
    contours = simplify_contours(contours, epsilon=epsilon, include=mask_contours)

    if to_show:
        plt.imshow(o_mask, cmap="grey")
        [plt.plot(*np.vstack((u[:, 0], u[0, 0])).T, '.-r') for u in contours]
        plt.show()
        plt.close()

    overlapping_areas = []
    for cv in contours:
        poly, _cvu = get_poly_cv(cv)
        overlapping_areas.append(
            OverlapingArea(
                idx=len(overlapping_areas) + 1,
                poly=poly,
                contour=_cvu,  # [N, 2]
            )
        )

    if len(overlapping_areas) > 0:
        check_unique(
            np.concatenate(
                [
                    oa.contour
                    for oa in overlapping_areas
                ],
                axis=0,
            )
        )

    return overlapping_areas


def get_overlapping_vertices(V, poly):
    overlapping_vertices = []
    for i, (x, y) in enumerate(V):
        #if poly.intersects(Point(x, y)):
        if intersects_xy(poly, x, y):
            overlapping_vertices.append(i)

    return overlapping_vertices


def disentangle_single_vertices(V, T, n_verts, v_ov_inds=None, vertex_to_tris=None):
    #if v_ov_inds is None:
    #    v_ov_inds = set()

    if n_verts is None:
        n_verts = len(V)

    vs = []
    newT = T.copy()
    if vertex_to_tris is None:
        vertex_to_tris = {}
        for t, (i, j, k) in enumerate(T):
            if i < n_verts:
                vertex_to_tris.setdefault(i, set()).add(t)
            if j < n_verts:
                vertex_to_tris.setdefault(j, set()).add(t)
            if k < n_verts:
                vertex_to_tris.setdefault(k, set()).add(t)

    newV = []
    for v, ts in vertex_to_tris.items():
        assert v < n_verts
        #if v >= n_verts:
        #    continue

        tsc = ts.copy()
        batterfly = []
        while tsc:
            t = tsc.pop()
            b = [t]
            while True:
                to_remove = None
                for b_ in b:
                    b_ = set(T[b_])
                    for p in tsc:
                        if len((b_ & set(T[p])) - {v}) > 0:
                            to_remove = p
                            break

                    if to_remove is not None:
                         break
                else:
                    assert to_remove == None
                    break

                tsc.remove(to_remove)
                b.append(to_remove)

            batterfly.append(b)

        if len(batterfly) != 2:
            continue

        t1, t2 = batterfly
        t12 = set(itertools.chain(*[T[t] for t in t1])) & set(itertools.chain(*[T[t] for t in t2]))
        assert len(t12) == 1

        t = t12.pop()
        assert t == v
        for t in t1:
            newT[t][T[t] == v] = len(V) + len(newV)

        #if v in v_ov_inds:
        #    continue

        vs.append(v)
        newV.append(V[v])

    newV = np.array(newV)
    if len(newV) > 0:
        V = np.vstack([V, newV])

    return V, newT, vs


def check_VT(V, T):
    t = sorted(set(itertools.chain(*T)))
    v = list(range(len(V)))
    return v == t, (len(v), len(t), set(v) - set(t))


def get_overlapping_triangles(V, T, poly):
    T_ov_inds = []
    for i, t in enumerate(T):
        triangle = Polygon(V[t])
        if poly.contains(triangle):
            T_ov_inds.append(i)

    return T_ov_inds


def copy_overlapping_mesh(V, T, T_ov_inds, unvis):
    T_ov = T[T_ov_inds].copy()
    v_ov_inds = set(T_ov.ravel())
    v_ov_inside_inds = sorted(v_ov_inds - set(unvis))

    mapping = dict(zip(v_ov_inside_inds, len(V) + np.arange(len(v_ov_inside_inds))))
    T_ov = np.array(
        [
            [mapping.get(v, v) for v in t]
            for t in T_ov
        ]
    )
    #assert sorted(set(T_ov.ravel() - len(V))) == list(range(len(v_ov_inside_inds))), f'gg {len(set(T_ov.ravel() - len(V)))}, {len(v_ov_inside_inds)}'
    V = np.concatenate([V, V[v_ov_inside_inds]], axis=0)
    T_ov_inds = np.arange(len(T_ov)) + len(T)
    T = np.concatenate([T, T_ov], axis=0)

    ok, d = check_VT(V, T)
    if not ok:
        _, _, d = d
        d = list(d)
        plt.triplot(V[:, 0], -V[:, 1], T)
        plt.scatter(V[v_ov_inside_inds, 0], -V[v_ov_inside_inds, 1], c="red", s=15)
        plt.scatter(V[d, 0], -V[d, 1], color="black", s=50)
        plt.show()
        plt.close()
        raise

    return V, T, mapping, T_ov_inds


def find_joints_inside_poly(skeleton, poly):
    joint_ids = []
    for i, (x, y) in enumerate(skeleton):
        if poly.contains(Point(x, y)):
            joint_ids.append(i)

    return joint_ids


def find_bones_intersect_poly(skeleton, poly, skeleton_data):
    bone_inds = []
    for i, (a, b) in enumerate(skeleton_data.skeleton):
        a = skeleton[a]
        b = skeleton[b]
        if poly.intersects(LineString([a, b])):
            bone_inds.append(i)

    if len(bone_inds) == 0:
        b_s = []
        for i, (a, b) in enumerate(skeleton_data.skeleton):
            a = skeleton[a]
            b = skeleton[b]
            distance_between_pts = poly.distance(LineString([a, b]))
            b_s.append((distance_between_pts, i))

        b_s = sorted(b_s, key=lambda x: x[0])
        r = l = False
        for i in range(len(b_s)):
            if r and l:
                break

            _, i = b_s[i]
            if any(skeleton_data.joints[a].startswith("Right") for a in skeleton_data.skeleton[i]):
                r = True
                bone_inds.append(i)
            if any(skeleton_data.joints[a].startswith("Left") for a in skeleton_data.skeleton[i]):
                l = True
                bone_inds.append(i)

    return bone_inds


def create_overlapping_mesh(B, skeleton, bone_to_top, t_jun, skeleton_data, n_pts=1):
    V, T = B["vertices"], B["triangles"]
    segment_markers = B["segment_markers"].ravel()
    S = B["segments"]
    bnd_edges = S[segment_markers == 1]
    bnd = np.unique(bnd_edges.ravel())

    gbnd = nx.Graph()
    gbnd.add_edges_from(bnd_edges)

    oas = []
    for oa in B["overlaping_areas"]:
        # TODO remove segment marksers use poly
        T_ov_inds = get_overlapping_triangles(V, T, oa.poly)
        assert len(T_ov_inds) > 0

        contour = igl.boundary_loop(T[T_ov_inds])

        g = gbnd.copy()
        g.add_edges_from(itertools.pairwise(contour))
        g.add_edge(contour[-1], contour[0])
        to_remove = []
        for v in bnd:
            if g.degree[v] == 2:
                to_remove.append(v)

        for v in to_remove:
            g.remove_node(v)

        for e in gbnd.edges:
            if g.has_edge(*e):
                g.remove_edge(*e)

        edges = np.array(g.edges)

        V, T, mapping, T_ov_inds_f = copy_overlapping_mesh(
            V,
            T,
            T_ov_inds,
            edges.ravel(),
        )

        oas.append(
            OverlapingArea(
                idx=oa.idx,
                poly=oa.poly,
                mapping=mapping,
                t_ov_inds=T_ov_inds,
                t_ov_inds_f=T_ov_inds_f,
                edges=edges,
            )
        )

    omesh = OverlapingMesh(
        vertices=V,
        triangles=T,
        overlaping_areas=oas,
        keypoints_2d=skeleton,
        bone_to_top=bone_to_top,
        t_jun=t_jun,
        n_sample_pts=n_pts,
        bnd=bnd,
        skeleton_data=skeleton_data,
    )

    return omesh


def get_overlapping_mesh(
    mask,
    o_mask,
    skeleton,
    bone_to_top,
    t_jun,
    skeleton_data,
    n_pts=1,
    epsilon=2,
    to_show=False,
):
    B = get_mesh_from_mask(
        mask,
        o_mask,
        skeleton,
        skeleton_data.skeleton,
        n_pts=n_pts,
        to_show=to_show,
        epsilon=epsilon,
    )

    omesh = create_overlapping_mesh(
        B,
        skeleton,
        bone_to_top,
        t_jun,
        skeleton_data=skeleton_data,
        n_pts=n_pts,
    )

    return omesh
