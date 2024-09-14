import os

os.environ["HSA_OVERRIDE_GFX_VERSION"] = "9.0.0"  # for AMD GPU
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["PYOPENGL_PLATFORM"] = "egl"  # "osmesa"

import argparse
import itertools
import math
import random
import shlex
import subprocess
import textwrap
import time
from distutils.spawn import find_executable
from pathlib import Path

import cv2

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import PIL.Image as Image
import igl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import optuna
import pyrender
import torch
import triangle as tr
import trimesh
from pyrender import Sampler
from pyrender.constants import GLTF
from pyrender.shader_program import ShaderProgramCache
from shapely.geometry import (
    Point,
    LineString,
    Polygon,
)
from scipy.spatial.distance import cdist
from scipy.spatial.transform import (
    Rotation,
    Slerp,
)
from skimage.measure import (
    label,
    regionprops,
)
from torch.nn import functional as F
from simple_lama_inpainting import SimpleLama

import deform.kvf
import dino_inference
import draw
import losses
import mesh_utils
import pose_estimation
import segment
import utils
import write_utils
from skeleton import (
    get_fk,
    Skeleton2d,
    Skeleton3d,
)
try:
    from animated_drawings.model.bvh import BVH
except ModuleNotFoundError as e:
    print(textwrap.dedent(
            f"""
            {e}.
            To read bvh animation, create a symbolic link to
            `animated_drawings`. directory from
            `https://github.com/facebookresearch/AnimatedDrawings` in `src`
            directory:
                `ln -s ABS_PATH_TO_ANIMATED_DRAWINGS/animated_drawings ./src`
            """
        )
    )

import biskin
import fastSymDir
try:
    from train_log.RIFE_HDv3 import Model
except ModuleNotFoundError as e:
    print(textwrap.dedent(
        f"""
        {e}.
        Required with the `--interactive NUM` key. For interactive
        manipulations, create a symbolic link to the `model` and `trian_log`
        directories from `https://github.com/hzwer/ECCV2022-RIFE` in `src`
        directory:
            `ln -s ABS_PATH_TO_RIFE/train_log ./src`
            `ln -s ABS_PATH_TO_RIFE/model ./src`
        """
        )
    )

GIMP = "gimp"
GIMP_EXE = find_executable(GIMP)
if not GIMP_EXE:
    print(textwrap.dedent(
        f"""
        `{GIMP}` not found. Install `{GIMP}` or other program for manual
        annotation or provide masks with $IMG_mask.$EXT and
        $IMG_occlusion_mask_pred.$EXT file names.
        """
        )
    )


def parse_args(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--img-paths",
        nargs="+",
        type=str,
        required=True,
        help="path to images",
    )
    parser.add_argument(
        "--guidance-paths",
        nargs="+",
        type=str,
        required=False,
        help="path to guidance images",
    )
    parser.add_argument(
        "--animation-paths",
        nargs="+",
        type=str,
        required=False,
        default=None,
        help="path to bvh animations",
    )

    parser.add_argument(
        "--frame-interpolation-model-path",
        type=str,
        default="./src/train_log",
        help="frame interpolation model path",
    )

    parser.add_argument(
        "--character-topology-path",
        type=str,
        default="./characters_topology/human_topology.json",
        help="path to charachter topology",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="path to save results",
    )

    parser.add_argument(
        "--pose-estimation-model-path",
        type=str,
        default=f"{Path('./models').resolve()}/hrn_w48_384x288.onnx",
        help="pose estimation model",
    )
    parser.add_argument(
        "--segmentation-model-path",
        type=str,
        default=f"{Path('./models').resolve()}/model_14_2.pth",
        help="pose estimation model",
    )

    parser.add_argument(
        "--n-steps",
        type=int,
        default=24,
        required=False,
        help="number of inbetweens",
    )

    parser.add_argument(
        "--n-pts",
        type=int,
        default=1,
        required=False,
        help="number of sampled points along the bone",
    )

    parser.add_argument(
        "--use-o-mask-gt",
        required=False,
        action="store_true",
        help="use gt occlusion mask",
    )

    parser.add_argument(
        "--inpaint-method",
        type=str,
        default="copy",
        required=False,
        choices=["copy", "cv2", "lama"],
        help="inpaint method",
    )

    parser.add_argument(
        "--deform-method",
        type=str,
        default="dirichlet",
        required=False,
        choices=["arap", "dirichlet", "kvf"],
        help="defomation method",
    )

    parser.add_argument(
        "--to-show",
        required=False,
        action="store_true",
        help="debug",
    )

    parser.add_argument(
        "--no-optuna",
        required=False,
        action="store_true",
        help="optuna optimization",
    )

    parser.add_argument(
        "--use-nearest",
        required=False,
        action="store_true",
        help="use nearest sampler instead of linear in gl",
    )

    parser.add_argument(
        "--interactive",
        required=False,
        type=int,
        default=0,
        help="intercative",
    )

    parser.add_argument(
        "--epsilon",
        type=float,
        default=2,
        required=False,
        help="simplify contours",
    )

    parser.add_argument(
        "--touch-pixels",
        type=int,
        default=2,
        required=False,
        help="touching pixels",
    )

    parser.add_argument(
        "--seed",
        required=False,
        type=int,
        default=31459,
        help="random seed",
    )

    args = parser.parse_args(args)

    return args


def normalize_keypoints(kpts, wh, root=0):
    kpts = kpts - root
    kpts = 2 * kpts / wh

    return kpts


def denormalize_keypoints(kpts, wh, root=0):
    kpts = wh * kpts / 2
    kpts = kpts + root

    return kpts


def get_uv_param(verts_normalized, root, wh, uv_wh):
    uv_param = denormalize_keypoints(verts_normalized, wh, root)
    uv_param = np.stack(
        [
            uv_param[:, 0] / uv_wh,
            1 - uv_param[:, 1] / uv_wh,
        ]
    ).T

    return uv_param


def get_trimesh(v, t, uv, img):
    h, w = img.shape[:2]
    assert h == w
    visual = trimesh.visual.texture.TextureVisuals(
        uv=uv,
        image=Image.fromarray(img),
    )
    mesh = trimesh.Trimesh(
        np.pad(v, ((0, 0), (0, 1))),
        t,
        visual=visual,
        process=False,
        validate=False,
    )

    return mesh


def get_pad_center(t, wh):
    d = (2 * t - wh).round().astype("int")
    x, y = t
    w, h = wh
    if x >= w // 2:  # x
        if y >= h // 2:  # y
            assert (d >= 0).all()
            pad = (
                [0, d[1]],
                [0, d[0]],
            )
        else:
            assert d[1] <= 0, d[1]
            assert d[0] >= 0
            pad = (
                [abs(d[1]), 0],
                [0, d[0]],
            )
    else:
        if y >= h // 2:  # y
            assert d[1] >= 0
            assert d[0] <= 0
            pad = (
                [0, d[1]],
                [abs(d[0]), 0],
            )
        else:
            assert (d <= 0).all()
            pad = (
                [abs(d[1]), 0],
                [abs(d[0]), 0],
            )

    return pad


def center_image(img, root, mod=4, mode="symmetric"):
    #print(f"Original {img.shape=}")
    wh_img = np.array(img.shape[:2][::-1])
    root = root.round()
    pad = get_pad_center(root, wh_img)
    pad_ch = ((0, 0), ) * (img.ndim == 3)
    img = np.pad(
        img,
        pad + pad_ch,
        #mode="edge",
        mode=mode,
    )
    #print(f"After center {img.shape=}")

    h, w = img.shape[:2]
    a = abs(h - w) // 2
    b = abs(h - w) % 2
    if h > w:
        img = np.pad(
            img,
            ((0, 0), (a, a + b)) + pad_ch,
            #mode="edge",
            mode=mode,
        )
    elif h < w:
        img = np.pad(
            img,
            ((a, a + b), (0, 0)) + pad_ch,
            #mode="edge",
            mode=mode,
        )

    #print(f"After square {img.shape=}")

    h, w = img.shape[:2]
    assert h == w
    wh = h
    wh_img = max(wh_img)
    assert wh >= wh_img, f"Centered image must not be less then original. {wh=} {wh_img}"
    #print(f"{wh=} {wh_img=}")

    # https://www.google.com/search?q=texture+is+divisible+by+4&newwindow=1&sca_esv=596270913&sxsrf=AM9HkKlax1j7kSewy-M18MD8ntHc-XlwtA%3A1704585258319&ei=KuiZZdjzEs6f5NoPw4WIiA8&ved=0ahUKEwjYsL72-smDAxXOD1kFHcMCAvEQ4dUDCBA&uact=5&oq=texture+is+divisible+by+4&gs_lp=Egxnd3Mtd2l6LXNlcnAiGXRleHR1cmUgaXMgZGl2aXNpYmxlIGJ5IDRI2ylQoQRYjSdwAXgBkAEAmAFjoAGXCKoBAjEyuAEDyAEA-AEBwgIEECMYJ8ICBxAjGLACGCfiAwQYACBBiAYB&sclient=gws-wiz-serp
    # https://stackoverflow.com/questions/57346317/opengl-greyscale-texture-data-layout-doesnt-match-when-the-dimensions-arent-d
    # https://support.lumion.com/hc/en-us/articles/7764034284188-What-do-the-properties-of-the-Standard-Material-mean-in-Lumion-2023-
    # https://www.khronos.org/opengl/wiki/Texture
    if wh_img % mod != 0:
        wh_img = (wh_img // mod + 1) * mod

    d = wh - wh_img
    if d > 0:
        __wh = (wh % mod)
        _wh = __wh // 2
        img = np.pad(
            img,
            ((_wh, _wh + (__wh % 2)), (_wh, _wh + (__wh % 2))) + pad_ch,
            mode=mode,
            #mode="edge",
        )
        #a = d // 2  # 8 (20), 4(25)
        #b = d % 2 # 8 (20), 4(25)
        #img = img[a:-a - b, a:-a - b].copy()
        #print(f"After crop to {mod} {img.shape=}")
    elif d < 0:
        new_h = (wh // mod + 1) * mod
        d = new_h - wh
        a = d // 2  # 8(20), 4(25)
        b = d % 2  # 8(20), 4(25)
        img = np.pad(
            img,
            ((a, a + b), (a, a + b)) + pad_ch,
            mode=mode,
            #mode="edge",
        )
        #print(f"After pad to {mod} {img.shape=}")

    #assert img.shape[0] == img.shape[1] == wh_img, (img.shape, wh_img)
    #print(f"Final {img.shape=}")
    _wh = 16
    img = np.pad(
        img,
        ((_wh, _wh), (_wh, _wh)) + pad_ch,
        mode=mode,
        #mode="edge",
    )

    return img, wh_img


def copy_overlapping_mesh_hidden(V, T, T_ov_inds, unvis):
    #unvis = sorted(set(unvis))
    mesh_utils.check_unique(np.array(unvis))
    unvis = list(unvis)
    assert sorted(set(unvis)) == sorted(unvis)
    mapping = dict(zip(unvis, len(V) + np.arange(len(unvis))))
    T = T.copy()
    for t in T_ov_inds:
        T[t, 0] = mapping.get(T[t, 0], T[t, 0])
        T[t, 1] = mapping.get(T[t, 1], T[t, 1])
        T[t, 2] = mapping.get(T[t, 2], T[t, 2])

    return unvis, T, mapping


def bones_to_verts_hierarchy(bones, skeleton_data):
    if skeleton_data.kps_to_hier is None:
        return

    vinds = []
    for i in bones:
        a, b = skeleton_data.skeleton[i]
        vinds.extend(
            [
                skeleton_data.joints.index(i)
                for i in skeleton_data.kps_to_hier[skeleton_data.joints[a]]
            ]
        )
        vinds.extend(
            [
                skeleton_data.joints.index(i)
                for i in skeleton_data.kps_to_hier[skeleton_data.joints[b]]
            ]
        )

    vinds = sorted(set(vinds))

    return vinds


def cut_mesh(vertices, triangles, T_ov_inds, path, n_verts, bnd, to_show=False):
    # cut mesh from boundary (path[0] must lie on the boundary)
    if len(path) < 2:
        return vertices, triangles, [], {}

    assert len(path) > 1, "cut mesh requires more than 1 edges"
    path_orig = path

    # if last on boundary, then don't remove, if not remove?
    #bnd = igl.boundary_loop(triangles)
    is_removed = False
    if path[-1] not in bnd:
        is_removed = True
        path = path[:-1]

    T_adj_inds = []  # list of adjacent triangles' indices from `T_ov_inds` to path `path`
    for t in T_ov_inds:
        for i in triangles[t]:
            if i in path:
                T_adj_inds.append(t)

    if to_show:
        plt.title("cut mesh")
        plt.triplot(vertices[:, 0], -vertices[:, 1], triangles)
        plt.triplot(vertices[:, 0], -vertices[:, 1], triangles[T_ov_inds])
        plt.scatter(
            vertices[path_orig, 0],
            -vertices[path_orig, 1],
            color="red",
            s=100,
            label="path to cut",
        )
        if len(T_adj_inds) > 0:
            plt.scatter(
                vertices[triangles[np.unique(T_adj_inds)], 0],
                -vertices[triangles[np.unique(T_adj_inds)], 1],
                color="cyan",
                s=50,
                label="adjacent vertices to cut",
            )

    # create new vertices along path `path` and change indices in triangles
    v_ov_inds, triangles, mapping = copy_overlapping_mesh_hidden(
        vertices, triangles, T_adj_inds, path,
    )

    if to_show:
        plt.scatter(
            vertices[v_ov_inds, 0],
            -vertices[v_ov_inds, 1],
            color="blue",
            s=25,
            label="cut",
        )
        #plt.show()
        #plt.close()

    vertices = np.concatenate(
        [
            vertices,
            vertices[v_ov_inds],
        ],
        axis=0,
    )
    # TODO: fix new veritex indices update mapping!!!
    _max_v = len(vertices)
    V, T, v_ov_inds_d = mesh_utils.disentangle_single_vertices(
        vertices,
        triangles,
        n_verts=n_verts,
        #v_ov_inds=v_ov_inds,
    )
    if not is_removed:
        assert len(v_ov_inds_d) == 0, (len(v_ov_inds_d), v_ov_inds_d)

    max_v = max(mapping.values()) + 1
    assert _max_v == max_v
    # if type is "f" and "b" and the origin is in one point, then it is not the case!!!
    assert len(set(v_ov_inds_d) & set(mapping)) == 0, (set(v_ov_inds_d) & set(mapping), v_ov_inds_d)
    for i, nv in enumerate(v_ov_inds_d):
        mapping[nv] = max_v + i

    v_ov_inds.extend(v_ov_inds_d)
    vertices = V
    triangles = T

    ok, d = mesh_utils.check_VT(vertices, triangles)
    if not ok:
        _, _, d = d
        d = list(d)
        if not to_show:
            plt.triplot(vertices[:, 0], -vertices[:, 1], triangles)
            plt.triplot(vertices[:, 0], -vertices[:, 1], triangles[T_ov_inds])
            plt.scatter(vertices[path_orig, 0], -vertices[path_orig, 1], color="red", s=100)
            if len(T_adj_inds) > 0:
                plt.scatter(
                    vertices[triangles[np.unique(T_adj_inds)], 0],
                    -vertices[triangles[np.unique(T_adj_inds)], 1],
                    color="cyan",
                    s=50,
                )

        plt.scatter(vertices[d, 0], -vertices[d, 1], color="black", s=50)
        plt.show()
        plt.close()
        raise

    if to_show:
        plt.gca().set_aspect("equal")
        plt.legend()
        plt.show()
        plt.close()

    return vertices, triangles, v_ov_inds, mapping


def plot_skel_cv(img, joints=None, skeleton=None, t_jun=None, joint_names=None):
    blue_rgb = (85, 153, 255)
    red_rgb = (255, 85, 85)
    blue = blue_rgb[::-1]
    red = red_rgb[::-1]
    #h, w = img.shape[:2]
    pt_size = int(0.005 * max(img.shape[:2]))
    if joints is not None:
        for x, y in joints:
            x = int(x)
            y = int(y)
            #y = h - int(y)
            cv2.circle(img, (x, y), pt_size, blue, lineType=cv2.LINE_AA, thickness=cv2.FILLED)

        if skeleton is not None:
            for a, b in skeleton:
                if joint_names is not None and "right" in joint_names[a].lower():
                    color = blue#red
                else:
                    color = blue

                a = joints[a]
                #a = [a[0], h - a[1]]
                b = joints[b]
                #b = [b[0], h - b[1]]
                cv2.line(img, a, b, color, lineType=cv2.LINE_AA)

    if t_jun is not None and len(t_jun) > 0:
        for x, y in t_jun:
            #y = h - y
            cv2.circle(img, (x, y), pt_size, red, lineType=cv2.LINE_AA, thickness=cv2.FILLED)


class SkeletonData:
    def __init__(self, joints, root, skeleton, parents2d, parents2d_to_kps, names, zero_inds, dj, type, kps_to_hier=None):
        self.joints = joints

        self.root_name = root
        self.root = self.joints.index(self.root_name)
        self.skeleton = skeleton

        self.right_inds = [
            i
            for i, (a, _) in enumerate(self.skeleton)
            if "right" in self.joints[a].lower()
        ]

        self.parents2d = parents2d
        self.parents2d_to_kps = parents2d_to_kps
        self.names = names
        assert len(self.parents2d) == len(self.parents2d_to_kps) == len(self.names)

        self.zero_inds = [self.names.index(i) for i in zero_inds]

        self.dj = dj

        self.type = type

        self.kps_to_hier = kps_to_hier

        self.kps_to_skeleton = [
            [
                j
                for j, (a, b) in enumerate(self.skeleton)
                if self.joints[a] == kp or self.joints[b] == kp
            ]
            for kp in self.joints
        ]

        self.g = nx.Graph()
        self.g.add_edges_from(self.skeleton)

        self.end_effectors_inds = [n for n in self.g.nodes if self.g.degree[n] == 1]
        print(f"{self.end_effectors_inds=}")

        self.adjacent = [n for n in self.g.nodes if self.g.degree[n] > 2]
        print(f"{self.adjacent=}")

        self.end_effector2par = {}
        if len(self.adjacent) > 0:
            self.end_effector2par = {
                v: min(
                    (
                        nx.shortest_path(self.g, v, t)
                        for t in self.adjacent
                    ),
                    key=len,
                )[1:]
                for v in self.end_effectors_inds
            }

        self.manifold_joints = [
            nx.shortest_path(self.g, v, ee)[1]
            for v in self.adjacent
            for ee in self.end_effectors_inds
        ]
        self.manifold_joints = set(
            p
            for p in self.manifold_joints
            if self.is_symmetric_joint(p)
        )
        self.manifold_bones = set(
            self.get_bone(v, u)
            for v in self.manifold_joints
            for u in self.g.neighbors(v)
        )

    def is_symmetric_joint(self, joint_ind):
        joint = self.joints[joint_ind].lower()

        return "left" in joint or "right" in joint

    def get_symmetric_joint(self, joint_ind):
        if not self.is_symmetric_joint(joint_ind):
            return joint_ind

        joint = self.joints[joint_ind].lower()
        if "left" in joint:
            joint_sym = self.joints[joint_ind].lower().replace("left", "right")
        else:
            joint_sym = self.joints[joint_ind].lower().replace("right", "left")

        joint_sym_ind, = [i for i, joint in enumerate(self.joints) if joint.lower == joint_sym]

        return joint_sym_ind

    def get_bone(self, a, b):
        try:
            b = self.skeleton.index([a, b])
        except ValueError:
            b = self.skeleton.index([b, a])

        return b

    def is_symmetric_bone(self, a, b):
        return self.is_symmetric_joint(a) or self.is_symmetric_joint(b)

    @classmethod
    def from_json(cls, path):
        meta = write_utils.load_json(path)

        c = cls(
            joints=meta["joints"],
            root=meta["root"],
            skeleton=meta["skeleton"],
            parents2d=meta["parents2d"],
            parents2d_to_kps=meta["parents2d_to_kps"],
            names=meta["names"],
            zero_inds=meta["zero_inds"],
            dj=meta["dj"],
            type=meta["type"],
            kps_to_hier=meta.get("kps_to_hier", None),
        )

        return c


def inpaint_lama(texture, mask):
    if not mask.any():
        return texture

    uv_wh = texture.shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    simple_lama = SimpleLama(device)
    image = Image.fromarray(texture)
    mask = Image.fromarray(mask).convert("L")
    result = simple_lama(image, mask)
    texture = np.array(result)
    if texture.shape[0] != uv_wh:  # w
        texture = texture[:uv_wh, :uv_wh].copy()

    return texture


class Inbetweener:
    def __init__(
        self,
        skeleton_data,
        img,
        mask,
        o_mask,
        keypoints_2d,
        z_order,
        bone_to_top,
        t_jun,
        inpaint_method="copy",
        epsilon=2,
        n_pts=1,
        to_show=False,
        device="cpu",
        nearest=False,
    ):
        self.skeleton_data = skeleton_data
        self.nearest = nearest
        self.inpaint_method = inpaint_method
        self.img = img
        self.mask = mask
        self.root = keypoints_2d[self.skeleton_data.root].copy()
        self.wh = max(self.img.shape[:2])
        # todo enlarge texture due to croping some body parts
        self.texture, self.wh_skeleton = center_image(self.img, self.root)
        assert self.texture.shape[0] == self.texture.shape[1], f"Texture must be squared. {self.texture.shape=}"
        assert self.texture.shape[0] % 4 == 0, f"Texture must be divisible by 4. {self.texture.shape=}"

        self.o_mask_texture, _ = center_image(o_mask, self.root)

        self.uv_wh = max(self.texture.shape[:2])

        self.z_order = z_order
        self.epsilon = epsilon

        self.omeshes = []
        self.omesh_ind = 0
        for _t_jun in [[]]:
            if len(_t_jun) == 0:
                _o_mask = o_mask
            else:
                raise

            self.omeshes.append(
                mesh_utils.get_overlapping_mesh(
                    mask,
                    _o_mask,
                    keypoints_2d,
                    bone_to_top,
                    t_jun,
                    skeleton_data=self.skeleton_data,
                    n_pts=n_pts,
                    epsilon=self.epsilon,
                    to_show=to_show,
                )
            )

        self._set_vt()

        self.device = device
        self._path = {"visible": [], "hidden": []}
        self.saved_mesh: list = []

        #self.render(
        #    out_path=Path("./TEST/c"),
        #)
        #raise

    def _set_vt(self):
        self.n_verts = len(self.omesh.vertices)
        self.vertices = normalize_keypoints(
            self.omesh.vertices,
            self.wh,
            root=self.root,
        )
        self.triangles = self.omesh.triangles.copy()
        self.mapping = self.omesh.mapping.copy()
        self.set_trimesh(self.vertices)
        self._recalculate_skinning(force_rewrite=True)

    @property
    def omesh(self):
        return self.omeshes[self.omesh_ind]

    @property
    def skeleton(self):
        return self.vertices[self.omesh.vert_to_skel].copy()

    @skeleton.setter
    def skeleton(self, skeleton):
        self.vertices[self.omesh.vert_to_skel] = skeleton.copy()

    def set_trimesh(self, vertices):
        self.vertices = vertices.copy()
        self.uv = get_uv_param(
            self.vertices,
            #root=[self.root[0] - self.uv_wh // 2, self.root[1] - self.uv_wh // 2],
            #root=self.root,
            root=[self.uv_wh // 2 + 0.5, self.uv_wh // 2 + 0.5],
            wh=self.wh,
            uv_wh=self.uv_wh,
        )
        self.mesh = get_trimesh(
            self.vertices,
            self.triangles,
            self.uv,
            self.texture,
        )

    def prepare_mesh_for_render(self, deforms=None, out_path=None, interactive=False, rotpi=False):
        mesh = self.mesh.copy()
        if deforms is None:
            deforms = np.zeros_like(mesh.vertices)

        if deforms.shape[-1] != 3:
            #deforms = np.pad(deforms, ((0, 0), (0, 1)))
            deforms = self._add_z(deforms, 1, np.zeros(len(deforms)))

        vert_o = mesh.vertices.copy()

        mesh.vertices = mesh.vertices + deforms

        if not interactive:
            tr = trimesh.transformations.translation_matrix(-mesh.vertices[self.omesh.vert_to_skel[self.skeleton_data.root]])
            mesh.apply_transform(tr)

        rot = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
        if rotpi:
            mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [0, 1, 0]))
            mesh.invert()

        if out_path is not None:
            m = trimesh.Trimesh(
                mesh.vertices,
                mesh.faces,
                process=False,
                validate=False,
            )

            color = np.array(m.visual.vertex_colors)
            color[:] = [233, 233, 233, 255]
            red = [255, 0, 0, 255]
            #color[self.omesh.vert_to_skel] = red

            if hasattr(self, "_path"):
                green = [0, 255, 0, 255]
                blue = [0, 0, 255, 255]
                for ci, _path in self._path.items():
                    if len(_path) > 0:
                        if ci == "visible":
                            color[_path] = green
                        elif ci == "hidden":
                            color[_path] = blue

            face_colors = np.full(
                (len(self.triangles), 4),
                fill_value=233,
                dtype="uint8",
            )
            face_colors[:, 3] = 255
            red = np.full(
                (len(self.triangles), 3),
                fill_value=0,
                dtype="uint8",
            )
            red[:, 0] = 255

            energy = self._verts_inds_bad_quality(
                mesh.vertices,
                mesh.faces,
                V_old=vert_o,
            )
            np.savetxt(out_path.with_name(f"{out_path.stem}_path_deformation"), energy)
            x = np.clip((energy - 4.0) / 0.5, 0, 1)
            x = x[:, None]  # [N, 1]
            face_colors[:, :3] = (
                (1 - x) * face_colors[:, :3] + x * red
            ).astype("uint8")

            m.visual.face_colors = face_colors
            m.apply_transform(rot)
            m.export(out_path.with_suffix(".ply"))

            #m.visual.face_colors = None
            m.visual.vertex_colors = color
            m.export(out_path.with_name(f"{out_path.stem}_path.ply"))

        mesh.apply_transform(rot)
        if out_path is not None:
            mesh.export(out_path.with_suffix(".glb"))

        return mesh

    def render(
        self,
        deforms=None,
        out_path=None,
        bg_color=None,
        viewport_width=512,
        viewport_height=512,
        uv=None,
        texture=None,
        layer=None,
        smooth=False,
        interactive=False,
        rotpi=False,
    ):
        if out_path is not None:
            if isinstance(out_path, str):
                out_path = Path(out_path)

        mesh = self.prepare_mesh_for_render(
            deforms=deforms,
            out_path=out_path,
            interactive=interactive,
            rotpi=rotpi,
        )

        #vertices = mesh.vertices
        #assert np.allclose(
        #    vertices[self.omesh.vert_to_skel[9]][:2],
        #    [0, 0],
        #), (vertices[self.omesh.vert_to_skel[9]][:2], "hips must be centered")

        mesh = pyrender.Mesh.from_trimesh(
            mesh,
            smooth=smooth,
        )
        # https://github.com/mmatl/pyrender/issues/51
        for primitive in mesh.primitives:
            primitive.material.baseColorFactor = [1., 1., 1., 1.]
            if self.nearest and primitive.material.baseColorTexture is not None:
                primitive.material.baseColorTexture.sampler = Sampler(
                    minFilter=GLTF.NEAREST,
                    magFilter=GLTF.NEAREST,
                )

            if layer is not None:
                assert uv is not None
                if not smooth:
                    indices = self.triangles
                    uv = uv[indices].reshape((3 * len(indices), uv.shape[1]))
                    layer = np.repeat(layer, 3, axis=0)

            if uv is not None:
                primitive.color_0 = layer

                primitive.texcoord_1 = uv
                primitive.material.emissiveTexture = texture
                if self.nearest:
                    primitive.material.emissiveTexture.sampler = Sampler(
                        minFilter=GLTF.NEAREST,
                        magFilter=GLTF.NEAREST,
                    )

        scene = pyrender.Scene(
            bg_color=bg_color,
            ambient_light=(1.0, 1.0, 1.0),
        )
        scene.add(mesh, "mesh")

        camera_pose = np.eye(4)
        camera_pose[2, -1] = 1
        #assert self.uv_wh / self.wh == 1
        pyrencamera = pyrender.camera.OrthographicCamera(
            self.uv_wh / self.wh,
            self.uv_wh / self.wh,
        )
        scene.add(pyrencamera, pose=camera_pose)

        r = pyrender.OffscreenRenderer(
            #viewport_width=viewport_width,  # self.uv_wh,
            #viewport_height=viewport_height,  # self.uv_wh,
            viewport_width=self.uv_wh,
            viewport_height=self.uv_wh,
        )
        # https://github.com/mmatl/pyrender/issues/39
        # https://learnopengl.com/Getting-started/Textures
        if uv is not None:
            r._renderer._program_cache = ShaderProgramCache(
                shader_dir="./shaders",
            )
            color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
        else:
            #r._renderer._program_cache = ShaderProgramCache()
            color, _ = r.render(scene, flags=pyrender.RenderFlags.FLAT | pyrender.RenderFlags.RGBA)

        r.delete()

        assert color.shape[-1] == 4
        alpha = color[..., 3]
        color = color[..., :3]
        #else:
        #    alpha = None

        if rotpi:
            if alpha is not None:
                alpha = alpha[:, ::-1]
            color = color[:, ::-1]

        if out_path is not None:
            Image.fromarray(color).save(out_path.with_suffix(".png"))

        return color, alpha

    def get_renderer(self):
        r = pyrender.OffscreenRenderer(
            #viewport_width=viewport_width,  # self.uv_wh,
            #viewport_height=viewport_height,  # self.uv_wh,
            viewport_width=self.uv_wh,
            viewport_height=self.uv_wh,
        )
        if self.uv_b is not None:
            r._renderer._program_cache = ShaderProgramCache(
                shader_dir="./shaders",
            )

        return r

    def fast_render(
        self,
        r,
        deforms=None,
        bg_color=None,
        viewport_width=512,
        viewport_height=512,
        smooth=False,
        out_path=None,
        interactive=False,
    ):
        uv = self.uv_b
        texture = self.texture_b
        layer = self.layer_b

        mesh = self.prepare_mesh_for_render(deforms=deforms, interactive=interactive, out_path=out_path)

        #vertices = mesh.vertices
        #assert np.allclose(
        #    vertices[self.omesh.vert_to_skel[9]][:2],
        #    [0, 0],
        #), (vertices[self.omesh.vert_to_skel[9]][:2], "hips must be centered")

        mesh = pyrender.Mesh.from_trimesh(
            mesh,
            smooth=smooth,
        )
        # https://github.com/mmatl/pyrender/issues/51
        for primitive in mesh.primitives:
            primitive.material.baseColorFactor = [1., 1., 1., 1.]
            if self.nearest and primitive.material.baseColorTexture is not None:
                primitive.material.baseColorTexture.sampler = Sampler(
                    minFilter=GLTF.NEAREST,
                    magFilter=GLTF.NEAREST,
                )

            if not smooth:
                if uv is not None:
                    assert layer is not None
                    indices = self.triangles
                    uv = uv[indices].reshape((3 * len(indices), uv.shape[1]))
                    layer = np.repeat(layer, 3, axis=0)

            if uv is not None:
                primitive.color_0 = layer
                primitive.texcoord_1 = uv
                primitive.material.emissiveTexture = texture
                if self.nearest:
                    primitive.material.emissiveTexture.sampler = Sampler(
                        minFilter=GLTF.NEAREST,
                        magFilter=GLTF.NEAREST,
                    )

        scene = pyrender.Scene(
            bg_color=bg_color,
            ambient_light=(1.0, 1.0, 1.0),
        )
        scene.add(mesh, "mesh")

        camera_pose = np.eye(4)
        camera_pose[2, -1] = 1
        pyrencamera = pyrender.camera.OrthographicCamera(
            self.uv_wh / self.wh,
            self.uv_wh / self.wh,
        )
        scene.add(pyrencamera, pose=camera_pose)
        color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color[..., :3]

        if out_path is not None:
            Image.fromarray(color).save(out_path.with_suffix(".png"))

        return color

    def _add_z(self, vertices, theta, z_order):
        vertices = np.pad(vertices, ((0, 0), (0, 1)))

        for joint_idx in range(len(self.skeleton_data.joints)):
            sk = self.skinning[joint_idx]
            z = theta * self.z_order[joint_idx] + (1 - theta) * z_order[joint_idx]
            vertices[:, 2] += sk * z
            bones_idx = self.skeleton_data.kps_to_skeleton[joint_idx]

            for bone_idx in bones_idx:
                bone_idx_in_skinning = len(self.skeleton_data.joints) + bone_idx
                sk = self.skinning[bone_idx_in_skinning]

                z1 = (
                    theta * self.z_order[self.skeleton_data.skeleton[bone_idx][0]]
                    +
                    (1 - theta) * z_order[self.skeleton_data.skeleton[bone_idx][0]]
                )
                z2 = (
                    theta * self.z_order[self.skeleton_data.skeleton[bone_idx][1]]
                    +
                    (1 - theta) * z_order[self.skeleton_data.skeleton[bone_idx][1]]
                )

                vertices[:, 2] += 0.5 * sk * (z1 + z2)

        vertices[:, 2] *= 2 * 10 * 2 / 224

        return vertices

    def dirichlet_step(
        self,
        pins,
        wc=100,
        nzinds=None,
        n_iters=5_000,  # 2_000,
        tol=1e-2,       # 1e-1,
        vertices=None,
    ):
        pins = pins.copy()
        if nzinds is not None:
            zinds = sorted(set(range(len(self.skeleton_data.joints))) - set(nzinds))
            pins[zinds] = self.skeleton[zinds].copy()

        if self.omesh.n_sample_pts > 0:
            pins = np.concatenate(
                [
                    pins,
                    self.omesh.sample_pts_on_bones(pins),
                ],
                axis=0,
            )

        if vertices is None:
            vertices = self.vertices + pins[self.skeleton_data.root]

        x = np.ndarray((self.vertices.shape[0], 1))
        y = np.ndarray((self.vertices.shape[0], 1))
        energy = fastSymDir.optimizeDeformation(
            x,
            y,
            vertices,
            self.triangles,
            self.omesh.vert_to_skel + self.omesh.sverts_inds,
            pins,
            wc,
            n_iters,
            tol,
        )
        new_vertices = np.hstack([x, y])

        return new_vertices, energy

    def dirichlet(
        self,
        pins,
        z_order,
        wc=100,
        out_dir=None,
        suffix="",
        nzinds=None,
        n_iters=5_000,
        tol=1e-2,
    ):
        if out_dir is not None:
            deformation = np.zeros_like(self.vertices)
            deformation = self._add_z(deformation, 1, z_order)
            self.prepare_mesh_for_render(deformation, out_path=out_dir / f"img_deformed_dirichlet_{suffix}_{0:0>3}")

        new_vertices, energy = self.dirichlet_step(
            pins,
            wc=wc,
            nzinds=nzinds,
            n_iters=n_iters,
            tol=tol,
        )

        deformation = new_vertices - self.vertices
        deformation = self._add_z(deformation, 0, z_order)

        if out_dir is not None:
            self.prepare_mesh_for_render(deformation, out_path=out_dir / f"img_deformed_dirichlet_{suffix}_{1:0>3}")

        return new_vertices, deformation, energy

    def deformN(
        self,
        pinss,
        z_order,
        out_dir=None,
        suffix="",
        wc=100,
        uv=None,
        texture=None,
        layer=None,
        n_iters=50_000,
        tol=1e-4,
        method="dirichlet",
        interactive=False,
    ):
        r = self.get_renderer()

        #wh = self.uv_wh
        wh = self.wh_skeleton
        root = np.array([self.uv_wh, self.uv_wh]) // 2
        fill_value = 255
        skeleton_img = np.full((self.uv_wh, self.uv_wh, 3), fill_value=fill_value, dtype="uint8")

        if method == "arap":
            deformator = igl.ARAP(
                self.vertices,
                self.triangles,
                2,
                np.array(self.omesh.vert_to_skel + self.omesh.sverts_inds),
                igl.ARAP_ENERGY_TYPE_ELEMENTS,
            )

        deformations = []
        new_vertices = None
        for n_iter, (theta, pins) in enumerate(zip(np.linspace(1, 0, len(pinss)), pinss)):
            if out_dir is not None:
                skeleton_img[:] = fill_value
                plot_skel_cv(
                    img=skeleton_img,
                    joints=denormalize_keypoints(
                        pins,
                        wh=wh,#/2,
                        root=root,
                    ).round().astype("int"),
                    skeleton=self.skeleton_data.skeleton,
                    joint_names=self.skeleton_data.joints,
                )
                op = out_dir / f"img_deformed_{method}_N_{suffix}_{n_iter:0>3}"
                cv2.imwrite(str(op.with_name(f"skeleton_{op.stem}.png")), skeleton_img)

            # fastfix for test_2_col manual animation
            if False and n_iter == 0:
            #if n_iter == 0:
                new_vertices, energy = self.vertices.copy() + pins[self.skeleton_data.root], 4.0
            else:
                if method == "dirichlet":
                    new_vertices, energy = self.dirichlet_step(
                        pins,
                        wc=wc,
                        n_iters=n_iters,
                        tol=tol,
                        vertices=new_vertices,
                    )
                else:
                    if self.omesh.n_sample_pts > 0:
                        pins = np.concatenate(
                            [
                                pins,
                                self.omesh.sample_pts_on_bones(pins),
                            ],
                            axis=0,
                        )
                    new_vertices = deformator.solve(
                        pins,
                        self.vertices + pins[self.skeleton_data.root],
                    )
                    energy = 0

            deformation = new_vertices - self.vertices
            deformation = self._add_z(deformation, theta, z_order)
            deformations.append(deformation)
            if out_dir is not None:
                i = self.fast_render(
                    r,
                    deforms=deformation,
                    out_path=out_dir / f"img_deformed_{method}_N_{suffix}_{n_iter:0>3}",
                    #uv=uv,
                    #texture=texture,
                    #layer=layer,
                    interactive=interactive,
                )
                if n_iter == 0:
                    deforms0 = deformation.copy()
                    deforms0[:, :2] -= pins[self.skeleton_data.root]
                    i = self.fast_render(
                        r,
                        deforms=deforms0,
                        #uv=uv,
                        #texture=texture,
                        #layer=layer,
                        interactive=interactive,
                    )
                    i = cv2.cvtColor(i, cv2.COLOR_RGB2BGR)
                    plot_skel_cv(
                        img=i,
                        joints=denormalize_keypoints(
                            pins - pins[self.skeleton_data.root],
                            wh=wh,
                            root=root,
                        ).round().astype("int"),
                        skeleton=self.skeleton_data.skeleton,
                        joint_names=self.skeleton_data.joints,
                        t_jun=denormalize_keypoints(
                            normalize_keypoints(self.omesh.t_jun, self.wh, self.root),
                            wh=wh,
                            root=root,
                        ).round().astype("int") if len(self.omesh.t_jun) > 0 else None,
                    )
                    op = out_dir / f"img_deformed_{method}_N_{suffix}_{0:0>3}"
                    cv2.imwrite(str(op.with_name(f"t_jun_skeleton_{op.stem}.png")), i)

        r.delete()

        ios = []
        for n_iter, deformation in enumerate(deformations):
            # unite masks?
            is_any = False
            _m, _, _m_fg = self.get_o_mask(deformation)
            #cv2.imwrite(str(out_dir / f"mask_deformed_{method}_N_{suffix}_{n_iter:0>3}.png"), _m)
            #cv2.imwrite(str(out_dir / f"mask_fg_deformed_{method}_N_{suffix}_{n_iter:0>3}.png"), _m_fg)
            _m = _m.ravel()
            _mmax = _m.max()
            if _mmax > 0:
                assert _mmax == 255
                _m = _m / _mmax

            _mmax = _m_fg.max()
            if _mmax != 1:
                #assert _mmax == 255, _mmax
                assert _mmax > 0
                _m_fg = _m_fg.ravel() / _mmax
            else:
                _m_fg = _m_fg.ravel()
            num = _m * _m_fg
            den = _m + _m_fg - num
            io = (num / den).mean()
            ios.append(io)

        np.savetxt(str(out_dir / f"ios_{method}_N_{suffix}.txt"), np.array(ios))

        return new_vertices, deformations, energy

    def deform(
        self,
        pins,
        z_order,
        method="arap",
        wc=100,
        n_iters=50_000,
        tol=1e-4,
    ):
        assert method in ("arap", "dirichlet", "kvf")

        if method == "arap":
            deformator = igl.ARAP(
                self.vertices,
                self.triangles,
                2,
                np.array(self.omesh.vert_to_skel + self.omesh.sverts_inds),
                igl.ARAP_ENERGY_TYPE_ELEMENTS,
            )
            #deformator = deform.arap.ARAP(skeleton, self.triangles, vertices)

            #deformator = deform.arap2.ARAP(
            #    vertices,
            #    self.triangles,
            #    #self.vert_to_skel,
            #    np.array(self.vert_to_skel + self.sverts_inds),
            #)
            # (V[vinds] * B).sum(1)
            #_, vbinds, B = self._sample_pts_on_bones(skeleton, is_bary=True)
            #deformator = deform.arap2.ARAPbary(
            #    vertices,
            #    self.triangles,
            #    vbinds,
            #    B,
            #)
        elif method == "kvf":
            deformator = deform.kvf.KVF(
                self.vertices,
                self.triangles,
                np.array(self.omesh.vert_to_skel + self.omesh.sverts_inds),
            )

        if method == "dirichlet":
            new_vertices, energy = self.dirichlet_step(
                pins,
                wc=wc,
                n_iters=n_iters,
                tol=tol,
            )
        else:
            if self.omesh.n_sample_pts > 0:
                pins = np.concatenate(
                    [
                        pins,
                        self.omesh.sample_pts_on_bones(pins),
                    ],
                    axis=0,
                )
            new_vertices = deformator.solve(
                pins,
                self.vertices + pins[self.skeleton_data.root],
            )
            energy = 0

        deformation = new_vertices - self.vertices
        deformation = self._add_z(deformation, 0, z_order)

        return new_vertices, deformation, energy

    def _mesh_quality(self, V, T, V_old):
        V_new = torch.from_numpy(V[:, :2])
        V_old = torch.from_numpy(V_old[:, :2])
        V_new = V_new[T]
        V_old = V_old[T]
        mesh_quality = losses.calc_symmetric_dirichlet(V_old, V_new).numpy()

        return mesh_quality

    def _verts_inds_bad_quality(self, V, T, V_old):
        V = V.copy()
        V[:, 2] = 0
        energy = self._mesh_quality(V, T, V_old)

        return energy

    def _recalculate_skinning(self, force_rewrite=False, inds=None):
        if inds is None:
            if force_rewrite or not hasattr(self, "skinning"):
                W = biskin.biskin(
                    np.pad(self.vertices, ((0, 0), (0, 1))),
                    self.triangles,
                    self.omesh.BE,
                    self.omesh.CE,
                    self.omesh.vert_to_skel,
                )
                W = np.ascontiguousarray(W).T
                assert np.allclose(W.sum(0), 1)
                self.skinning = W

            ret = 0
        else:
            self.skinning = np.hstack(
                [
                    self.skinning,
                    self.skinning[:, inds],
                ]
            )
            ret = 0

        return ret

    def _push_mesh(self):
        self.saved_mesh.append(
            (
                self.vertices.copy(),
                self.triangles.copy(),
                self.skinning.copy(),
                self.mapping.copy(),
            )
        )

    def _restore_mesh(self):
        assert len(self.saved_mesh) > 0
        vertices, triangles, skinning, mapping = self.saved_mesh[0]
        self.vertices = vertices.copy()
        self.triangles = triangles.copy()
        self.skinning = skinning.copy()
        self.mapping = mapping.copy()
        self.set_trimesh(self.vertices)
        self.saved_mesh.clear()
        if hasattr(self, "old_vertices"):
            delattr(self, "old_vertices")
            delattr(self, "old_triangles")
            delattr(self, "old_skinning")

    def _pop_mesh(self):
        assert len(self.saved_mesh) > 0
        vertices, triangles, skinning, mapping = self.saved_mesh[-1]
        self.vertices = vertices.copy()
        self.triangles = triangles.copy()
        self.skinning = skinning.copy()
        self.mapping = mapping.copy()
        self.set_trimesh(self.vertices)

    def _save_mesh(self):
        self.old_triangles = self.triangles.copy()
        self.old_vertices = self.vertices.copy()
        self.old_skinning = self.skinning.copy()
        self.old_mapping = self.mapping.copy()

    def _load_mesh(self):
        self.vertices = self.old_vertices.copy()
        self.triangles = self.old_triangles.copy()
        self.skinning = self.old_skinning.copy()
        self.mapping = self.old_mapping.copy()
        self.set_trimesh(self.vertices)

    def _connected_components(self):
        g = nx.Graph()
        for t in self.triangles:
            g.add_edge(t[0], t[1])
            g.add_edge(t[1], t[2])
            g.add_edge(t[2], t[0])

        cc = list(nx.connected_components(g))

        return cc

    def _cc_inds(self):
        cc = self._connected_components()

        inds = [sorted(c) for c in cc]

        return inds

    def left_right_cut_mesh(
        self,
        contour,
        t_ov_inds2,
        m,
        predicted_keypoints_2d_normalized,
        z_order,
        wc=100,
        out_dir=None,
        nzinds=None,
        force_rewrite=False,
        to_show=False,
    ):
        self._load_mesh()

        t_ov_inds = t_ov_inds2[1] if contour.left_t == "f" else t_ov_inds2[0]
        self.vertices, self.triangles, new_verts_inds, mapping = cut_mesh(
            self.vertices,
            self.triangles,
            t_ov_inds,
            contour.path[:m + 1],
            n_verts=self.n_verts,
            bnd=self.omesh.bnd,
            to_show=to_show,  #to_show,  # False,  # to_show,
        )
        assert len(set(self.mapping) & set(mapping)) == 0
        self.mapping.update(mapping)

        t_ov_inds = t_ov_inds2[1] if contour.right_t == "f" else t_ov_inds2[0]
        self.vertices, self.triangles, new_verts_inds_r, mapping = cut_mesh(
            self.vertices,
            self.triangles,
            t_ov_inds,
            contour.path[m:][::-1],
            n_verts=self.n_verts,
            bnd=self.omesh.bnd,
            to_show=to_show,  #to_show,  # False,  # to_show,
        )
        assert len(set(self.mapping) & set(mapping)) == 0, (set(self.mapping) & set(mapping))
        self.mapping.update(mapping)
        new_verts_inds.extend(new_verts_inds_r)
        # TODO: FIXME must be unique but in some cases are not! <18-12-23 kbrodt>
        # if type is "f" and "b" and the origin is in one point, then it is not the case!!!
        assert sorted(set(new_verts_inds)) == sorted(new_verts_inds), (sorted(set(new_verts_inds)), sorted(new_verts_inds))

        self.set_trimesh(self.vertices)

        if force_rewrite:
            self._recalculate_skinning(force_rewrite=True)#, inds=new_verts_inds)
        else:
            self._recalculate_skinning(inds=new_verts_inds)

        self._path["visible"].clear()
        self._path["hidden"].clear()
        if contour.left_t == "f":
            self._path["visible"].extend(contour.path[:m + 1])
            self._path["hidden"].extend(contour.path[m:])
        else:
            self._path["hidden"].extend(contour.path[:m + 1])
            self._path["visible"].extend(contour.path[m:])

        _, _, energy = self.dirichlet(
            predicted_keypoints_2d_normalized,
            z_order,
            wc=wc,
            out_dir=out_dir,
            suffix=f"p{m:0>3}",
            nzinds=nzinds,
            #n_iters=40_000,
            #tol=1e-4,
        )

        return energy

    def _cut_path2(
        self,
        contour,
        t_ov_inds,
        predicted_keypoints_2d_normalized,
        z_order,
        out_dir,
        use_min=False,
        wc=100,
        nzinds=None,
        use_optuna=True,
        to_show=False,
    ):
        n_path = len(contour.path)
        assert n_path > 2, "Path must have more than one edge"

        energies = []
        if use_optuna:
            study = optuna.create_study(
                sampler=optuna.samplers.TPESampler(
                    n_startup_trials=max(min(3, n_path - 2), int(0.1 * (n_path - 2))),
                    seed=314159,
                ),
                direction="minimize",
            )
            def objective(trial):
                i = trial.suggest_int("i", 1, n_path - 2)

                return self.left_right_cut_mesh(
                    contour,
                    t_ov_inds,
                    i,
                    predicted_keypoints_2d_normalized=predicted_keypoints_2d_normalized,
                    z_order=z_order,
                    wc=wc,
                    out_dir=out_dir,
                    nzinds=nzinds,
                    #to_show=to_show,
                )

            study.optimize(
                objective,
                n_trials=max(1, (n_path - 2) // 2),
                n_jobs=1,
            )

            for trial in study.trials:
                _energy, = trial.values
                i = trial.params["i"]
                energies.append((_energy, i))

            energies = sorted(energies, key=lambda x: x[1])
        else:
            for i in range(1, n_path - 1):
                _energy = self.left_right_cut_mesh(
                    contour,
                    t_ov_inds,
                    i,
                    predicted_keypoints_2d_normalized=predicted_keypoints_2d_normalized,
                    z_order=z_order,
                    wc=wc,
                    out_dir=out_dir,
                    nzinds=nzinds,
                )
                energies.append((_energy, i))

        if use_min:
            _, i = min(energies, key=lambda x: x[0])
            #test_2_col_6_7_copy_fin2_2d_v_FIN_man_a3_TEST
            #if len(energies) < 20:
            #    i = 2
            contour.m = i
            _energy = self.left_right_cut_mesh(
                contour,
                t_ov_inds,
                i,
                predicted_keypoints_2d_normalized=predicted_keypoints_2d_normalized,
                z_order=z_order,
                wc=wc,
                out_dir=out_dir,
                nzinds=nzinds,
                force_rewrite=True,
            )

        return energies

    def _cut_mesh_via_deformation(
        self,
        predicted_keypoints_2d_normalized,
        z_order,
        occ_contour,
        out_dir,
        wc=100,
        use_min=False,
        to_show=False,
        nzinds=None,
        use_optuna=True,
    ):
        print(f"{nzinds=}")
        _, _, energy = self.dirichlet(
            predicted_keypoints_2d_normalized,
            z_order,
            #wc=wc,
            wc=wc,
            out_dir=out_dir,
            nzinds=nzinds,
        )

        self._save_mesh()
        print(f"{energy=}")

        energies = []
        for c in occ_contour.contours:
            if c.left_t != c.right_t:
                continue

            t_ov_inds = occ_contour.oa.t_ov_inds_f if c.left_t == "f" else occ_contour.oa.t_ov_inds
            self.vertices, self.triangles, new_verts_inds, mapping = cut_mesh(
                self.vertices,
                self.triangles,
                t_ov_inds,
                c.path,
                n_verts=self.n_verts,
                bnd=self.omesh.bnd,
                to_show=to_show,  #to_show,  # False,  # to_show,
            )
            # ??? if type is the same and the origin is in one point, then it is not the case!!!
            assert len(set(self.mapping) & set(mapping)) == 0, (set(self.mapping) & set(mapping), )
            self.mapping.update(mapping)

            if len(self._connected_components()) > 1:
                return [(np.inf, None)]

            self.set_trimesh(self.vertices)
            self._recalculate_skinning(force_rewrite=True)#, inds=new_verts_inds)

            self._path["visible"].clear()
            self._path["hidden"].clear()
            if c.left_t == "f":
                self._path["visible"].extend(c.path)
            else:
                self._path["hidden"].extend(c.path)

            energies.extend(new_verts_inds)
            self._save_mesh()

        if len(energies) > 0:
            # TODO: fix non unique vertices <20-12-23 kbrodt> #
            assert sorted(set(energies)) == sorted(energies), energies
            _, _, energy = self.dirichlet(
                predicted_keypoints_2d_normalized,
                z_order,
                wc=wc,
                out_dir=out_dir,
                suffix=f"_ca_{0:0>4}",
                nzinds=nzinds,
            )
            print(f"{energy=}")

        if not occ_contour.has_tri():
            return [(energy, None)]

        # only one "odd" contour
        c = [
            c
            for c in occ_contour.contours
            if c.left_t != c.right_t
        ]
        c, = c

        self._load_mesh()

        # only one type fb or bf
        return self._cut_path2(
            c,
            (occ_contour.oa.t_ov_inds, occ_contour.oa.t_ov_inds_f),
            predicted_keypoints_2d_normalized,
            z_order=z_order,
            out_dir=out_dir,
            wc=wc,
            use_min=use_min,
            nzinds=nzinds,
            use_optuna=use_optuna,
            to_show=to_show,
        )

    def get_o_mask(self, deformation, out_dir=None):
        mesh = self.mesh.copy()

        face_colors = np.zeros(
            (len(self.omesh.triangles), 4),
            dtype="uint8",
        )
        face_colors[:, 3] = 255
        # todo: choose visible side
        v = self._add_z(self.vertices, 1, self.z_order)
        for oa in self.omesh.overlaping_areas:
            if len(oa.t_ov_inds) == 0:
                continue

            z_f = v[np.unique(self.triangles[oa.t_ov_inds]).ravel(), -1].mean()
            assert len(oa.t_ov_inds_f) > 0
            z_h = v[np.unique(self.triangles[oa.t_ov_inds_f]).ravel(), -1].mean()

            if z_f < z_h:
                face_colors[oa.t_ov_inds_f, :3] = 255
            else:
                face_colors[oa.t_ov_inds, :3] = 255

        #vertex_colors = np.zeros_like(self.mesh.vertices.view(np.ndarray), dtype="uint8")
        self.mesh.visual = trimesh.visual.color.ColorVisuals(
            #vertex_colors=vertex_colors,
            face_colors=face_colors,
        )
        #self.mesh.visual.face_colors = face_colors

        #self.mesh.visual = trimesh.visual.texture.TextureVisuals(
        #    uv=self.mesh.visual.uv,
        #    image=Image.fromarray(self.o_mask_texture),
        #)

        self.render(
            bg_color=[0, 0, 0, 1],
            viewport_width=self.uv_wh,
            viewport_height=self.uv_wh,
            out_path=out_dir / "maskd_mesh_0" if out_dir is not None else None,
            smooth=False,
            #rotpi=True,
        )

        mask, mask_fg = self.render(
            deforms=deformation,
            bg_color=[0, 0, 0, 1],
            viewport_width=self.uv_wh,
            viewport_height=self.uv_wh,
            out_path=out_dir / "maskd_mesh_1" if out_dir is not None else None,
            smooth=False,
        )

        maskb, _ = self.render(
            deforms=deformation,
            bg_color=[0, 0, 0, 1],
            viewport_width=self.uv_wh,
            viewport_height=self.uv_wh,
            out_path=out_dir / "maskd_mesh_1b" if out_dir is not None else None,
            smooth=False,
            rotpi=True,
        )

        self.mesh = mesh.copy()

        mask = mask.any(axis=-1).astype("uint8") * 255
        maskb = maskb.any(axis=-1).astype("uint8") * 255
        maskb = cv2.bitwise_and(maskb, cv2.bitwise_not(mask))
        if out_dir is not None:
            cv2.imwrite(str(out_dir / "maskd.png"), mask)
            cv2.imwrite(str(out_dir / "maskdb.png"), maskb)
            cv2.imwrite(str(out_dir / "maskd_fg.png"), mask_fg)

        return mask, maskb, mask_fg

    def inpaint(self, deformation, mask, maskb, mask_from, texture_from, out_dir=None):
        texture, _ = self.render(
            deforms=deformation,
            viewport_width=self.uv_wh,
            viewport_height=self.uv_wh,
            out_path=out_dir / "texture_mesh" if out_dir is not None else None,
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.dilate(mask, kernel, iterations=3)
        maskb = cv2.dilate(maskb, kernel, iterations=3)
        mask_from = cv2.dilate(mask_from, kernel, iterations=3)

        if self.inpaint_method == "copy":
            texture = texture.copy()
            if out_dir is not None:
                cv2.imwrite(str(out_dir / "texture_to.png"), cv2.cvtColor(texture, cv2.COLOR_RGB2BGR))
            assert mask.shape[0] == mask.shape[1]
            assert mask_from.shape[0] == mask_from.shape[1]
            d = mask.shape[0] - mask_from.shape[0]
            assert d % 2 == 0
            d = d // 2
            print(f"copy textures {d=} {texture.shape=} {mask.shape} {texture_from.shape=} {mask_from.shape=}")
            if d > 0:  # mask > mask_from
                if out_dir is not None:
                    cv2.imwrite(str(out_dir / "texture_from.png"), cv2.cvtColor(texture_from, cv2.COLOR_RGB2BGR))

                # first inpaint
                #m = cv2.bitwise_or(maskb[d:-d, d:-d], mask[d:-d, d:-d])
                #if out_dir is not None:
                #    cv2.imwrite(str(out_dir / "m_ov.png"), m)
                #l = inpaint_lama(texture[d:-d, d:-d], m)
                #if out_dir is not None:
                #    cv2.imwrite(str(out_dir / "texture_l.png"), cv2.cvtColor(l, cv2.COLOR_RGB2BGR))
                #texture[d:-d, d:-d] = l
                # end first inpaint

                m = cv2.bitwise_and(mask[d:-d, d:-d], cv2.bitwise_not(mask_from))
                m = m > 0
                if m.any():
                    _texture = texture[d:-d, d:-d]
                    _texture[m] = texture_from[m].copy()
                    assert np.allclose(_texture, texture[d:-d, d:-d])
                    texture[d:-d, d:-d] = _texture.copy()
                    if out_dir is not None:
                        cv2.imwrite(str(out_dir / "texture_b.png"), cv2.cvtColor(texture, cv2.COLOR_RGB2BGR))

                    # last inpaint
                    m = cv2.bitwise_or(maskb[d:-d, d:-d], cv2.bitwise_and(mask[d:-d, d:-d], mask_from))
                    if out_dir is not None:
                        cv2.imwrite(str(out_dir / "m_ov.png"), m)
                    l = inpaint_lama(texture[d:-d, d:-d], m)
                    if out_dir is not None:
                        cv2.imwrite(str(out_dir / "texture_l.png"), cv2.cvtColor(l, cv2.COLOR_RGB2BGR))
                    texture[d:-d, d:-d] = l
                    # end last inpaint
            elif d < 0:  # mask < mask_from
                if out_dir is not None:
                    cv2.imwrite(str(out_dir / "texture_from.png"), cv2.cvtColor(texture_from[-d:d, -d:d], cv2.COLOR_RGB2BGR))

                # first inpaint
                #m = cv2.bitwise_or(maskb, mask)
                #if out_dir is not none:
                #    cv2.imwrite(str(out_dir / "m_ov.png"), m)
                #l = inpaint_lama(texture, m)
                #if out_dir is not none:
                #    cv2.imwrite(str(out_dir / "texture_l.png"), cv2.cvtcolor(l, cv2.color_rgb2bgr))
                #texture = l
                # end first inpaint

                m = cv2.bitwise_and(mask, cv2.bitwise_not(mask_from[-d:d, -d:d]))
                m = m > 0
                if m.any():
                    _texture = texture_from[-d:d, -d:d].copy()
                    texture[m] = _texture[m]
                    if out_dir is not None:
                        cv2.imwrite(str(out_dir / "texture_b.png"), cv2.cvtColor(texture, cv2.COLOR_RGB2BGR))

                    # last inpaint
                    m = cv2.bitwise_or(maskb, cv2.bitwise_and(mask, mask_from[-d:d, -d:d]))
                    if out_dir is not None:
                        cv2.imwrite(str(out_dir / "m_ov.png"), m)
                    l = inpaint_lama(texture, m)
                    if out_dir is not None:
                        cv2.imwrite(str(out_dir / "texture_l.png"), cv2.cvtColor(l, cv2.COLOR_RGB2BGR))
                    texture = l
                    # end last inpaint
            else:  # mask == mask_from
                if out_dir is not None:
                    cv2.imwrite(str(out_dir / "texture_from.png"), cv2.cvtColor(texture_from, cv2.COLOR_RGB2BGR))

                # first inpaint
                #m = cv2.bitwise_or(maskb, mask)
                #if out_dir is not None:
                #    cv2.imwrite(str(out_dir / "m_ov.png"), m)
                #l = inpaint_lama(texture, m)
                #if out_dir is not None:
                #    cv2.imwrite(str(out_dir / "texture_l.png"), cv2.cvtColor(l, cv2.COLOR_RGB2BGR))
                #texture = l
                # last inpaint

                m = cv2.bitwise_and(mask, cv2.bitwise_not(mask_from))
                m = m > 0
                texture[m] = texture_from[m].copy()
                if out_dir is not None:
                    cv2.imwrite(str(out_dir / "texture_b.png"), cv2.cvtColor(texture, cv2.COLOR_RGB2BGR))

                # last inpaint
                m = cv2.bitwise_or(maskb, cv2.bitwise_and(mask, mask_from))
                if out_dir is not None:
                    cv2.imwrite(str(out_dir / "m_ov.png"), m)
                l = inpaint_lama(texture, m)
                if out_dir is not None:
                    cv2.imwrite(str(out_dir / "texture_l.png"), cv2.cvtColor(l, cv2.COLOR_RGB2BGR))
                texture = l
                # last inpaint
        elif self.inpaint_method == "cv2":
            texture = cv2.inpaint(texture, mask, 3, cv2.INPAINT_TELEA)
            print(f"inpaint: new texture {texture.shape=} ({self.uv_wh=})")
        elif self.inpaint_method == "lama":
            texture = inpaint_lama(texture, mask)
            if out_dir is not None:
                cv2.imwrite(str(out_dir / "texture_b.png"), cv2.cvtColor(texture, cv2.COLOR_RGB2BGR))

            l = inpaint_lama(texture, maskb)
            if out_dir is not None:
                cv2.imwrite(str(out_dir / "texture_l.png"), cv2.cvtColor(l, cv2.COLOR_RGB2BGR))
            texture = l
            print(f"inpaint: new texture {texture.shape=} ({self.uv_wh=})")
            if texture.shape[0] != self.uv_wh:  # w
                texture = texture[:self.uv_wh, :self.uv_wh].copy()

        if out_dir is not None:
            cv2.imwrite(str(out_dir / "texture.png"), cv2.cvtColor(texture, cv2.COLOR_RGB2BGR))

        return texture

    def optimize(
        self,
        predicted_keypoints_2d_normalized,
        z_order,
        out_dir=None,
        wc=100,
        use_optuna=True,
        to_show=False,
    ):
        best_omesh_inds = []
        for self.omesh_ind, _ in enumerate(self.omeshes):
            print(f"Processing {self.omesh_ind=}")
            self._set_vt()

            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)

            ax.set_title("image with mesh")
            ax.imshow(self.img)
            #ax.imshow(self.mask)

            _, ax = self.omesh.plot(ax=ax)

            fig.tight_layout()
            plt.savefig(out_dir / f"mesh{self.omesh_ind}.svg")

            if to_show:
                plt.show()

            plt.close(fig)

            self._push_mesh()

            occ_contours = self.omesh.get_paths(to_show=to_show)
            print(occ_contours)
            print(f"{self.omesh.poly_to_bones=}")

            occ_contour_opts = []
            #for occ_contour_na in reversed(occ_contours):  # aladin 3 pose new 11 12
            for occ_contour_na in occ_contours:
                energies = []
                #test_dvor_girl_lama_FIN_man_TEST
                #_asd = 0
                for occ_contour in occ_contour_na.assign_contours():
                    self._pop_mesh()

                    _out_dir = (
                        out_dir
                        /
                        (
                            f"{occ_contour.idx}_"
                            +
                            "_".join(f"{cp.left_t}{cp.right_t}" for cp in occ_contour.contours)
                        )
                    )
                    print(occ_contour)
                    _out_dir.mkdir(exist_ok=True, parents=True)
                    _energies = self._cut_mesh_via_deformation(
                        predicted_keypoints_2d_normalized,
                        z_order,
                        occ_contour,
                        out_dir=_out_dir,
                        wc=wc,
                        to_show=to_show,
                        #nzinds=bones_to_verts_hierarchy(
                            #self.omesh.poly_to_bones[occ_contour.idx],
                            #self.skeleton_data,
                        #),
                        use_optuna=use_optuna,
                    )

                    #test_dvor_girl_lama_FIN_man_TEST
                    #if occ_contour.idx == 3 and _asd != 1:
                    #    _asd += 1
                    #    continue
                    #_asd += 1

                    energies.append((_energies, occ_contour))

                if len(energies) == 0:
                    continue

                energies_opt, occ_contour_opt = min(energies, key=lambda x: min(x[0]))
                if energies_opt[0][0] == np.inf:
                    continue

                # 2_col/target-7-0.png
                #if len(energies) == 2 and occ_contour_opt.idx == 2 and "bwd" in str(out_dir):
                #    energies_opt, occ_contour_opt = max(energies, key=lambda x: min(x[0]))

                # 2_col/target-6-7.png + test_2_col_6_7_copy_fin2_2d_v_FIN_man_a3_TEST
                #if len(energies) == 2 and occ_contour_opt.idx == 1 and "bwd" in str(out_dir):
                #    energies_opt, occ_contour_opt = max(energies, key=lambda x: min(x[0]))

                # TODO if min == max then choose based on bone_to_top

                plt.title(f"{occ_contour_opt.idx}")
                for _energies, _occ_contour in energies:
                    label = "_".join(f"{cp.left_t}{cp.right_t}" for cp in _occ_contour.contours)
                    if label == "_".join(f"{cp.left_t}{cp.right_t}" for cp in occ_contour_opt.contours):
                        continue

                    plt.plot(list(zip(*_energies))[0], label=f"{label} {min(list(zip(*_energies))[0]):.4f}")

                plt.plot(
                    list(zip(*energies_opt))[0],
                    label=f"{'_'.join(f'{cp.left_t}{cp.right_t}' for cp in occ_contour_opt.contours)} min: {min(list(zip(*energies_opt))[0]):.4f}",
                )
                np.savetxt(out_dir / f"energies_{occ_contour_opt.idx}.txt", np.array(list(zip(*energies_opt))[0]))

                plt.legend()
                plt.grid(ls="--")
                plt.savefig(out_dir / f"energies_{occ_contour_opt.idx}.svg")
                if to_show:
                    plt.show()

                plt.close()

                self._pop_mesh()
                self._cut_mesh_via_deformation(
                    predicted_keypoints_2d_normalized,
                    z_order,
                    occ_contour_opt,
                    out_dir=out_dir,
                    wc=wc,
                    use_min=True,
                    to_show=to_show,
                    #nzinds=bones_to_verts_hierarchy(
                        #self.omesh.poly_to_bones[occ_contour_opt.idx],
                        #self.skeleton_data,
                        #),
                    use_optuna=use_optuna,
                )
                self._push_mesh()
                occ_contour_opts.append(occ_contour_opt)

            self._pop_mesh()
            _, _, energy = self.dirichlet(
                predicted_keypoints_2d_normalized,
                z_order,
                #wc=wc,
                wc=wc,
                out_dir=out_dir,
                suffix=f"oi{self.omesh_ind:0>2}",
            )
            best_omesh_inds.append(
                (
                    energy,
                    self.omesh_ind,
                    self.vertices.copy(),
                    self.triangles.copy(),
                    self.skinning.copy(),
                    self.mapping.copy(),
                    occ_contour_opts,
                )
            )
            print(f"{energy=}, {self.omesh_ind=}")
            self._restore_mesh()

        print([(energy, i) for energy, i, *_ in best_omesh_inds])

        (
            _,
            self.omesh_ind,
            self.vertices,
            self.triangles,
            self.skinning,
            self.mapping,
            occ_contour_opts,
        ) = min(best_omesh_inds, key=lambda x: x[0])

        self._path["visible"].clear()
        self._path["hidden"].clear()
        for occ_contour in occ_contour_opts:
            for c in occ_contour.contours:
                if c.left_t != c.right_t:
                    m = c.m
                    assert m is not None
                    if c.left_t == "f":
                        self._path["visible"].extend(c.path[:m + 1])
                        self._path["hidden"].extend(c.path[m:])
                    else:
                        self._path["hidden"].extend(c.path[:m + 1])
                        self._path["visible"].extend(c.path[m:])
                else:
                    if c.left_t == "f":
                        self._path["visible"].extend(c.path)
                    else:
                        self._path["hidden"].extend(c.path)

        self.set_trimesh(self.vertices)

    def forward(
        self,
        img,
        mask,
        o_mask,
        keypoints_2d,
        z_order,
        bone_to_top,
        t_jun,
        out_dir,
        img_g=None,
        keypoints_2d_g=None,
        z_order_g=None,
        wc=100,
        to_show=False,
        use_optuna=True,
    ):
        out_dir.mkdir(exist_ok=True, parents=True)

        predicted_keypoints_2d_normalized = normalize_keypoints(
            keypoints_2d,
            max(img.shape[:2]),
            root=keypoints_2d[self.skeleton_data.root].copy(),
        )

        if keypoints_2d_g is None:
            predicted_keypoints_2d_normalized_g = predicted_keypoints_2d_normalized
            z_order_g = z_order
        else:
            predicted_keypoints_2d_normalized_g = normalize_keypoints(
                keypoints_2d_g,
                max(img_g.shape[:2]),
                root=keypoints_2d_g[self.skeleton_data.root].copy(),
            )

        self.optimize(
            predicted_keypoints_2d_normalized_g,
            z_order_g,
            out_dir=out_dir,
            wc=wc,
            use_optuna=use_optuna,
            to_show=to_show,
        )

        _, deformation, _ = self.dirichlet(
            predicted_keypoints_2d_normalized,
            z_order,
            out_dir=out_dir,
            wc=2000,
            suffix="N",
            n_iters=1_000_000,
            tol=1e-8,
        )
        inbetweener = Inbetweener(
            self.skeleton_data,
            img,
            mask,
            o_mask,
            keypoints_2d,
            z_order,
            bone_to_top,
            t_jun=t_jun,
            inpaint_method=self.inpaint_method,
            n_pts=self.omesh.n_sample_pts,
            epsilon=self.epsilon,
            to_show=to_show,
            device=self.device,
            nearest=self.nearest,
        )

        # BEGIN INPAINT
        o_mask_d, o_mask_db, _ = self.get_o_mask(deformation, out_dir=out_dir)

        self.uv_b = None
        self.texture_b = None
        self.layer_b = None

        if o_mask_d.any():
            texture = self.inpaint(
                deformation,
                o_mask_d,
                o_mask_db,
                inbetweener.o_mask_texture,
                inbetweener.texture,
                out_dir=out_dir,
            )
            v = self.vertices + deformation[:, :2]
            v = v - v[self.omesh.vert_to_skel[9]]
            uv = get_uv_param(
                v,
                np.array([self.uv_wh, self.uv_wh]) // 2 + 0.5,
                wh=self.wh,
                uv_wh=self.uv_wh,
            )

            layer = np.zeros(
                (len(self.omesh.triangles), 4),
                dtype="uint8",
            )
            layer[:, 3] = 255
            layer[self.omesh.t_ov_inds, :3] = 255

            self.uv_b = uv
            self.texture_b = texture
            self.layer_b = layer

            assert texture.shape[:2] == (self.uv_wh, self.uv_wh), (texture.shape, self.uv_wh)

        # END INPAINT

        #if is_backward:

            #self._restore_mesh()

        #self.r.delete()

        return inbetweener


class Interpolator:
    def __init__(self, inb_0, inb_1, fi_model_path=None):
        self.inb_0 = inb_0
        self.inb_1 = inb_1

        if fi_model_path is not None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = Model()
            self.model.load_model(fi_model_path, -1)
            self.model.eval()
            self.model.device()

    def _slerp(self, angles_i, angles, order, n_steps=24, degrees=False):
        key_times = [0, n_steps - 1]
        times = np.arange(0, n_steps)
        animation = []
        for a_i, a in zip(angles_i, angles):  # [w, x, y, z]
            key_rots = Rotation.concatenate(
                [
                    Rotation.from_euler(order, a_i, degrees=degrees),
                    Rotation.from_euler(order, a, degrees=degrees),
                    #Rotation.from_quat(a_i[[1, 2, 3, 0]]),
                    #Rotation.from_quat(a[[1, 2, 3, 0]]),
                ]
            )
            slerp = Slerp(key_times, key_rots)
            interp_rots = slerp(times)
            anim = interp_rots.as_euler(order, degrees=degrees)  # [n_steps + 1, 3]
            #anim = interp_rots.as_quat()  # [n_steps + 1, 3]
            #anim = anim[:, [3, 0, 1, 2]]  # x. y, z, w -> w, x, y, z
            animation.append(anim)

        animation = np.array(animation)  # [J, n_steps + 1, 3]
        animation = np.transpose(animation, (1, 0, 2))  # [n_steps + 1, J, 3]

        return animation

    def _eu_to_quat(self, animation_seq, order, degrees=False):
        new_animation = []
        for animation in animation_seq:
            animation = Rotation.from_euler(order, animation, degrees=degrees)
            animation = animation.as_quat()  # x, y, z, w
            animation = animation[:, [3, 0, 1, 2]]  # w, x, y, z
            new_animation.append(animation)

        animation_seq = np.array(new_animation)

        return animation_seq

    def _quat_to_eu(self, animation_seq, order, degrees=False):
        new_animation = []
        for animation in animation_seq:  # w, x, y, z
            animation = animation[:, [1, 2, 3, 0]]  # x, y, z, w
            animation = Rotation.from_quat(animation)
            animation = animation.as_euler(order, degrees=degrees)
            new_animation.append(animation)

        animation_seq = np.array(new_animation)

        return animation_seq

    def _get_animation(self, s0, z0, s1, z1, n_steps=24, out_dir=None, intermediate=None, dim=2):
        skel = get_fk(
            s0,
            parents2d=self.inb_0.skeleton_data.parents2d,
            parents2d_to_kps=self.inb_0.skeleton_data.parents2d_to_kps,
            zero_inds=self.inb_0.skeleton_data.zero_inds,
            dim=dim,
        )

        s0s1 = [(s0, z0)]
        if intermediate is not None:
            for kps_g, z_order_g in intermediate:
                s0s1.append((kps_g, z_order_g))

        s0s1.append((s1, z1))
        n_stepss = np.arange(n_steps - len(s0s1))
        n_stepss = np.array_split(n_stepss, len(s0s1) - 1)
        n_stepss = [len(n) + 2 for n in n_stepss]
        #n_steps = n_steps // (len(s0s1) - 1)

        animation_full = []
        kss_full = []
        zs_full = []
        root_pos_full = []
        for i, (n_steps, ((s0, z0), (s1, z1))) in enumerate(zip(n_stepss, itertools.pairwise(s0s1))):
            root_pos0 = s0[self.inb_0.skeleton_data.root]
            if skel.dim == 2:
                angles_i, ks_i, err = skel.calc_angles_and_offsets_ik(
                    torch.from_numpy(s0),
                    root_pos=torch.from_numpy(root_pos0),
                )
            else:
                angles_i, err = skel.calc_angles_ik(
                    torch.from_numpy(np.hstack([s0, z0[:, None]])),
                )

            print(f"IK1 {err=}")

            root_pos1 = s1[self.inb_1.skeleton_data.root]
            if skel.dim == 2:
                angles, ks, err = skel.calc_angles_and_offsets_ik(
                    torch.from_numpy(s1),
                    root_pos=torch.from_numpy(root_pos1),
                )
            else:
                angles, err = skel.calc_angles_ik(
                    torch.from_numpy(np.hstack([s1, z1[:, None]])),
                )

            print(f"IK2 {err=}")

            if skel.dim == 2:
                a1 = np.pad(angles_i.cpu().numpy()[:, None], ((0, 0), (2, 0)))
                a2 = np.pad(angles.cpu().numpy()[:, None], ((0, 0), (2, 0)))
            else:
                a1 = angles_i.cpu().numpy()
                a2 = angles.cpu().numpy()

            animation = self._slerp(
                a1,
                a2,
                order=skel.order,
                n_steps=n_steps,
                degrees=False,
            )

            assert animation.shape == (n_steps, len(skel.offsets), 3)
            #animation = self._eu_to_quat(animation, skel.order, degrees=False)
            #assert animation.shape == (n_steps + 1, len(skel.offsets), 4)

            if skel.dim == 2:
                kss = np.linspace(ks_i.cpu().numpy(), ks.cpu().numpy(), n_steps)  # [n_steps + 1, J, 1]
                assert kss.shape == (n_steps, len(skel.offsets), 1)
            else:
                kss = np.ones(n_steps)  # [n_steps + 1, J, 1]

            zs = np.linspace(z0, z1, n_steps)  # [n_steps + 1, J, 1]
            root_poss = np.linspace(root_pos0, root_pos1, n_steps)  # [n_steps + 1, J, 1]

            if i > 0:
                animation = animation[1:]
                kss = kss[1:]
                zs = zs[1:]
                root_poss = root_poss[1:]

            animation_full.extend(animation)
            kss_full.extend(kss)
            zs_full.extend(zs)
            root_pos_full.extend(root_poss)

        animation = np.array(animation_full)
        kss = np.array(kss_full)
        zs = np.array(zs_full)
        root_poss = np.array(root_pos_full)

        # end animation
        pinss = []
        #iiii = 0
        for k, a, root_pos in zip(kss, animation, root_poss):
            if skel.dim == 2:
                a = a[:, 2]
                #if iiii > 2:
                    #print(f"{iiii=}", a[17])
                    #a[17] = animation[-1][17, 2]
                    #print(f"{iiii=}", a[17])
                k = torch.from_numpy(k).to(self.inb_0.device)
            else:
                k = None

            pins = skel.forward(
                theta=torch.from_numpy(a).to(self.inb_0.device),
                k=k,
                root_pos=torch.from_numpy(root_pos),
                reduce=True,
            )
            pins = pins[:, :2].cpu().numpy()
            pinss.append(pins)
            #iiii += 1

        pinss = np.array(pinss)

        if out_dir is not None:
            offset = skel.offsets#.cpu().numpy()
            offset = offset.cpu().numpy()
            if offset.shape[1] == 2:
                offset = np.pad(offset, ((0, 0), (0, 1)))

            animation = np.rad2deg(animation)
            _animation = Rotation.from_euler(skel.order, np.concatenate(animation), degrees=True)
            assert np.allclose(np.concatenate(animation), _animation.as_euler(skel.order, degrees=True))
            np.savetxt(out_dir / "skeleton.txt", _animation.as_matrix().reshape(animation.shape[0] * animation.shape[1], 9))
            animation = _animation.as_euler("XYZ", degrees=True).reshape(len(animation), skel.num_joints(), 3)
            position = np.zeros((len(animation), 3))
            write_utils.write_bvh(
                parent=self.inb_0.skeleton_data.parents2d,
                offset=offset,
                rotation=animation,
                position=position,
                names=["_".join(name.split()) for name in self.inb_0.skeleton_data.names],
                frametime=1,
                order=skel.order,
                path=out_dir / "skeleton.bvh",
                endsite=None,
            )

        return pinss, zs

    def _read_angles(self, animation_path):
        animation_path = Path(animation_path)
        if animation_path.suffix == ".bvh":
            return self._read_bvh_angles(animation_path)
        elif animation_path.suffix == ".ma":
            return self._read_ma_angles(animation_path)
        elif animation_path.suffix == ".txt":
            return self._read_txt_angles(animation_path)

        raise

    def _read_txt_angles(self, animation_path):
        pins = []
        with open(animation_path) as fin:
            while True:
                line = fin.readline()  # Frame
                if not line:
                    break

                print(line)
                p = np.zeros((len(self.inb_0.skeleton_data.joints), 2), dtype="float64")
                for _ in range(len(self.inb_0.skeleton_data.joints)):
                    line = fin.readline()
                    print(line)
                    joint_name, xyz = line.split(":")
                    x, y, _ = map(float, xyz.split(","))
                    i = self.inb_0.skeleton_data.joints.index(joint_name.replace("_", " "))
                    p[i] = x, y

                pins.append(p)
                line = fin.readline()  # new line

        pins = np.array(pins)

        # fastfix for test_2_col manual animation
        pins[:, :, 0] += 0.5
        # [T, J, D]

        #t = len(pins)
        #new_pins = np.zeros(
        #    (2*t - 1, pins.shape[1], pins.shape[2]),
        #    dtype=pins.dtype,
        #)
        #x = np.linspace(0, t, 2 * t - 1)
        #xp = np.linspace(0, t, t)
        #for j in range(pins.shape[1]):
        #    for d in range(pins.shape[2]):
        #        new_pins[:, j, d] = np.interp(x, xp, pins[:, j, d])

        #pins = new_pins

        return pins, None


    def _read_ma_angles(self, animation_path):
        parents2d = self.inb_0.skeleton_data.parents2d
        parents2d_to_kps = self.inb_0.skeleton_data.parents2d_to_kps
        all_offsets = utils.get_offsets_from_ma(
            animation_path,
            parents2d=parents2d,
            parents2d_to_kps=parents2d_to_kps,
            names=self.inb_0.skeleton_data.names,
            dim=2,
        )

        inds = [
            parents2d_to_kps.index(i)
            for i in range(len(set(parents2d_to_kps)))
        ]

        pins = []
        for offsets in all_offsets:
            offsets = torch.from_numpy(offsets)
            skel = Skeleton2d(parents2d, offsets, inds)
            p = skel.forward(
                reduce=True,
            )
            p = p[:, :2].cpu().numpy()
            pins.append(p)

        pins = np.array(pins)

        return pins, None

    def _read_bvh_angles(self, animation_path):
        bvh = BVH.from_file(animation_path)
        order = "".join(o[0] for o in bvh.root_joint.channel_order[-3:]).lower()
        assert order in ["xyz"], "Only supported \"xyz\" order"

        jnames = bvh.get_joint_names()
        inds_to_use = [i for i, name in enumerate(jnames) if name != "End Site"]
        jnames = [jnames[i] for i in inds_to_use]
        offsets = bvh.get_offsets()
        offsets = [offsets[i] for i in inds_to_use]
        jnames2ind = dict(zip(jnames, range(len(jnames))))
        parents2d = [
            jnames2ind.get(j, j)
            for j in bvh.get_parents()
        ]
        parents2d = [parents2d[i] for i in inds_to_use]
        assert parents2d == self.inb_0.skeleton_data.parents2d, (parents2d, self.inb_0.skeleton_data.parents2d)

        # parents2d = self.inb_0.skeleton_data.parents2d
        parents2d_to_kps = self.inb_0.skeleton_data.parents2d_to_kps
        zero_inds = self.inb_0.skeleton_data.zero_inds
        inds = [
            parents2d_to_kps.index(i)
            for i in range(len(set(parents2d_to_kps)))
        ]
        skel = Skeleton3d(parents2d, offsets, inds, zero_inds)

        animation = []
        rot_data = bvh.rot_data[:, inds_to_use]
        for q in rot_data:
            q = Rotation.from_quat(q[:, [1, 2, 3, 0]])  # [w, x, y, z] -> [x, y, z, w]
            animation.append(
                q.as_euler(
                    order,
                    degrees=False
                )
            )

        pinss = []
        zs = []
        for a, root_pos in zip(animation, bvh.pos_data):
            pins = skel.forward(
                theta=torch.from_numpy(a).to(self.inb_0.device),
                root_pos=torch.from_numpy(root_pos),
                reduce=True,
            )
            pins, z = pins[:, :2].cpu().numpy(), pins[:, 2].cpu().numpy()
            pinss.append(pins)
            zs.append(z)

        pinss = np.array(pinss)

        return pinss, zs

    def interpolate(
        self,
        out_dir,
        animation_path=None,
        n_steps=24,
        wc=100,
        dim=2,
        intermediate=None,
        interactive=0,
        deform_method="dirichlet",
    ):
        if animation_path is not None:
            assert not interactive, "Interactive is not supported with BVH animation at the same time"
            if intermediate is not None:
                assert intermediate[0][0] is None, "Guidance and BVH animation are not supported at the same time"

        out_dir.mkdir(exist_ok=True, parents=True)

        intermediate_normalized = []
        if intermediate is not None:
            for img_g, kps_g, z_order_g in intermediate:
                if img_g is None:
                    continue

                kps_g = normalize_keypoints(
                    kps_g,
                    wh=max(img_g.shape[:2]),
                    root=kps_g[self.inb_0.skeleton_data.root],
                )
                intermediate_normalized.append((kps_g, z_order_g))

        if interactive > 0:
            pins, zs = self._interactive(
                out_dir=out_dir,
                wc=wc,
                dim=dim,
                n_key_poses=interactive,  # (interactive + 2) * (len(intermediate_normalized) + 1),
                intermediate=intermediate_normalized,
            )
        elif animation_path is None:
            pins = [
                self.inb_0.skeleton,
                self.inb_1.skeleton,
            ]
            zs = [
                self.inb_0.z_order,
                self.inb_1.z_order,
            ]

            #intermediate_normalized = list(zip(pins[1:-1], zs[1:-1]))

            pins, _ = self._get_animation(
                pins[0],
                zs[0],
                pins[-1],
                zs[-1],
                n_steps=n_steps,
                intermediate=intermediate_normalized,
                out_dir=out_dir,
                dim=dim,
            )
        else:
            pins, _ = self._read_angles(animation_path)
            print("HERE")

        self._interpolate(
            pins,
            out_dir,
            wc=wc,
            deform_method=deform_method,
            interactive=interactive > 0 or animation_path is not None,
        )

    def _crop_img(self, img0, img1, img_size=None):
        if img_size is None:
            img_size = img0.shape[:2]
        dh = (img1.shape[0] - img_size[0]) // 2
        _ph = (img1.shape[0] - img_size[0]) % 2 != 0
        dw = (img1.shape[1] - img_size[1]) // 2
        _pw = (img1.shape[1] - img_size[1]) % 2 != 0
        img1 = img1[dh:-dh - _ph, dw:-dw - _pw].copy()

        return img1

    def _prepare_imgs(self, img0, img1, img_size=None):
        if img0.ndim != 3:
            img0 = np.stack([img0] * 3, axis=2)
        if img1.ndim != 3:
            img1 = np.stack([img1] * 3, axis=2)

        if img0.shape[0] < img1.shape[0]:
            img1 = self._crop_img(img0, img1, img_size=img_size)
        elif img1.shape[0] < img0.shape[0]:
            img0 = self._crop_img(img1, img0, img_size=img_size)

        img0 = (torch.from_numpy(img0.transpose(2, 0, 1)).to(self.device) / 255.).unsqueeze(0)
        img1 = (torch.from_numpy(img1.transpose(2, 0, 1)).to(self.device) / 255.).unsqueeze(0)

        _, _, h, w = img0.shape
        ph = ((h - 1) // 32 + 1) * 32
        pw = ((w - 1) // 32 + 1) * 32
        padding = (0, pw - w, 0, ph - h)
        img0 = F.pad(img0, padding)
        img1 = F.pad(img1, padding)

        return img0, img1, h, w

    def _fi(self, img0, img1, ratio=0.5, rthreshold=0.02, rmaxcycles=8):
        #ratio = 0.5
        #return cv2.addWeighted(img0, ratio, img1, 1 - ratio, 0.0)
        img0_ratio = 0.0
        img1_ratio = 1.0
        if ratio <= img0_ratio + rthreshold / 2:
            i = img0
        elif ratio >= img1_ratio - rthreshold / 2:
            i = img1
        else:
            img0, img1, h, w = self._prepare_imgs(img0, img1)
            tmp_img0 = img0
            tmp_img1 = img1
            for _ in range(rmaxcycles):
                with torch.inference_mode():
                    middle = self.model.inference(tmp_img0, tmp_img1)

                if isinstance(middle, tuple):
                    _, _, middle = middle
                middle_ratio = (img0_ratio + img1_ratio) / 2
                if ratio - (rthreshold / 2) <= middle_ratio <= ratio + (rthreshold / 2):
                    break

                if ratio > middle_ratio:
                    tmp_img0 = middle
                    img0_ratio = middle_ratio
                else:
                    tmp_img1 = middle
                    img1_ratio = middle_ratio

            i = middle.squeeze(0)
            i = (i * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]
            i = np.ascontiguousarray(i, dtype=np.uint8)

        return i

    def _interactive(
        self,
        out_dir,
        n_key_poses=1,
        wc=100,
        dim=2,
        intermediate=None,
    ):
        r0 = self.inb_0.get_renderer()
        r1 = self.inb_1.get_renderer()

        pins = self.inb_0.skeleton
        wh = self.inb_0.uv_wh
        wh = self.inb_0.wh_skeleton
        root = np.array([self.inb_0.uv_wh, self.inb_0.uv_wh]) // 2,

        n_steps = 2
        if intermediate is not None:
            n_steps += len(intermediate)

        n_steps += (n_steps - 1) * n_key_poses
        pins, zs = self._get_animation(
            self.inb_0.skeleton,
            self.inb_0.z_order,
            self.inb_1.skeleton,
            self.inb_1.z_order,
            n_steps=n_steps,
            intermediate=intermediate,
            out_dir=out_dir,
            dim=dim,
        )

        pins = denormalize_keypoints(
            pins,
            wh=wh,
            root=root,
        ).round().astype("int")

        alpha = 0.5
        beta = 1 - alpha
        tot_time = 0
        tot_n = 0
        fps = 30
        def img_fn(joints, step, method, hq=False):
            nonlocal tot_n
            nonlocal tot_time
            nonlocal fps
            tot_n += 1
            pins = normalize_keypoints(joints, wh=wh, root=root)

            print()
            start = time.monotonic()
            _, deformation, _ = self.inb_0.deform(
                pins,
                self.inb_1.z_order,
                method=method,
                # increase these numbers for better deformation
                n_iters=2_000,
                tol=1e-2,
            )
            elapsed = time.monotonic() - start
            tot_time += elapsed
            print(f"deform 1: {elapsed:.3f}")

            start = time.monotonic()
            i0 = self.inb_0.fast_render(
                r0,
                deforms=deformation,
                interactive=True,
            ).copy()
            elapsed = time.monotonic() - start
            tot_time += elapsed
            print(f"render 1: {elapsed:.3f}")

            start = time.monotonic()
            _, deformation, _ = self.inb_1.deform(
                pins,
                self.inb_0.z_order,
                method=method,
                # increase these numbers for better deformation
                n_iters=2_000,
                tol=1e-2,
            )
            elapsed = time.monotonic() - start
            tot_time += elapsed
            print(f"deform 2: {elapsed:.3f}")

            start = time.monotonic()
            i1 = self.inb_1.fast_render(
                r1,
                deforms=deformation,
                interactive=True,
            ).copy()
            elapsed = time.monotonic() - start
            tot_time += elapsed
            print(f"render 2: {elapsed:.3f}")

            #i = cv2.addWeighted(i0, alpha, i1, beta, 0.0)

            start = time.monotonic()
            is_resized = False
            if not hq and fps < 10:
                is_resized = True
                if fps > 5:
                    fx = fy = 0.5
                else:
                    fx = fy = 0.25

                i0 = cv2.resize(i0, None, fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
                i1 = cv2.resize(i1, None, fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)

            elapsed = time.monotonic() - start
            print(f"resize: {elapsed:.3f}")
            tot_time += elapsed

            start = time.monotonic()
            i = self._fi(i0, i1, ratio=step / (n_steps - 1))
            i = cv2.cvtColor(i, cv2.COLOR_RGB2BGR)
            elapsed = time.monotonic() - start
            tot_time += elapsed
            print(f"frame int: {elapsed:.3f}")
            print(f"tot time: {tot_time / tot_n:.3f}")
            fps = tot_n / tot_time
            print(f"fps: {fps:.3f}")

            if i0.shape[0] < i.shape[0]:
                i = self._crop_img(i0, i)
            elif i.shape[0] < i0.shape[0]:
                # center pad i
                _sh = (i0.shape[0] - i.shape[0]) // 2
                _ph = (i0.shape[0] - i.shape[0]) % 2 != 0
                _sw = (i0.shape[1] - i.shape[1]) // 2
                _pw = (i0.shape[1] - i.shape[1]) % 2 != 0
                i = np.pad(
                    i,
                    ((_sh, _sh + _ph), (_sw, _sw + _pw), (0, 0)),
                    mode="symmetric",
                )

            if is_resized:
                i = cv2.resize(i, None, fx=1/fx, fy=1/fy, interpolation=cv2.INTER_CUBIC)

            return i

        pins = draw.SketcherInb(
            img=self.inb_0.img.copy(),
            animation=pins,
            root_ind=self.inb_0.skeleton_data.root,
            img_fn=img_fn,
            #deform_methods=["arap", "dirichlet", "kvf"],  # igl.ARAP doesn't seem to work
            deform_methods=["dirichlet", "kvf"],
            skeleton=self.inb_0.skeleton_data.skeleton,
            right_inds=self.inb_0.skeleton_data.right_inds,
            windowname="inbetweening",
        ).annotate()

        #r0.delete()
        r1.delete()

        pins = normalize_keypoints(
            pins,
            wh=wh,
            root=root,
        )

        return pins, zs

    def _interpolate(self, pins, out_dir, wc=100, deform_method="dirichlet", interactive=False):
        _out_dir = out_dir / "fwd"
        _out_dir.mkdir(exist_ok=True, parents=True)
        for deform_method in ["dirichlet"]:
            np.save(_out_dir / "pins.npy", pins)
            self.inb_0.deformN(
                pins,
                self.inb_1.z_order,
                out_dir=_out_dir,
                wc=wc,
                suffix="N",
                uv=self.inb_0.uv_b,
                texture=self.inb_0.texture_b,
                layer=self.inb_0.layer_b,
                method=deform_method,
                interactive=interactive,
            )

        _out_dir = out_dir / "bwd"
        _out_dir.mkdir(exist_ok=True, parents=True)
        for deform_method in ["dirichlet"]:
            self.inb_1.deformN(
                pins[::-1],
                self.inb_0.z_order,
                out_dir=_out_dir,
                wc=wc,
                suffix="N",
                uv=self.inb_1.uv_b,
                texture=self.inb_1.texture_b,
                layer=self.inb_1.layer_b,
                method=deform_method,
                interactive=interactive,
            )

    def interactive_free(self, method="arap"):
        pins = self.inb_0.skeleton
        wh = self.inb_0.uv_wh
        root = np.array([self.inb_0.uv_wh, self.inb_0.uv_wh]) // 2,

        #alpha = 0.5
        #beta = 0.5
        def img_fn(joints):
            pins = normalize_keypoints(joints, wh=wh, root=root)
            _, deformation, _ = self.inb_0.deform(
                pins,
                self.inb_1.z_order,
                method=method,
                n_iters=2_000,
                tol=1e-2,
            )
            i, _ = self.inb_0.render(
                deforms=deformation,
                uv=self.inb_0.uv_b,
                texture=self.inb_0.texture_b,
                layer=self.inb_0.layer_b,
            )

            i0 = i.copy()

            _, deformation, _ = self.inb_1.deform(
                pins,
                self.inb_0.z_order,
                method=method,
                n_iters=2_000,
                tol=1e-2,
            )
            i, _ = self.inb_1.render(
                deforms=deformation,
                uv=self.inb_1.uv_b,
                texture=self.inb_1.texture_b,
                layer=self.inb_1.layer_b,
            )

            i1 = i.copy()
            #i = cv2.addWeighted(i0, alpha, i1, beta, 0.0)

            i = self._fi(i0, i1)
            i = cv2.cvtColor(i, cv2.COLOR_RGB2BGR)

            return i

        #img = cv2.cvtColor(self.inb_0.img, cv2.COLOR_RGB2BGR)
        img = self.inb_0.img.copy()
        joints = denormalize_keypoints(
            pins,
            wh=wh,
            root=root,
        ).round().astype("int")

        draw.SketcherSkeleton(
            img=img,
            joints=joints,
            img_fn=img_fn,
            skeleton=self.inb_0.skeleton_data.skeleton,
            right_inds=self.inb_0.skeleton_data.right_inds,
            windowname="inbetweening",
        ).annotate()


def set_seed(seed=314159):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_img(img_path):
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

    return img


def get_mask(img_path):
    img_path = Path(img_path)
    mask_path = img_path.with_stem(f"{img_path.stem}_mask")

    if not mask_path.is_file():
        print(f"Mask file not found. Using `segment` model")

        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = segment.segment(img)

        alpha = 0.5
        plt.imshow(add_mask(img, mask, alpha=alpha))
        plt.axis("off")
        plt.show()
        plt.close()

        ok = input("Mask is ok? Y/n/b(box)\n").lower()
        if ok in ["b", "box"]:
            return None
        if ok not in ["", "y"]:
            app_exe = find_executable(GIMP)
            if not app_exe:
                raise FileNotFoundError(
                    f"`{GIMP}` not found. Install `{GIMP}` to manual annotation or provide the mask in {mask_path} file."
                )

            amask_path = mask_path.with_stem(f"{mask_path.stem}_mask_segment")
            cv2.imwrite(str(amask_path), mask)
            cmd = textwrap.dedent(
                f"""
                    {app_exe} \
                        "{str(img_path)}" \
                        "{str(amask_path)}"
                """
            )
            subprocess.run(shlex.split(cmd))
        else:
            amask_path = mask_path.with_stem(f"{mask_path.stem}")
            cv2.imwrite(str(amask_path), mask)

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    return mask


def crop_img_by_mask(img, mask):
    label_mask = label(mask)
    prop, = regionprops(label_mask)
    y_min, x_min, y_max, x_max = prop.bbox
    r = 0.1
    d = math.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2)
    d = r * d
    y_min = max(0, int(y_min - d))
    x_min = max(0, int(x_min - d))
    y_max = min(img.shape[0] - 1, int(y_max + d))
    x_max = min(img.shape[1] - 1, int(x_max + d))
    box = y_min, x_min, y_max, x_max
    img_crop = img[y_min:y_max, x_min:x_max].copy()

    return img_crop, box


def get_overlapping_mask(img_path, model_path, bbox, thresh=0.5, use_gt=False, to_show=False, use_slider=False):
    img_path = Path(img_path)

    mask_path = img_path.with_stem(f"{img_path.stem}_mask")
    o_mask_path = img_path.with_stem(f"{img_path.stem}_occlusion_mask")
    mask_path_pred = img_path.with_stem(f"{img_path.stem}_mask_pred")
    o_mask_path_pred = img_path.with_stem(f"{img_path.stem}_occlusion_mask_pred")
    o_mask_path_pred_pre = img_path.with_stem(f"{img_path.stem}_occlusion_mask_pred_pre")

    if mask_path.is_file() and o_mask_path_pred.is_file():
        print(f"Loading pred mask {o_mask_path_pred}")
        o_mask = cv2.imread(str(o_mask_path_pred), cv2.IMREAD_GRAYSCALE)
    elif use_gt and mask_path.is_file() and o_mask_path.is_file():
        print(f"Loading gt mask {o_mask_path}")
        o_mask = cv2.imread(str(o_mask_path), cv2.IMREAD_GRAYSCALE)
    elif Path(model_path).is_file():
        print(f"Using nn model {model_path}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model, transform = dino_inference.get_model(
            model_fpath=model_path,
        )

        model.to(device)
        img = dino_inference.imread(img_path)

        # ymin, xmin, ymax, xmax
        img_crop = img[bbox[0]: bbox[2], bbox[1]: bbox[3]]

        mask_p, o_mask_p = dino_inference.inference_model(
            model,
            img_crop,
            transform,
            device=device,
        )

        if to_show:
            alpha = 0.5
            plt.subplot(1, 2, 1)
            plt.imshow(add_mask(img_crop, mask_p, alpha=alpha, soft=True))
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.imshow(add_mask(img_crop, o_mask_p, alpha=alpha, soft=True))
            plt.axis("off")

            plt.tight_layout()
            plt.show()
            plt.close()

        if use_slider:
            _mask, _thresh_m = draw.OMaskSlider(
                img=cv2.cvtColor(img_crop, cv2.COLOR_RGB2BGR),
                mask=mask_p,
                windowname="mask",
            ).annotate()
            _o_mask, _thresh = draw.OMaskSlider(
                img=cv2.cvtColor(img_crop, cv2.COLOR_RGB2BGR),
                mask=o_mask_p,
                windowname="omask",
            ).annotate()

        #o_mask = dino_inference.inference_model(model, img, transform)
        o_mask_pre = (o_mask_p * 255).astype("uint8")
        o_mask = ((o_mask_p > thresh) * 255).astype("uint8")
        if use_slider:
            if _thresh == thresh:
                assert (o_mask == _o_mask).all()
            else:
                print(f"{thresh=} {_thresh=}")
                o_mask = _o_mask

        o_mask = dino_inference.remove_small_objects(o_mask)

        mask = ((mask_p > thresh) * 255).astype("uint8")
        if use_slider:
            if _thresh_m == thresh:
                assert (mask == _mask).all()
            else:
                print(f"{thresh=} {_thresh_m=}")
                mask = _mask

        #mask = dino_inference.remove_small_objects(mask, thresh=None)

        y_min, x_min, y_max, x_max = bbox
        tmp_mask = np.zeros_like(img[:, :, 0])
        tmp_mask[y_min:y_max, x_min:x_max] = o_mask_pre
        o_mask_pre = tmp_mask

        y_min, x_min, y_max, x_max = bbox
        tmp_mask = np.zeros_like(img[:, :, 0])
        tmp_mask[y_min:y_max, x_min:x_max] = o_mask
        o_mask = tmp_mask

        tmp_mask = np.zeros_like(img[:, :, 0])
        tmp_mask[y_min:y_max, x_min:x_max] = mask
        mask = tmp_mask

        cv2.imwrite(str(mask_path_pred), mask)
        cv2.imwrite(str(o_mask_path_pred), o_mask)
        cv2.imwrite(str(o_mask_path_pred_pre), o_mask_pre)

        plt.subplot(1, 2, 1)
        plt.imshow(add_mask(img, mask, alpha=0.5))
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(add_mask(img, o_mask, alpha=0.5))
        plt.axis("off")

        plt.tight_layout()
        plt.show()
        plt.close()

        ok = input("Mask is ok? Y/n\n").lower()
        if ok not in ["", "y"]:
            app_exe = find_executable(GIMP)
            if not app_exe:
                raise FileNotFoundError(
                    f"`{GIMP}` not found. Install `{GIMP}` to manual annotation or provide the mask in {mask_path} file."
                )

            cmd = textwrap.dedent(
                f"""
                    {app_exe} \
                        "{str(img_path)}" \
                        "{str(o_mask_path_pred)}" \
                        "{str(mask_path_pred)}"
                """
            )
            subprocess.run(shlex.split(cmd))
            o_mask = cv2.imread(str(o_mask_path_pred), cv2.IMREAD_GRAYSCALE)
        elif not mask_path.is_file():
            cv2.imwrite(str(mask_path), mask)
    else:
        print(f"Overlaping mask file not found. Using manual annotation.")

        app_exe = find_executable(GIMP)
        if not app_exe:
            raise FileNotFoundError(
                f"`{GIMP}` not found. Install `{GIMP}` to manual annotation or provide the mask in {mask_path} file."
            )

        cmd = textwrap.dedent(
            f"""
                {app_exe} \
                    "{str(img_path)}"
            """
        )
        subprocess.run(shlex.split(cmd))
        o_mask = cv2.imread(str(o_mask_path), cv2.IMREAD_GRAYSCALE)

    assert mask_path.is_file()
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    _, o_mask = cv2.threshold(o_mask, 127, 255, cv2.THRESH_BINARY)

    return mask, o_mask


def get_skeleton_2d(skeleton_data, img, model_path, mask, o_mask, bbox=None, to_ann=False, to_show=False):
    if skeleton_data.type == "human18" and Path(model_path).is_file():
        print(f"2D skeleton file not found. Using 2D pose estimation model")

        if bbox is None:
            bbox = [0, 0, img.shape[0] - 1, img.shape[1] - 1]

        img_crop = img[bbox[0]: bbox[2], bbox[1]: bbox[3]]
        print('get_skeleton_2d', img.shape, img_crop.shape)

        model = cv2.dnn.readNetFromONNX(model_path)
        (
            _,
            keypoints_2d,
            _,
            _,
        ) = pose_estimation.infer_single_image(
            model,
            img_crop,
            input_img_size=pose_estimation.IMG_SIZE,
            return_kps=True,
        )
        keypoints_2d[:, 0] += bbox[1]
        keypoints_2d[:, 1] += bbox[0]
        keypoints_2d = keypoints_2d.astype("float64")
        _kps = keypoints_2d.astype("int")
        _kps[:, 0] = np.clip(_kps[:, 0], 0, mask.shape[1] - 1)
        _kps[:, 1] = np.clip(_kps[:, 1], 0, mask.shape[0] - 1)
        is_out_of_mask = not mask[*_kps.T[::-1]].all()
        to_ann = to_ann or is_out_of_mask
    else:
        print(
            f"Cannot find neither 2D pose estimation model {model_path} Using manual annotation."
        )
        keypoints_2d = None


    if keypoints_2d is None or to_ann:
        if keypoints_2d is None:
            keypoints_2d = np.array(skeleton_data.dj)
            h, w = img.shape[:2]
            keypoints_2d *= [w, h]

        while True:
            keypoints_2d = draw.SketcherSkeleton(
                img=add_mask(
                    add_mask(
                        cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                        mask,
                        color=(0, 255, 0),
                        alpha=0.6,
                    ),
                    o_mask,
                    color=(0, 0, 255),
                    alpha=0.6,
                ),
                joints=keypoints_2d,
                skeleton=skeleton_data.skeleton,
                right_inds=skeleton_data.right_inds,
                windowname="skeleton",
            ).annotate()
            _kps = keypoints_2d.astype("int")
            _kps[:, 0] = np.clip(_kps[:, 0], 0, mask.shape[1] - 1)
            _kps[:, 1] = np.clip(_kps[:, 1], 1, mask.shape[0] - 1)
            if mask[*_kps.T[::-1]].all():
                break

            # TODO: make sure bones are inside the foreground mask!"

            print("Make sure 2d joints are inside the foreground mask!")

    if to_show:
        img_to_show = img.copy()
        plot_skel_cv(
            img=img_to_show,
            joints=keypoints_2d.round().astype("int"),
            skeleton=skeleton_data.skeleton,
            joint_names=skeleton_data.joints,
        )
        plt.imshow(cv2.cvtColor(img_to_show, cv2.COLOR_BGR2RGB))
        plt.show()
        plt.close()

    return keypoints_2d


def get_z_order(skeleton_data, img, keypoints_2d, z_order_path, to_ann=False):
    try:
        import s2p.pose
        model = s2p.pose.PoseEstiomator3D(s2p.pose.Args())
        print(f"Using 3D pose estimation model")
        z_order, mesh_3d, _ = model.predict(
            img,
            keypoints_2d,
        )
        z_order = z_order[:, -1]
        mesh_path = Path(z_order_path).with_name(f"{Path(z_order_path).stem}_mesh3d.ply")
        mesh_3d.export(mesh_path)
    except FileNotFoundError:
        print("Not found 3D pose estimation model")
        model = None
    except ModuleNotFoundError as e:
        print(f"{e.msg}. Install sketch2pose to use 3D pose estimation model")
        model = None

    if model is None or to_ann:
        print("Using manual annotation")
        z_order = draw.SketcherZorder(
            img=img,
            joints=keypoints_2d,
            skeleton=skeleton_data.skeleton,
            right_inds=skeleton_data.right_inds,
            windowname="z order",
        ).annotate()

    return z_order


def get_t_junctions(img, mask, o_mask, to_ann=False, to_show=False):
    if not o_mask.any():
        return []

    print("Using manual annotation.")

    contours, *_ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    boundary = np.zeros_like(mask)
    cv2.drawContours(boundary, contours, -1, (255, ))

    img = add_mask(img, o_mask)
    t_jun = draw.SketcherPoint(
        img=add_mask(img, boundary, color=(0, 0, 255)),
        boundary=np.concatenate(contours, axis=0)[:, 0],
        windowname="tjun",
    ).annotate()

    if to_show:
        img_to_show = img.copy()
        plot_skel_cv(
            img=img_to_show,
            t_jun=t_jun.round().astype("int"),
        )
        plt.imshow(cv2.cvtColor(img_to_show, cv2.COLOR_BGR2RGB))
        plt.show()
        plt.close()

    if to_ann:
        contours, *_ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        boundary = np.zeros_like(mask)
        cv2.drawContours(boundary, contours, -1, (255, ))
        t_jun = draw.SketcherPoint(
            img=add_mask(img, boundary, color=(0, 0, 255)),
            boundary=np.concatenate(contours, axis=0)[:, 0],
            points=t_jun.tolist(),
            windowname="tjun",
        ).annotate()

    t_jun = t_jun.astype("int")

    return t_jun


def fit_mask(mask, t_jun, w=100, triangulate_args="pq", to_show=False):
    wh = max(mask.shape)
    if "a" not in triangulate_args:
        triangulate_args = f"{triangulate_args}a{wh / 15}"

    if to_show:
        plt.imshow(mask, cmap="grey")
        plt.scatter(t_jun[:, 0], t_jun[:, 1], color="yellow")

    contours, *_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    assert len(contours) == 1
    contours = np.concatenate(contours, axis=0)[:, 0]

    n = len(contours)
    ran = np.arange(n)
    segments = np.stack([ran, np.roll(ran, -1)], axis=1)

    A = dict(
        vertices=contours,
        segments=segments,
    )
    B = tr.triangulate(A, triangulate_args)
    V = B["vertices"]
    T = B["triangles"]

    if to_show:
        plt.triplot(*V.T, T, ms=10)

    pw = cdist(V, t_jun, metric="euclidean")  # [N, T]
    inds = pw.argmin(axis=0)  # [T]

    x = np.ndarray((V.shape[0], 1))
    y = np.ndarray((V.shape[0], 1))
    fastSymDir.optimizeDeformation(
        x,
        y,
        V,
        T,
        inds,
        t_jun,
        w,
        100_000,
        1e-3,
    )
    V = np.hstack([x, y])

    if to_show:
        plt.title("new tri")
        plt.triplot(*V.T, T)
        plt.show()
        plt.close()

    new_mask = np.zeros_like(mask)
    bnd = igl.boundary_loop(T)
    pts = V[bnd][:, None].round().astype("int")
    cv2.drawContours(new_mask, [pts], -1, (255, ), thickness=cv2.FILLED)

    contours, *_ = cv2.findContours(new_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    new_mask = np.zeros_like(mask)
    cv2.drawContours(new_mask, contours, -1, (255, ), thickness=cv2.FILLED)

    contours = np.concatenate(contours, axis=0)[:, 0]
    poly = Polygon(contours)
    to_repeat = False
    for x, y in t_jun:
        if poly.distance(Point(x, y)) > 0:
            to_repeat = True
            break

    if to_show:
        plt.title("old_mask")
        plt.imshow(mask)
        plt.triplot(*V.T, T)
        plt.scatter(*t_jun.T, color="red")
        plt.show()
        plt.close()

        plt.title("new mask")
        plt.imshow(new_mask)
        plt.triplot(*V.T, T)
        plt.scatter(*t_jun.T, color="red")
        plt.show()
        plt.close()

    return new_mask, to_repeat


def assign_tjun_to_mask(t_jun, o_mask):
    contours, *_ = cv2.findContours(o_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    c_to_jun = {}
    for ti, tjun in enumerate(t_jun):
        pw_min = float("+inf")
        ci_min = None
        for ci, c in enumerate(contours):
            pw = cdist(c[:, 0], tjun[None], metric="euclidean")
            pw = pw.min()
            if pw < pw_min:
                pw_min = pw
                ci_min = ci

        c_to_jun.setdefault(ci_min, []).append(ti)
    print(c_to_jun)
    out = []
    t_jun_new = []
    for ci, ti in c_to_jun.items():
        t_jun_new.extend(t_jun[ti])
        out.append((contours[ci], t_jun[ti]))

    assert len(t_jun_new) == len(t_jun)
    sc = [contours[i] for i in set(range(len(contours))) - set(c_to_jun)]
    assert len(contours) == len(c_to_jun) + len(sc), (len(contours), len(c_to_jun), len(sc))

    return out, sc


def fix_omask(o_mask, mask, t_jun, to_show=False):
    mask_contours, *_ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    mask_contours = np.concatenate(mask_contours, axis=0)[:, 0]

    new_mask = np.zeros_like(o_mask)
    contours_tjun, sc = assign_tjun_to_mask(t_jun, o_mask)

    for c, tjun in contours_tjun:
        mask = np.zeros_like(o_mask)
        cv2.drawContours(mask, [c], -1, (255, ), thickness=cv2.FILLED)

        if len(tjun) == 0:
            cv2.bitwise_or(new_mask, mask, new_mask)
            continue

        while True:
            mask, to_repeat = fit_mask(mask, tjun, to_show=to_show)
            if not to_repeat:
                break

            print("repeating")

        cv2.bitwise_or(new_mask, mask, new_mask)

    # todo: take into acount occlusions without tjunctions
    #for c in sc:
    #    mask = np.zeros_like(o_mask)
    #    cv2.drawContours(mask, [c], -1, (255, ), thickness=cv2.FILLED)
    #    cv2.bitwise_or(new_mask, mask, new_mask)

    if to_show:
        plt.title("new_mask")
        plt.imshow(new_mask)
        plt.show()
        plt.close()

    return new_mask


def add_mask(img, mask, color=(255, 0, 0), alpha=0.5, soft=False):
    if soft:
        mask = (mask * 255).astype("uint8")
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    else:
        mask = np.where(
            mask[..., None],
            np.array(color, dtype=img.dtype),
            img,
        )
    img = cv2.addWeighted(
        img,
        alpha,
        mask,
        1 - alpha,
        0.0,
    )

    return img


def remove_touching_pixels(o_mask, mask, wh=1, t_jun=None):
    cv2.bitwise_and(mask, o_mask, o_mask)
    mask_contours, *_ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cont = np.zeros_like(mask)
    cv2.drawContours(cont, mask_contours, -1, (255, ), thickness=wh)
    cv2.bitwise_and(o_mask, cv2.bitwise_not(cont), o_mask)
    if t_jun is not None:
        for x, y in t_jun:
            x1 = x - wh
            y1 = y - wh
            x2 = x + wh
            y2 = y + wh
            cv2.rectangle(o_mask, (x1, y1), (x2, y2), (255, ), cv2.FILLED)

    cv2.bitwise_and(mask, o_mask, o_mask)


def get_img_masks_keypoints(
    skeleton_data,
    img_path,
    model_2d_pose_estimation_path,
    model_mask_path,
    use_o_mask_gt=False,
    to_show=False,
    is_guid=False,
    touch_pixels=2,
):
    img_path = Path(img_path)

    img = read_img(img_path)

    h, w = img.shape[:2]

    mask = get_mask(img_path)

    if mask is not None:
        assert img.shape[:2] == mask.shape[:2], (img.shape[:2], mask.shape[:2])
        _, bbox = crop_img_by_mask(img, mask)
    else:
        bbox = draw.BBox(
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
            windowname="bbox",
        ).annotate()

    if not is_guid:
        mask, o_mask = get_overlapping_mask(
            img_path,
            model_mask_path,
            bbox=bbox,
            thresh=0.8,
            use_gt=use_o_mask_gt,
            to_show=to_show,
            use_slider=True,
        )
        assert img.shape[:2] == mask.shape[:2]
    else:
        o_mask = mask

    # removes small dots
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    #o_mask = cv2.dilate(o_mask, kernel, iterations=1)
    #o_mask = cv2.erode(o_mask, kernel, iterations=1)

    #o_mask = dilate(o_mask)
    assert img.shape[:2] == o_mask.shape[:2], (img.shape, o_mask.shape)

    # erode
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    #o_mask = cv2.erode(o_mask, kernel, iterations=5)

    ## erode
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    #o_mask = cv2.erode(o_mask, kernel, iterations=1)

    # remove boundary pixels of occlusion mask
    if not is_guid:
        remove_touching_pixels(o_mask, mask, wh=touch_pixels)

    json_path = img_path.with_suffix(".json")
    if json_path.is_file():
        metadata = write_utils.load_json(json_path)
    else:
        metadata = {
            "joints": skeleton_data.joints,
            "root": skeleton_data.root_name,
            "skeleton": skeleton_data.skeleton,
        }

    if bbox not in metadata:
        metadata["bbox"] = {
            "y_min": bbox[0] / h,
            "x_min": bbox[1] / w,
            "y_max": bbox[2] / h,
            "x_max": bbox[3] / w,
        }

    bbox = [
        int(metadata["bbox"]["y_min"] * h),
        int(metadata["bbox"]["x_min"] * w),
        int(metadata["bbox"]["y_max"] * h),
        int(metadata["bbox"]["x_max"] * w),
    ]

    if "kps2d" not in metadata:
        keypoints_2d = get_skeleton_2d(
            skeleton_data,
            img,
            model_2d_pose_estimation_path,
            mask=mask,
            o_mask=o_mask,
            bbox=bbox,
            to_ann=True,
            to_show=to_show,
        )
        metadata["kps2d"] = {
            j: (x / w, y / h)
            for j, (x, y) in zip(skeleton_data.joints, keypoints_2d.tolist())
        }
        write_utils.save_json(metadata, json_path, indent=4)

    keypoints_2d = np.array(
        [
            metadata["kps2d"][k]
            for k in metadata["joints"]
        ],
    )
    assert 0 <= keypoints_2d.max() <= 1
    keypoints_2d *= [w, h]

    # TODO: make sure bones are inside the foreground mask!"
    if not mask[*keypoints_2d.astype("int").T[::-1]].all():
        # ignore this rise for dvor_lady
        raise RuntimeError(f"Make sure 2d joints are inside the foreground mask! (remove kps2d from {json_path})")
        #keypoints_2d = draw.SketcherSkeleton(
        #        img=add_mask(
        #            add_mask(
        #                img,
        #                mask,
        #                color=(0, 255, 0),
        #                alpha=0.6,
        #            ),
        #            o_mask,
        #            color=(0, 0, 255),
        #            alpha=0.6,
        #        ),
        #        joints=keypoints_2d,
        #        skeleton=skeleton_data.skeleton,
        #        right_inds=skeleton_data.right_inds,
        #        windowname="skeleton",
        #    ).annotate()
        #    metadata["kps2d"] = {
        #        j: (x / w, y / h)
        #        for j, (x, y) in zip(skeleton_data.joints, keypoints_2d.tolist())
        #    }
        #    write_utils.save_json(metadata, json_path, indent=4)

        #    keypoints_2d = np.array(
        #        [
        #            metadata["kps2d"][k]
        #            for k in metadata["joints"]
        #        ],
        #    )
        #    keypoints_2d *= [w, h]

    print(f"{keypoints_2d=}")

    if "z_order" not in metadata:
        z_order_path = img_path.with_name(f"{Path(img_path).stem}_kps3d.txt")
        z_order = get_z_order(skeleton_data, img, keypoints_2d, z_order_path, to_ann=False)
        metadata["z_order"] = {
            j: xy
            for j, xy in zip(skeleton_data.joints, z_order.tolist())
        }
        write_utils.save_json(metadata, json_path, indent=4)

    z_order = np.array(
        [
            metadata["z_order"][k]
            for k in metadata["joints"]
        ]
    )

    print(f"{z_order=}")

    if not is_guid:
        if "t_jun" not in metadata:
            t_jun = get_t_junctions(
                img,
                mask,
                o_mask,
                to_ann=True,
                to_show=to_show,
            )
            if len(t_jun) > 0:
                t_jun = np.array(t_jun) / [w, h]

            metadata["t_jun"] = t_jun if isinstance(t_jun, list) else t_jun.tolist()
            write_utils.save_json(metadata, json_path, indent=4)

        t_jun = np.array(metadata["t_jun"])
        if len(t_jun) > 0:
            assert 0 <= t_jun.max() <= 1
            t_jun *= [w, h]

        #np.set_printoptions(precision=16)
        t_jun = t_jun.round().astype("int")
        print(t_jun)

        if "bone_to_top" not in metadata:
            if o_mask.any():
                to_hl = []
                contours, *_ = cv2.findContours(o_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                for cv in contours:
                    cv = cv[:, 0]
                    c = np.vstack([cv, cv[[0]]])
                    if len(c) < 4:
                        print(c)
                        plt.imshow(o_mask)
                        plt.show()
                        plt.close()
                        continue

                    poly = Polygon(c)

                    for i, (a, b) in enumerate(skeleton_data.skeleton):
                        a = keypoints_2d[a]
                        b = keypoints_2d[b]
                        if poly.intersects(LineString([a, b])):
                            to_hl.append(i)

                bone_to_top = draw.SketcherBoneToTop(
                    img=add_mask(img, o_mask),
                    joints=keypoints_2d,
                    skeleton=skeleton_data.skeleton,
                    right_inds=skeleton_data.right_inds,
                    to_hl=to_hl,
                    windowname="bones to top",
                ).annotate()
            else:
                bone_to_top = []

            metadata["bone_to_top"] = [
                [
                    skeleton_data.joints[skeleton_data.skeleton[bi][0]],
                    skeleton_data.joints[skeleton_data.skeleton[bi][1]],
                ]
                for bi in bone_to_top
            ]
            write_utils.save_json(metadata, json_path, indent=4)

        bone_to_top = [
            metadata["skeleton"].index(
                [
                    metadata["joints"].index(a),
                    metadata["joints"].index(b),
                ]
            )
            for a, b in metadata["bone_to_top"]
        ]

        print(f"{bone_to_top=}")

        # TODO: fix only for non-boundary o_mask <05-12-23 kbrodt> #
        o_mask = fix_omask(o_mask, mask, t_jun, to_show=to_show)
        remove_touching_pixels(o_mask, mask, t_jun=t_jun, wh=touch_pixels)
    else:
        bone_to_top = []
        t_jun = []

    if to_show:
        alpha = 0.6
        green = np.array([0, 255, 0], dtype="uint8")
        red = np.array([0, 0, 255], dtype="uint8")
        img_to_show = add_mask(img, mask, color=green, alpha=alpha)
        img_to_show = add_mask(img_to_show, o_mask, color=red, alpha=alpha)
        plt.imshow(cv2.cvtColor(img_to_show, cv2.COLOR_BGR2RGB))
        plt.scatter(*keypoints_2d.T, color="blue")
        for a, b in skeleton_data.skeleton:
            if "right" in skeleton_data.joints[a].lower():
                color = "red"
            else:
                color = "blue"

            plt.plot(*keypoints_2d[[a, b]].T, color=color)
        #plt.scatter(*t_jun.T, color="red")
        plt.show()
        plt.close()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    assert keypoints_2d.shape == (len(skeleton_data.joints), 2), (keypoints_2d, img_path)
    assert z_order.shape == (len(skeleton_data.joints), ), (z_order, img_path)

    return img, mask, o_mask, keypoints_2d, z_order, bone_to_top, t_jun


def main():
    args = parse_args()
    print(args)

    set_seed(args.seed)

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    to_show = args.to_show
    img_path = args.img_paths[0]

    skeleton_data = SkeletonData.from_json(args.character_topology_path)

    # TODO: not all bottom joints are inside overlaping area <05-12-23 kbrodt> #
    img_0, mask_0, o_mask_0, keypoints_2d_0, z_order_0, bone_to_top_0, t_jun_0 = get_img_masks_keypoints(
        skeleton_data,
        img_path,
        args.pose_estimation_model_path,  # "hrn_w48_384x288.onnx"
        args.segmentation_model_path,
        use_o_mask_gt=args.use_o_mask_gt,
        to_show=to_show,
        touch_pixels=args.touch_pixels,
    )

    use_optuna = not args.no_optuna
    optuna.logging.disable_default_handler()

    inb_0 = Inbetweener(
        skeleton_data,
        img_0,
        mask_0,
        o_mask_0,
        keypoints_2d_0,
        z_order_0,
        bone_to_top_0,
        t_jun=t_jun_0,
        n_pts=args.n_pts,
        inpaint_method=args.inpaint_method,
        epsilon=args.epsilon,
        to_show=to_show,
        device=device,
        nearest=args.use_nearest,
    )
    for i, img_path in enumerate(args.img_paths[1:]):
        print(f"processing image {img_path}")
        img_1, mask_1, o_mask_1, keypoints_2d_1, z_order_1, bone_to_top_1, t_jun_1 = get_img_masks_keypoints(
            skeleton_data,
            img_path,
            args.pose_estimation_model_path,
            args.segmentation_model_path,
            use_o_mask_gt=args.use_o_mask_gt,
            to_show=to_show,
            touch_pixels=args.touch_pixels,
        )

        if args.guidance_paths is not None:
            intermediate = []
            for _pg in args.guidance_paths:
                img_g, _, _, keypoints_2d_g, z_order_g, _, _ = get_img_masks_keypoints(
                    skeleton_data,
                    _pg,
                    args.pose_estimation_model_path,
                    args.segmentation_model_path,
                    use_o_mask_gt=args.use_o_mask_gt,
                    to_show=to_show,
                    is_guid=True,
                    touch_pixels=args.touch_pixels,
                )
                intermediate.append((img_g, keypoints_2d_g, z_order_g))
            #img_g, _, _, keypoints_2d_g, z_order_g, _, _ = get_img_masks_keypoints(
            #    skeleton_data,
            #    args.guidance_paths[i],
            #    args.pose_estimation_model_path,
            #    args.segmentation_model_path,
            #    use_o_mask_gt=args.use_o_mask_gt,
            #    to_show=to_show,
            #    touch_pixels=args.touch_pixels,
            #)
        else:
            img_g = None
            keypoints_2d_g = None
            z_order_g = None
            intermediate = [(img_g, keypoints_2d_g, z_order_g)]

        inb_1 = inb_0.forward(
            img_1,
            mask_1,
            o_mask_1,
            keypoints_2d_1,
            z_order_1,
            bone_to_top_1,
            t_jun_1,
            out_dir=out_dir / f"fwd_{i:0>3}",
            img_g=img_g,
            keypoints_2d_g=keypoints_2d_g,
            z_order_g=z_order_g,
            to_show=to_show,
            use_optuna=use_optuna,
        )

        # backward
        inb_1.forward(
            img_0,
            mask_0,
            o_mask_0,
            keypoints_2d_0,
            z_order_0,
            bone_to_top_0,
            t_jun_0,
            out_dir=out_dir / f"bwd_{i:0>3}",
            img_g=img_g,
            keypoints_2d_g=keypoints_2d_g,
            z_order_g=z_order_g,
            to_show=to_show,
            use_optuna=use_optuna,
        )

        if args.animation_paths is not None:
            assert img_g is None, "Guidance and BVH animation are not supported at the same time"
            assert not args.interactive, "Interactive is not supported with BVH animation at the same time"
            animation_path = args.animation_paths[i]
        else:
            animation_path = None

        interpolator = Interpolator(
            inb_0,
            inb_1,
            fi_model_path=args.frame_interpolation_model_path if args.interactive else None,
        )
        interpolator.interpolate(
            out_dir / f"int_{i:0>3}",
            animation_path=animation_path,
            n_steps=args.n_steps,
            #intermediate=[
                #(img_g, keypoints_2d_g, z_order_g),
                #],
            intermediate=intermediate,
            interactive=args.interactive,
            deform_method=args.deform_method,
            dim=2,
        )

        (
            img_0, mask_0, o_mask_0, keypoints_2d_0, z_order_0, bone_to_top_0, t_jun_0,
        ) = (
            img_1, mask_1, o_mask_1, keypoints_2d_1, z_order_1, bone_to_top_1, t_jun_1,
        )
        inb_0 = inb_1


if __name__ == "__main__":
    main()
