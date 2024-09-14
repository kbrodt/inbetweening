import sys
import copy
import itertools

import cv2
import numpy as np
from shapely.geometry import (
    Point,
    LineString,
)
from scipy.spatial.distance import cdist


DJ = np.array([
    [0.5, 0.1],  # nead
    [0.5, 0.2],  # neck

    [0.7, 0.3],  # r shoulder
    [0.8, 0.4],  # r arm
    [0.8, 0.5],  # r hand

    [0.4, 0.3],  # l shoulder
    [0.3, 0.4],  # l arm
    [0.3, 0.5],  # l han

    [0.5, 0.3],  # spine
    [0.5, 0.4],  # hips

    [0.7, 0.6],  # r upper leg
    [0.8, 0.7],  # r leg
    [0.8, 0.8],  # r foot

    [0.4, 0.6],  # l upper leg
    [0.3, 0.7],  # l leg
    [0.3, 0.8],  # l foot

    [0.2, 0.8],  # l toe

    [0.9, 0.8],  # r toe
])

SKELETON = (
    (1, 0),    # 0 neck head
    (1, 8),    # 1 neck spine
    (8, 9),    # 2 spine hips
    (9, 10),   # 3 hips right upper leg
    (9, 13),   # 4 hips left upper leg
    (10, 11),  # 5 right upper leg leg
    (11, 12),  # 6 right leg foot
    (13, 14),  # 7 left upper leg leg
    (14, 15),  # 8 left leg foot
    (1, 2),    # 9 neck right shoulder
    (2, 3),    # 10 right shoulder arm
    (3, 4),    # 11 right arm hand
    (1, 5),    # 12 neck left shoulder
    (5, 6),    # 13 left shoulder arm
    (6, 7),    # 14 left arm hand
    (15, 16),  # 15 left foot toe
    (12, 17),  # 16 right foot toe
)  # 11, 10, 9 14, 13, 12, 16, 6, 5, 4, 15, 8, 7, 6, 0, 1, 2

BLUE_RGB = (85, 153, 255)
RED_RGB = (255, 85, 85)


class SketcherPoint:
    def __init__(
        self,
        img,
        boundary,
        points=None,
        color=BLUE_RGB[::-1],
        windowname="point",
        max_w=800,
    ):

        self.points = points if points is not None else []
        self.pt_size = int(0.0075 * max(img.shape[:2]))
        self.windowname = windowname
        self.orig_img = img.copy()
        self.img = self.orig_img.copy()
        self.boundary = boundary.copy()
        self.color = color
        self.i = None

        cv2.namedWindow(self.windowname, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.windowname, self.on_mouse)
        self.show()
        h, w = img.shape[:2]
        r = max_w / h
        w = int(r * w)
        h = int(r * h)
        cv2.resizeWindow(self.windowname, w, h)

    def show(self):
        self.refresh()

        for x, y in self.points:
            x = int(x)
            y = int(y)
            cv2.circle(self.img, (x, y), self.pt_size, self.color, lineType=cv2.LINE_AA)

        cv2.imshow(self.windowname, self.img)

    def _on_mouse(self, event, x, y, flags, param):
        pt = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.prev_pt = pt
        elif event == cv2.EVENT_LBUTTONUP:
            self.prev_pt = None

        if self.prev_pt and flags & cv2.EVENT_FLAG_LBUTTON:
            cv2.line(self.img, self.prev_pt, pt, self.color, self.pt_size, lineType=cv2.LINE_AA)

            #self.dirty = True
            self.prev_pt = pt
            self.show()

            self.points.append(self.prev_pt)

    def on_mouse(self, event, x, y, flags, param):
        pt = np.array((x, y))
        if event == cv2.EVENT_LBUTTONDOWN:
            pw = cdist(self.boundary, pt[None])
            i, _ = np.unravel_index(pw.argmin(), pw.shape)
            if pw[i] < self.pt_size:
                pt = self.boundary[i]
                if len(self.points) == 0:
                    self.points.append(pt)
                    self.show()

                pw = cdist(np.array(self.points), pt[None])
                i, _ = np.unravel_index(pw.argmin(), pw.shape)
                if pw[i] < self.pt_size:
                    self.i = i
                    #self.d = self.points[self.i] - pt
                else:
                    self.i = None
                    #self.d = 0
                    self.points.append(pt)
                    self.show()

        elif self.i is not None and event == cv2.EVENT_MOUSEMOVE:
            pw = cdist(self.boundary, pt[None])
            i, _ = np.unravel_index(pw.argmin(), pw.shape)
            pt = self.boundary[i]
            self.points[self.i] = pt
            self.show()

        elif event == cv2.EVENT_LBUTTONUP:
            self.i = None
            #self.d = 0

    def refresh(self):
        self.img = self.orig_img.copy()

    def reset(self):
        self.img = self.orig_img.copy()
        self.points.clear()
        self.show()

    def undo(self):
        if len(self.points) > 0:
            self.points.pop()
            self.show()

    def annotate(self):
        while True:
            ch = cv2.waitKey()
            if ch == ord("q"):  # Escape
                break

            if ch == ord("r"):
                self.reset()

            if ch == ord("u"):
                self.undo()

        cv2.destroyWindow(self.windowname)

        return np.array(self.points)


class SketcherSkeleton:
    def __init__(
        self,
        img,
        img_fn=None,
        joints=None,
        skeleton=None,
        right_inds=None,
        color=BLUE_RGB[::-1],
        rcolor=RED_RGB[::-1],
        windowname="skeleton",
        max_w=800,
    ):
        self.pt_size = int(0.0075 * max(img.shape[:2]))
        self.prev_pt = None
        self.windowname = windowname
        self.orig_img = img.copy()

        joints = joints if joints is not None else DJ * np.array(img.shape[:2])[None, ::-1]
        skeleton = skeleton if skeleton is not None else SKELETON

        self.orig_joints = copy.deepcopy(joints)
        self.skeleton = copy.deepcopy(skeleton)
        self.img = self.orig_img.copy()
        self.joints = copy.deepcopy(self.orig_joints)

        self.right_inds = right_inds if right_inds is not None else []
        self.color = color
        self.rcolor = rcolor
        self.drawing = False
        self.i = None

        self.img_fn = img_fn if img_fn is not None else lambda *_: self.orig_img.copy()

        cv2.namedWindow(self.windowname, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.windowname, self.on_mouse)
        self.show()
        h, w = img.shape[:2]
        r = max_w / h
        w = int(r * w)
        h = int(r * h)
        cv2.resizeWindow(self.windowname, w, h)

    def show(self):
        img = self.img_fn(self.joints)

        for x, y in self.joints:
            x = int(x)
            y = int(y)
            cv2.circle(img, (x, y), self.pt_size, self.color, lineType=cv2.LINE_AA)

        for bi, (i, j) in enumerate(self.skeleton):
            a = self.joints[i].astype("int")
            b = self.joints[j].astype("int")
            if bi in self.right_inds:
                cv2.line(img, a, b, self.rcolor, lineType=cv2.LINE_AA)
            else:
                cv2.line(img, a, b, self.color, lineType=cv2.LINE_AA)

        cv2.imshow(self.windowname, img)

    def on_mouse(self, event, x, y, flags, param):
        pt = np.array([[x, y]])
        if event == cv2.EVENT_LBUTTONDOWN:
            pw = cdist(self.joints, pt)
            i, _ = np.unravel_index(pw.argmin(), pw.shape)
            if pw[i] < self.pt_size:
                self.i = i
                self.d = self.joints[self.i] - pt
            else:
                self.i = None
                self.d = 0

        elif self.i is not None and event == cv2.EVENT_MOUSEMOVE:
            self.joints[self.i] = pt + self.d
            self.show()
        elif event == cv2.EVENT_LBUTTONUP:
            self.i = None
            self.d = 0

    def reset(self):
        self.joints = copy.deepcopy(self.orig_joints)
        self.show()

    def undo(self):
        self.show()

    def annotate(self):
        while True:
            ch = cv2.waitKey()
            if ch == ord("q"):  # Escape
                break

            if ch == ord("r"):
                self.reset()

            if ch == ord("u"):
                self.undo()

        cv2.destroyWindow(self.windowname)

        return np.array(self.joints)


class SketcherBoneToTop:
    def __init__(
        self,
        img,
        joints=None,
        skeleton=None,
        right_inds=None,
        color=BLUE_RGB[::-1],
        rcolor=RED_RGB[::-1],
        to_hl=None,
        windowname="bone_to_top",
        max_w=800,
    ):
        self.pt_size = int(0.0075 * max(img.shape[:2]))
        self.prev_pt = None
        self.windowname = windowname
        self.orig_img = img.copy()

        joints = joints if joints is not None else DJ * np.array(img.shape[:2])[None, ::-1]
        skeleton = skeleton if skeleton is not None else SKELETON

        self.orig_joints = copy.deepcopy(joints)
        self.skeleton = copy.deepcopy(skeleton)
        self.img = self.orig_img.copy()
        self.joints = copy.deepcopy(self.orig_joints)
        self.right_inds = right_inds if right_inds is not None else []
        self.color = color
        self.rcolor = rcolor
        self.drawing = False
        self.i = None

        self.hl_bones = [False] * len(self.skeleton)
        self.to_hl = to_hl if to_hl is not None else []
        self.bones = []
        for bi in self.to_hl:
            i, j = self.skeleton[bi]
            a = self.joints[i].astype("int")
            b = self.joints[j].astype("int")
            self.bones.append(LineString([a, b]))

        cv2.namedWindow(self.windowname, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.windowname, self.on_mouse)
        self.show()
        h, w = img.shape[:2]
        r = max_w / h
        w = int(r * w)
        h = int(r * h)
        cv2.resizeWindow(self.windowname, w, h)

    def show(self):
        self.refresh()
        overlay = self.orig_img.copy()
        # https://stackoverflow.com/questions/69432439/how-to-add-transparency-to-a-line-with-opencv-python

        for bi, (i, j) in enumerate(self.skeleton):
            a = self.joints[i].astype("int")
            b = self.joints[j].astype("int")
            if self.hl_bones[bi]:
                cv2.line(overlay, a, b, (0, 255, 0), 2, lineType=cv2.LINE_AA)
            elif bi in self.to_hl:
                if bi in self.right_inds:
                    cv2.line(overlay, a, b, self.rcolor, 2, lineType=cv2.LINE_AA)
                else:
                    cv2.line(overlay, a, b, self.color, 2, lineType=cv2.LINE_AA)
            else:
                if bi in self.right_inds:
                    cv2.line(self.img, a, b, self.rcolor, 2, lineType=cv2.LINE_AA)
                else:
                    cv2.line(self.img, a, b, self.color, 2, lineType=cv2.LINE_AA)

        alpha = 0.3
        self.img = cv2.addWeighted(self.img, alpha, overlay, 1 - alpha, 0)
        cv2.imshow(self.windowname, self.img)

    def on_mouse(self, event, x, y, flags, param):
        pt = Point(x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, bone in enumerate(self.bones):
                pw = bone.distance(pt)
                if pw < self.pt_size:
                    i = self.to_hl[i]
                    self.hl_bones[i] = not self.hl_bones[i]
                    self.show()
                    break

    def refresh(self):
        self.img = self.orig_img.copy()

    def reset(self):
        self.img = self.orig_img.copy()
        self.joints = copy.deepcopy(self.orig_joints)
        self.show()

    def undo(self):
        self.show()

    def annotate(self):
        while True:
            ch = cv2.waitKey()
            if ch == ord("q"):  # Escape
                break

            if ch == ord("r"):
                self.reset()

            if ch == ord("u"):
                self.undo()

        cv2.destroyWindow(self.windowname)

        return [i for i, m in enumerate(self.hl_bones) if m]


class SketcherZorder:
    def __init__(
        self,
        img,
        joints=None,
        skeleton=None,
        right_inds=None,
        color=BLUE_RGB[::-1],
        rcolor=RED_RGB[::-1],
        windowname="zorder",
        max_w=800,
    ):
        self.pt_size = int(0.01 * max(img.shape[:2]))
        self.prev_pt = None
        self.windowname = windowname
        self.orig_img = img.copy()

        joints = joints if joints is not None else DJ * np.array(img.shape[:2])[None, ::-1]
        skeleton = skeleton if skeleton is not None else SKELETON

        self.orig_joints = copy.deepcopy(joints)
        self.skeleton = copy.deepcopy(skeleton)
        self.img = self.orig_img.copy()
        self.joints = copy.deepcopy(self.orig_joints)
        self.right_inds = right_inds if right_inds is not None else []
        self.color = color
        self.rcolor = rcolor
        self.drawing = False

        self.jcolors = [
            itertools.cycle(
                [
                    (-0.1, (255, 0, 0)),
                    #(255, 255, 0),
                    (0, (0, 255, 0)),
                    #(0, 100, 200),
                    (0.1, (0, 0, 255)),
                ]
            )
            for _ in range(len(joints))
        ]
        self.jcolor = [
            next(color)
            for color in self.jcolors
        ]
        self.jcolor = [
            next(color)
            for color in self.jcolors
        ]

        cv2.namedWindow(self.windowname, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.windowname, self.on_mouse)
        self.show()
        h, w = img.shape[:2]
        r = max_w / h
        w = int(r * w)
        h = int(r * h)
        cv2.resizeWindow(self.windowname, w, h)

    def show(self):
        self.refresh()
        overlay = self.orig_img.copy()

        for (_, color), (x, y) in zip(self.jcolor, self.joints):
            x = int(x)
            y = int(y)
            cv2.circle(overlay, (x, y), self.pt_size, color, thickness=-1, lineType=cv2.LINE_AA)

        for bi, (i, j) in enumerate(self.skeleton):
            a = self.joints[i].astype("int")
            b = self.joints[j].astype("int")
            if bi in self.right_inds:
                cv2.line(self.img, a, b, self.rcolor, 2, lineType=cv2.LINE_AA)
            else:
                cv2.line(self.img, a, b, self.color, 2, lineType=cv2.LINE_AA)

        alpha = 0.3
        self.img = cv2.addWeighted(self.img, alpha, overlay, 1 - alpha, 0)
        cv2.imshow(self.windowname, self.img)

    def on_mouse(self, event, x, y, flags, param):
        pt = np.array([[x, y]])
        if event == cv2.EVENT_LBUTTONDOWN:
            pw = cdist(self.joints, pt)
            i, _ = np.unravel_index(pw.argmin(), pw.shape)
            if pw[i] < self.pt_size:
                self.jcolor[i] = next(self.jcolors[i])
                self.show()

    def refresh(self):
        self.img = self.orig_img.copy()

    def reset(self):
        self.img = self.orig_img.copy()
        self.joints = copy.deepcopy(self.orig_joints)
        self.show()

    def undo(self):
        self.show()

    def annotate(self):
        while True:
            ch = cv2.waitKey()
            if ch == ord("q"):  # Escape
                break

            if ch == ord("r"):
                self.reset()

            if ch == ord("u"):
                self.undo()

        cv2.destroyWindow(self.windowname)

        return np.array([i for i, _ in self.jcolor])


class BBox:
    def __init__(
        self,
        img,
        color=BLUE_RGB[::-1],
        alpha=0.5,
        windowname="bbox",
        max_w=800,
    ):
        self.pt_size = int(0.0075 * max(img.shape[:2]))
        self.windowname = windowname
        self.orig_img = img.copy()
        self.img = self.orig_img.copy()
        self.color = color
        self.pt1 = None
        self.pt2 = None

        cv2.namedWindow(self.windowname, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.windowname, self.on_mouse)
        self.show()
        h, w = img.shape[:2]
        r = max_w / h
        w = int(r * w)
        h = int(r * h)
        cv2.resizeWindow(self.windowname, w, h)

    def show(self):
        self.refresh()
        overlay = self.orig_img.copy()
        # https://stackoverflow.com/questions/69432439/how-to-add-transparency-to-a-line-with-opencv-python
        if self.pt1 is not None and self.pt2 is not None:
            x1, y1 = self.pt1
            x2, y2 = self.pt2
            cv2.rectangle(overlay, (x1, y1), (x2, y2), self.color, thickness=self.pt_size, lineType=cv2.LINE_AA)
            cv2.circle(overlay, (x2, y2), self.pt_size, self.color[::-1], lineType=cv2.LINE_AA)
            cv2.circle(overlay, (x1, y1), self.pt_size, self.color, lineType=cv2.LINE_AA)

        alpha = 0.3
        self.img = cv2.addWeighted(self.img, alpha, overlay, 1 - alpha, 0)
        cv2.imshow(self.windowname, self.img)

    def on_mouse(self, event, x, y, flags, param):
        pt = np.array((x, y))
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.pt1 is None:
                self.pt1 = pt
            elif self.pt2 is not None:
                pw = cdist(self.pt2[None], pt[None])
                if pw.item() < self.pt_size:
                    self.pt2 = pt

        elif event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON:
            self.pt2 = pt
            self.show()

        elif event == cv2.EVENT_LBUTTONUP:
            self.pt2 = pt
            self.show()

    def refresh(self):
        self.img = self.orig_img.copy()

    def reset(self):
        self.img = self.orig_img.copy()
        self.pt1 = None
        self.pt2 = None
        self.show()

    def annotate(self):
        while True:
            ch = cv2.waitKey()
            if ch == ord("q"):  # Escape
                break

            if ch == ord("r"):
                self.reset()

        cv2.destroyWindow(self.windowname)

        bbox = self.pt1[1], self.pt1[0], self.pt2[1], self.pt2[0]

        return bbox


class OMaskSlider:
    def __init__(
        self,
        img,
        mask,
        color=BLUE_RGB[::-1],
        alpha=0.5,
        windowname="mask",
        max_w=800,
    ):
        self.windowname = windowname
        self.img = img.copy()
        self.mask = mask.copy()
        self.color = np.array(color, dtype=self.img.dtype)
        self.alpha = alpha

        cv2.namedWindow(self.windowname, cv2.WINDOW_NORMAL)
        cv2.createTrackbar("threshold", self.windowname , 0, 100, self.on_trackbar)
        self.on_trackbar(80)
        h, w = img.shape[:2]
        r = max_w / h
        w = int(r * w)
        h = int(r * h)
        cv2.resizeWindow(self.windowname, w, h)

    def on_trackbar(self, thresh=80):
        self.thresh = thresh / 100
        self.res = self.mask > self.thresh
        mask = np.where(
            self.res[..., None],
            self.color,
            self.img,
        )
        self.res = (self.res * 255).astype("uint8")
        img = cv2.addWeighted(
            self.img,
            self.alpha,
            mask,
            1 - self.alpha,
            0.0,
        )
        cv2.imshow(self.windowname, img)

    def reset(self):
        self.on_trackbar(80)

    def annotate(self):
        while True:
            ch = cv2.waitKey()
            if ch == ord("q"):  # Escape
                break

            if ch == ord("r"):
                self.reset()

        cv2.destroyWindow(self.windowname)

        return self.res, self.thresh


class SketcherInb:
    def __init__(
        self,
        img,
        animation,
        deform_methods,
        root_ind,
        img_fn=None,
        skeleton=None,
        right_inds=None,
        color=BLUE_RGB[::-1],
        rcolor=RED_RGB[::-1],
        windowname="skeleton",
        max_w=800,
    ):
        self.pt_size = int(0.009 * max(img.shape[:2]))
        self.prev_pt = None
        self.windowname = windowname
        self.orig_img = img.copy()

        skeleton = skeleton if skeleton is not None else SKELETON

        self.orig_animation = copy.deepcopy(animation)
        self.skeleton = copy.deepcopy(skeleton)
        self.img = self.orig_img.copy()

        self.right_inds = right_inds if right_inds is not None else []
        self.color = color
        self.rcolor = rcolor
        self.drawing = False
        self.i = None
        self.deforms = itertools.cycle(deform_methods)
        self.deform = next(self.deforms)
        self.root_ind = root_ind

        self.img_fn = img_fn if img_fn is not None else lambda *_: self.orig_img.copy()
        self.animation = copy.deepcopy(self.orig_animation)
        self.step = 0

        self.imgs = [None] * len(self.animation)
        self.show_adj = False

        cv2.namedWindow(self.windowname, cv2.WINDOW_NORMAL)
        cv2.createTrackbar("step", self.windowname , 0, len(self.animation) - 1, self.on_trackbar)
        cv2.setMouseCallback(self.windowname, self.on_mouse)
        self.show()
        h, w = img.shape[:2]
        r = max_w / h
        w = int(r * w)
        h = int(r * h)
        cv2.resizeWindow(self.windowname, w, h)

    @property
    def joints(self):
        return self.animation[self.step]

    @joints.setter
    def joints(self, joints):
        self.animation[self.step] = joints.copy()

    def on_trackbar(self, step=0):
        self.step = step
        self.show(recompute=False, hq=False)

    def show(self, recompute=False, hq=False):
        if hq or self.imgs[self.step] is None or recompute:
            self.imgs[self.step] = self.img_fn(self.joints, self.step, self.deform, hq=hq)

        img = self.imgs[self.step]

        overlay = img.copy()
        alpha = 0.2
        if self.show_adj and self.step > 0 and self.imgs[self.step - 1] is not None:
            overlay = cv2.addWeighted(
                self.imgs[self.step - 1],
                alpha,
                overlay,
                1 - alpha,
                0,
            )
        if self.show_adj and self.step + 1 < len(self.animation) and self.imgs[self.step + 1] is not None:
            overlay = cv2.addWeighted(
                self.imgs[self.step + 1],
                alpha,
                overlay,
                1 - alpha,
                0,
            )

        for x, y in self.joints:
            x = int(x)
            y = int(y)
            cv2.circle(overlay, (x, y), self.pt_size, self.color, lineType=cv2.LINE_AA)

        for bi, (i, j) in enumerate(self.skeleton):
            a = self.joints[i].astype("int")
            b = self.joints[j].astype("int")
            if bi in self.right_inds:
                cv2.line(overlay, a, b, self.rcolor, lineType=cv2.LINE_AA)
            else:
                cv2.line(overlay, a, b, self.color, lineType=cv2.LINE_AA)

        img = cv2.addWeighted(img, alpha, overlay, 1 - alpha, 0)
        cv2.imshow(self.windowname, img)

    def on_mouse(self, event, x, y, flags, param):
        pt = np.array([[x, y]])
        if event == cv2.EVENT_LBUTTONDOWN:
            pw = cdist(self.joints, pt)
            i, _ = np.unravel_index(pw.argmin(), pw.shape)
            if pw[i] < self.pt_size:
                self.i = i
                self.d = self.joints[self.i] - pt
            else:
                self.i = None
                self.d = 0

        elif self.i is not None and event == cv2.EVENT_MOUSEMOVE:
            if self.i == self.root_ind:
                d = pt + self.d - self.joints[self.i]
                self.joints += d
            else:
                self.joints[self.i] = pt + self.d

            self.show(recompute=True)
        elif event == cv2.EVENT_LBUTTONUP:
            self.i = None
            self.d = 0
            self.show(hq=True)

    def reset(self, step=None):
        if step is None:
            self.animation = copy.deepcopy(self.orig_animation)
        else:
            self.animation[step] = copy.deepcopy(self.orig_animation[step])

        self.show(hq=True)

    def annotate(self):
        while True:
            ch = cv2.waitKey()
            if ch == ord("q"):  # Escape
                break

            if ch == ord("d"):
                self.deform = next(self.deforms)
                print(f"Using {self.deform} deformation")
                self.show(hq=True)

            if ch == ord("r"):
                self.reset(self.step)

            if ch == ord("a"):
                self.reset()

            if ch == ord("h"):
                self.show(hq=True)

            if ch == ord("s"):
                self.show_adj = not self.show_adj
                self.show()

        cv2.destroyWindow(self.windowname)

        return self.animation


def main():
    img = cv2.imread(sys.argv[1], -1)
    pts = BBox(img)

    #img = np.zeros((512, 512, 3), dtype=np.uint8)
    for x, y in pts:
        cv2.circle(
            img,
            (x, y),
            int(0.0075 * max(img.shape[:2])),
            255,
            lineType=cv2.LINE_AA,
        )
        print(x, y)

    win_name = "img2"
    cv2.imshow(win_name, img)
    cv2.waitKey()
    cv2.destroyWindow(win_name)


if __name__ == "__main__":
    main()
