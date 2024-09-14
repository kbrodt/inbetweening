import json

import pickle
import numpy as np
#import collada
#import collada.source
#import collada.controller
from scipy.spatial.transform import (
    Rotation,
)

def save_pkl(obj, path):
    with open(path, "wb") as fout:
        pickle.dump(obj, fout)


def load_pkl(path):
    with open(path, "rb") as fin:
        obj = pickle.load(fin)

    return obj


def save_json(obj, path, indent=None):
    with open(path, "w") as fout:
        json.dump(obj, fout, indent=indent)


def load_json(path):
    with open(path, "r") as fout:
        return json.load(fout)


# https://github.com/PeizhuoLi/neural-blend-shapes/blob/main/dataset/bvh_writer.py
def write_bvh(
    parent,
    offset,
    rotation,
    position,
    names,
    frametime,
    order,
    path,
    endsite=None,
):
    file = open(path, "w")
    frame = rotation.shape[0]
    joint_num = rotation.shape[1]
    order = order.upper()

    file_string = 'HIERARCHY\n'

    seq = []

    def write_static(idx, prefix):
        nonlocal parent, offset, rotation, names, order, endsite, file_string, seq
        seq.append(idx)
        if idx == 0:
            name_label = 'ROOT ' + names[idx]
            channel_label = 'CHANNELS 6 Xposition Yposition Zposition {}rotation {}rotation {}rotation'.format(*order)
        else:
            name_label = 'JOINT ' + names[idx]
            channel_label = 'CHANNELS 3 {}rotation {}rotation {}rotation'.format(*order)
        offset_label = 'OFFSET %.6f %.6f %.6f' % (offset[idx][0], offset[idx][1], offset[idx][2])

        file_string += prefix + name_label + '\n'
        file_string += prefix + '{\n'
        file_string += prefix + '\t' + offset_label + '\n'
        file_string += prefix + '\t' + channel_label + '\n'

        has_child = False
        for y in range(idx+1, rotation.shape[1]):
            if parent[y] == idx:
                has_child = True
                write_static(y, prefix + '\t')
        if not has_child:
            file_string += prefix + '\t' + 'End Site\n'
            file_string += prefix + '\t' + '{\n'
            file_string += prefix + '\t\t' + 'OFFSET 0 0 0\n'
            file_string += prefix + '\t' + '}\n'

        file_string += prefix + '}\n'

    write_static(0, '')

    file_string += 'MOTION\n' + 'Frames: {}\n'.format(frame) + 'Frame Time: %.8f\n' % frametime
    for i in range(frame):
        file_string += '%.6f %.6f %.6f ' % (position[i][0], position[i][1], position[i][2])
        for j in range(joint_num):
            idx = seq[j]
            file_string += '%.6f %.6f %.6f ' % (rotation[i][idx][0], rotation[i][idx][1], rotation[i][idx][2])
        file_string += '\n'

    file.write(file_string)
    file.close()

    return file_string


class BVH_Joint:
    def __init__(self, name, offset, channel_order, children):
        self.name = name
        self.offset = offset
        self.channel_order = channel_order
        self.children = children

    def joint_count(self):
        """ Returns 1 + the number of Joint children in this joint's kinematic chain (recursive) """
        count = 1
        for c in self.get_children():
            if isinstance(c, BVH_Joint):
                if c.name != "End Site":
                    count += c.joint_count()

        return count

    def get_children(self):
        return self.children


def read_animation_from_bvh(bvh_file):
    def _parse_skeleton(lines):

        # Get the joint name
        if lines[0].strip().startswith('ROOT'):
            _, joint_name = lines.pop(0).strip().split(' ')
        elif lines[0].strip().startswith('JOINT'):
            _, joint_name = lines.pop(0).strip().split(' ')
        elif lines[0].strip().startswith('End Site'):
            joint_name = lines.pop(0).strip()
        else:
            msg = f'Malformed BVH. Line: {lines[0]}'
            assert False, msg

        if lines.pop(0).strip() != '{':
            msg = f'Malformed BVH in line preceding {lines}'
            assert False, msg

        # Get offset
        if not lines[0].strip().startswith('OFFSET'):
            msg = f'Malformed BVH in line preceding {lines}'
            assert False, msg
        _, *xyz = lines.pop(0).strip().split(' ')
        offset = list(map(float, xyz))

        # Get channels
        if lines[0].strip().startswith('CHANNELS'):
            channel_order = lines.pop(0).strip().split(' ')
            _, channel_num, *channel_order = channel_order
        else:
            channel_num, channel_order = 0, []
        if int(channel_num) != len(channel_order):
            msg = f'Malformed BVH in line preceding {lines}'
            assert False, msg

        # Recurse for children
        children = []
        while lines[0].strip() != '}':
            children.append(_parse_skeleton(lines))
        lines.pop(0)  # }

        return BVH_Joint(name=joint_name, offset=offset, channel_order=channel_order, children=children)

    def process_frame_data(skeleton, frames):
        """ Given skeleton and frame data, return root position data and joint quaternion data, separately"""

        def _get_frame_channel_order(joint, channels=[]):
            channels.extend(joint.channel_order)
            for child in [child for child in joint.get_children() if isinstance(child, BVH_Joint)]:
                _get_frame_channel_order(child, channels)

            return channels

        channels = _get_frame_channel_order(skeleton)

        # create a mask so we retain only joint rotations and root position
        mask = np.array(list(map(lambda x: True if 'rotation' in x else False, channels)))
        mask[:3] = True  # hack to make sure we keep root position

        frames = np.array(frames, dtype=np.float32)[:, mask]

        # split root pose data and joint euler angle data
        pos_data, ea_rots = np.split(np.array(frames, dtype=np.float32), [3], axis=1)
        assert ea_rots.shape == (len(frames), 3 * skeleton.joint_count())

        rot_data = np.empty([len(frames), skeleton.joint_count(), 3], dtype=np.float32)
        _pose_ea_to_q(skeleton, ea_rots, rot_data)

        return pos_data, rot_data

    def _pose_ea_to_q(joint, ea_rots, q_rots, p1=0, p2=0):
        axis_chars = "".join([c[0].lower() for c in joint.channel_order if c.endswith('rotation')])  # e.g. 'xyz'

        q_rots[:, p2] = ea_rots[:, p1:p1+len(axis_chars)]
        p1 += len(axis_chars)
        p2 += 1

        for child in joint.get_children():
            if isinstance(child, BVH_Joint) and child.name != "End Site":
                p1, p2 = _pose_ea_to_q(child, ea_rots, q_rots, p1, p2)

        return p1, p2

    with open(bvh_file, "r") as fin:
        line = fin.readline().strip()
        if line != 'HIERARCHY':
            msg = f'Malformed BVH in line preceding {line}'
            assert False, msg

        lines = fin.readlines()

    root_joint = _parse_skeleton(lines)
    print(root_joint.offset)
    print(root_joint.channel_order)

    #root_joint = parse_bvh_skeleton(lines)

    line = lines.pop(0).strip()
    if line != "MOTION":
        assert False, line

    #_, n_frames = fin.readline().split()
    _, n_frames = lines.pop(0).strip().split()
    n_frames = int(n_frames)
    lines.pop(0)

    frames = [
        list(map(float, line.strip().split()))
        #for line in fin
        for line in lines
    ]
    assert n_frames == len(frames)

    #inds = []
    #def dfs(node):
    #    inds.append(node.name)
    #    for c in node.get_children():
    #        dfs(c)

    #dfs(root_joint)
    #print(inds)
    #raise

    #pos_data, animation = process_frame_data(root_joint, frames)
    frames = np.array(frames, dtype=np.float32)

    return frames

    # split root pose data and joint euler angle data
    #pos_data, ea_rots = np.split(np.array(frames, dtype=np.float32), [3], axis=1)

    #return pos_data, ea_rots


# https://pycollada.readthedocs.io/en/latest/creating.html
# https://stackoverflow.com/questions/16955533/texturing-on-3d-blocks-using-pycollada
def write_dae(textures, V, T, z, wh, out_dir):
    mesh = collada.Collada()

    image_c = collada.material.CImage(
        id="material-image",
        path=textures,
    )
    surface = collada.material.Surface(
        id="material-image-surface",
        img=image_c,
    )
    sampler = collada.material.Sampler2D(
        id="material-image-sampler",
        surface=surface,
    )
    mp = collada.material.Map(
        sampler=sampler,
        texcoord="UVSET0",
    )
    effect = collada.material.Effect(
        "material-effect",
        [surface, sampler],
        "lambert",
        emission=(0.0, 0.0, 0.0, 1),
        ambient=(0.0, 0.0, 0.0, 1),
        diffuse=mp,
        transparent=mp,
        transparency=0.0,
        # double_sided=True,
    )
    mat = collada.material.Material("material-ID", "material", effect)

    mesh.effects.append(effect)

    mesh.materials.append(mat)

    mesh.images.append(image_c)

    vert_floats = np.hstack([V, 100 * z]).ravel()
    vert_src = collada.source.FloatSource("mesh-geometry-position", vert_floats, ("X", "Y", "Z"))

    m1uv = np.stack([V[:, 0] / wh, 1 - V[:, 1] / wh]).T.ravel()
    uv_src = collada.source.FloatSource("mesh-geometry-uv", np.array(m1uv), ("S", "T"))

    geom = collada.geometry.Geometry(mesh, "mesh-geometry", "mesh-geometry", [vert_src, uv_src])

    input_list = collada.source.InputList()
    input_list.addInput(0, "VERTEX", "#mesh-geometry-position")
    input_list.addInput(0, "TEXCOORD", "#mesh-geometry-uv")

    indices = T.ravel()
    triset = geom.createTriangleSet(indices, input_list, "material-ref")

    geom.primitives.append(triset)

    mesh.geometries.append(geom)

    #joint_source = "mesh-joints"
    #joint_matrix_source = "mesh-matrix"
    #weight_source = "mesh-weight"
    #weight_joint_source = "mesh-weight-joint"
    #sourcebyid = {
    #    "id": "skin",
    #    joint_source: collada.source.FloatSource(),
    #    joint_matrix_source: collada.source.FloatSource(np.zeros((4, 4))),
    #    weight_source: collada.source.FloatSource(),
    #    weight_joint_source: collada.source.FloatSource(),
    #}
    #bind_shape_matrix = np.zeros((4, 4))
    #bind_shape_matrix = bind_shape_matrix.ravel()
    #controller = collada.controller.Skin(
    #    sourcebyid,
    #    bind_shape_matrix=bind_shape_matrix,
    #    joint_source=joint_source,
    #    joint_matrix_source=joint_matrix_source,
    #    weight_source=weight_source,
    #    weight_joint_source=weight_joint_source,
    #    vcounts,
    #    vertex_weight_index,
    #    offsets,
    #    geometry=geom,
    #)
    #mesh.controllers.append(controller)

    matnode = collada.scene.MaterialNode("material-ref", mat, inputs=[])

    geomnode = collada.scene.GeometryNode(geom, [matnode])

    node = collada.scene.Node("node0", children=[geomnode])
    myscene = collada.scene.Scene("myscene", [node])
    mesh.scenes.append(myscene)
    mesh.scene = myscene
    mesh.write(out_dir / "mesh.dae")
