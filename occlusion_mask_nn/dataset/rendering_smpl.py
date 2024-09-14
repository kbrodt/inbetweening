# https://habr.com/ru/companies/friflex/articles/668978/
# https://habr.com/ru/companies/friflex/articles/673322/
# https://habr.com/ru/companies/friflex/articles/681646/

import math
from pathlib import Path

import bpy
import bpy_extras


def is_part_of_skeleton(name, skeleton_bones, not_skeleton_bones):
    for b in skeleton_bones:
        if b not in name:
            continue

        for nb in not_skeleton_bones:
            if nb in name:
                break
        else:
            return True

    return False


def get_keyframes(obj_list, keyframes=1):
    for obj in obj_list:
        anim = obj.animation_data
        if anim is None or anim.action is None:
            continue

        return len(anim.action.fcurves)

    return keyframes


def create_sketchy_material(name="Material"):
    mat = bpy.data.materials.new(name=name)

    mat.use_nodes = True

    nodes = mat.node_tree.nodes
    for node in nodes:
        nodes.remove(node)

    node_pbsdf = nodes.new(type="ShaderNodeHoldout")
    node_pbsdf.location = 0, 0

    node_output = nodes.new(type="ShaderNodeOutputMaterial")
    node_output.location = 0, 0
    node_output.target = "CYCLES"

    mat.node_tree.links.new(node_pbsdf.outputs["Holdout"], node_output.inputs["Surface"])

    mat.alpha_threshold = 1
    mat.diffuse_color = (0, 0, 0, 0)
    mat.blend_method = "CLIP"
    mat.shadow_method = "CLIP"
    mat.use_screen_refraction = True

    mat.roughness = 0

    mat.use_backface_culling = True
    mat.use_sss_translucency = True
    mat.show_transparent_back = True

    return mat


# https://www.youtube.com/watch?v=VSR9qdJ_dRo
def create_mask_material(name="Material"):
    mat = bpy.data.materials.new(name=name)

    mat.use_nodes = True

    nodes = mat.node_tree.nodes
    for node in nodes:
        nodes.remove(node)

    node_pbsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    node_pbsdf.location = 0, 0
    node_pbsdf.inputs["Base Color"].default_value = (0.0, 0.0, 1.0, 0.0)
    node_pbsdf.inputs["Subsurface IOR"].default_value = 0
    node_pbsdf.inputs["Specular"].default_value = 0
    node_pbsdf.inputs["Roughness"].default_value = 0
    node_pbsdf.inputs["Sheen Tint"].default_value = 0
    node_pbsdf.inputs["Clearcoat Roughness"].default_value = 0
    node_pbsdf.inputs["IOR"].default_value = 0
    node_pbsdf.inputs["Emission Strength"].default_value = 0
    node_pbsdf.inputs["Alpha"].default_value = 0.5

    node_output = nodes.new(type="ShaderNodeOutputMaterial")
    node_output.location = 0, 0
    node_output.target = "EEVEE"

    mat.node_tree.links.new(node_pbsdf.outputs["BSDF"], node_output.inputs["Surface"])

    mat.blend_method = "BLEND"

    return mat


def main():
    scene = bpy.context.scene
    for ob in scene.objects:
        bpy.data.objects.remove(ob, do_unlink=True)

    render = scene.render
    render.engine = "CYCLES"
    #render.engine = "BLENDER_EEVEE"

    # Set the device_type
    preferences = bpy.context.preferences
    preferences.addons["cycles"].preferences.compute_device_type = "CUDA"  # or "OPENCL"

    # get_devices() to let Blender detects GPU device
    # https://blender.stackexchange.com/questions/255371/why-does-getting-cycles-devices-fail-with-blender-3-0
    preferences.addons["cycles"].preferences.refresh_devices()
    print(preferences.addons["cycles"].preferences.compute_device_type)
    for d in preferences.addons["cycles"].preferences.devices:
        d["use"] = 1  # Using all devices, include GPU and CPU
        print(d["name"], d["use"])

    # Set the device and feature set
    scene.cycles.device = "GPU"
    scene.cycles.samples = 256

    render.resolution_x = 512
    render.resolution_y = 512
    render.resolution_percentage = 100

    render.use_freestyle = True
    #render.use_freestyle = False
    render.line_thickness = 1
    render.film_transparent = True
    render.image_settings.color_mode = "RGBA"

    freestyle = bpy.context.scene.view_layers["ViewLayer"].freestyle_settings
    linesets = freestyle.linesets.new("VisibleLineset")
    linesets.select_by_visibility = True
    linesets.select_by_edge_types = True
    linesets.select_by_image_border = True
    linesets.select_silhouette = True
    linesets.select_border = True
    linesets.select_crease = True

    skeleton_bones = [
        "pelvis",
        "spine",
        "neck",

        "head",

        "shoulder",
        "elbow",
        "wrist",

        "hip",
        "knee",
        "ankle",
        "foot",
    ]

    not_skeleton_bones = []

    n_camera_views = 8
    cameras = []
    for i in range(n_camera_views):
        #if i not in [
        #    n_camera_views - 4,
        #    n_camera_views - 3,
        #    n_camera_views - 2,
        #]:
        #    continue

        bpy.ops.object.camera_add(
            align="VIEW",
            rotation=[math.pi / 2, 0, 2 * math.pi * i / n_camera_views],
        )
        camera = bpy.context.object
        cameras.append(camera)

    frame_step = 30

    animation_txt = "./cmu.txt"
    with open(animation_txt) as fin:
        animation_list = []
        for line in fin:
            animation_list.append(line.strip())

    #animation_dir = Path("/mnt/research/assets/datasets/amass/CMU/")
    output_root = Path("./dataset")
    output_dir = output_root / "smplx"
    output_dir.mkdir(parents=True, exist_ok=True)

    mat_sketchy = create_sketchy_material()
    mat_mask = create_mask_material()

    global_cnt = 0
    #for animation_i, animation_file in enumerate(animation_dir.rglob("*.npz")):
    for animation_i, animation_file in enumerate(animation_list):
        bpy.ops.object.smplx_add_animation(filepath=str(animation_file))

        for ob in scene.objects:
            if ob.type in ["ARMATURE", "MESH"]:
                ob.select_set(True)
                if ob.type == "ARMATURE":
                    ob.hide_render = True

        selection = bpy.context.selected_objects
        key = get_keyframes(selection)
        scene.frame_start = 1
        scene.frame_end = key

        for frame in range(scene.frame_start, scene.frame_end, frame_step):
            bpy.context.scene.frame_set(frame)
            for camera_num, camera in enumerate(cameras):
                scene.camera = camera
                bpy.ops.view3d.camera_to_view_selected()
                bpy.context.scene.camera.data.sensor_width = 39

                out_name = (
                    f"{global_cnt:0>6}"
                    #f"_{animation_file.stem}"
                    f"_{animation_i:0>3}"
                    f"_{camera_num:0>3}"
                    f"_{frame:0>3}"
                )

                # extract skeleton joints
                with open(output_dir / f"{out_name}_skeleton.txt", "w") as fout:
                    for arm in scene.objects:
                        if "poses" not in arm.name:
                            continue

                        for i, b in enumerate(arm.pose.bones):
                            if not is_part_of_skeleton(b.name, skeleton_bones, not_skeleton_bones):
                                continue

                            global_location = arm.matrix_world @ b.head
                            coords_2d = bpy_extras.object_utils.world_to_camera_view(
                                scene,
                                camera,
                                global_location,
                            )

                            coords_pixel = (coords_2d[0], 1 - coords_2d[1])
                            fout.write(f"{b.name} {coords_pixel[0]} {coords_pixel[1]}\n")

                # render sketchy character
                for ob in scene.objects:
                    if ob.type == "MESH":
                        if ob.data.materials:
                            ob.data.materials[0] = mat_sketchy
                            ob.active_material = mat_sketchy
                        else:
                            ob.data.materials.append(mat_sketchy)

                render.engine = "CYCLES"
                render.use_freestyle = True
                render.filepath = str(output_dir / f"{out_name}_img")
                bpy.ops.render.render(write_still=True)

                # render mask character
                for ob in scene.objects:
                    if ob.type == "MESH":
                        if ob.data.materials:
                            ob.data.materials[0] = mat_mask
                            ob.active_material = mat_mask
                        else:
                            ob.data.materials.append(mat_mask)

                render.engine = "BLENDER_EEVEE"
                render.use_freestyle = False
                render.filepath = str(output_dir / f"{out_name}_mask")
                bpy.ops.render.render(write_still=True)

                global_cnt += 1

        for ob in scene.objects:
            if ob.type == "CAMERA":
               continue

            bpy.data.objects.remove(ob, do_unlink=True)

    print("Done!")
    print(f"{global_cnt=}")


if __name__ == "__main__":
    main()
