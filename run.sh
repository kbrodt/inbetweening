#!/usr/bin/env sh


set -euo pipefail


inpaint_method=copy
suffix=_${inpaint_method}_fin2_2d_v_FIN_man_a3
out_dir=test_2_col_0_1"${suffix}"
rm -rf "${out_dir}"
python src/inb.py \
    --inpaint-method "${inpaint_method}" \
    --out-dir "${out_dir}" \
    --touch-pixels 1 \
    --img-paths \
        ./data/imgs_m/2_col/target-0.png \
        ./data/imgs_m/2_col/target-1.png \
    --animation-paths \
        ./animations/test_2_col_5_6_7_0_copy_fin2_2d_v_FIN_man_skeleton1_00.txt \


inpaint_method=copy
suffix=_${inpaint_method}_fin2_2d_v_FIN_man_a3
out_dir=test_2_col_1_2"${suffix}"
rm -rf "${out_dir}"
python src/inb.py \
    --inpaint-method "${inpaint_method}" \
    --out-dir "${out_dir}" \
    --touch-pixels 1 \
    --img-paths \
        ./data/imgs_m/2_col/target-1.png \
        ./data/imgs_m/2_col/target-2.png \
    --animation-paths \
        ./animations/test_2_col_5_6_7_0_copy_fin2_2d_v_FIN_man_skeleton1_01.txt \


inpaint_method=copy
suffix=_${inpaint_method}_fin2_2d_v_FIN_man_a3
out_dir=test_2_col_2_3"${suffix}"
rm -rf "${out_dir}"
python src/inb.py \
    --inpaint-method "${inpaint_method}" \
    --out-dir "${out_dir}" \
    --touch-pixels 1 \
    --img-paths \
        ./data/imgs_m/2_col/target-2.png \
        ./data/imgs_m/2_col/target-3.png \
    --animation-paths \
        ./animations/test_2_col_5_6_7_0_copy_fin2_2d_v_FIN_man_skeleton1_02.txt \


inpaint_method=copy
suffix=_${inpaint_method}_fin2_2d_v_FIN_man_a3
out_dir=test_2_col_3_5"${suffix}"
rm -rf "${out_dir}"
python src/inb.py \
    --inpaint-method "${inpaint_method}" \
    --out-dir "${out_dir}" \
    --touch-pixels 1 \
    --img-paths \
        ./data/imgs_m/2_col2/target-3.png \
        ./data/imgs_m/2_col2/target-5.png \
    --animation-paths \
        ./animations/test_2_col_5_6_7_0_copy_fin2_2d_v_FIN_man_skeleton1_03.txt \


inpaint_method=copy
suffix=_${inpaint_method}_fin2_2d_v_FIN_man_a3
out_dir=test_2_col_5_6"${suffix}"
rm -rf "${out_dir}"
python src/inb.py \
    --inpaint-method "${inpaint_method}" \
    --out-dir "${out_dir}" \
    --touch-pixels 1 \
    --img-paths \
        ./data/imgs_m/2_col/target-5.png \
        ./data/imgs_m/2_col/target-6.png \
    --animation-paths \
        ./animations/test_2_col_5_6_7_0_copy_fin2_2d_v_FIN_man_skeleton1_04.txt \


inpaint_method=copy
suffix=_${inpaint_method}_fin2_2d_v_FIN_man_a3
out_dir=test_2_col_6_7"${suffix}"
rm -rf "${out_dir}"
python src/inb.py \
    --inpaint-method "${inpaint_method}" \
    --out-dir "${out_dir}" \
    --touch-pixels 1 \
    --img-paths \
        ./data/imgs_m/2_col/target-6.png \
        ./data/imgs_m/2_col/target-7.png \
    --animation-paths \
        ./animations/test_2_col_5_6_7_0_copy_fin2_2d_v_FIN_man_skeleton1_05.txt \


inpaint_method=copy
suffix=_${inpaint_method}_fin2_2d_v_FIN_man_a3
out_dir=test_2_col_7_0"${suffix}"
rm -rf "${out_dir}"
python src/inb.py \
    --inpaint-method "${inpaint_method}" \
    --out-dir "${out_dir}" \
    --touch-pixels 1 \
    --img-paths \
        ./data/imgs_m/2_col/target-7.png \
        ./data/imgs_m/2_col/target-0.png \
    --animation-paths \
        ./animations/test_2_col_5_6_7_0_copy_fin2_2d_v_FIN_man_skeleton1_06.txt \

mkdir -p test_2_col3

cp -r test_2_col_0_1_copy_fin2_2d_v_FIN_man_a3/fwd_000 test_2_col3/fwd_000
cp -r test_2_col_0_1_copy_fin2_2d_v_FIN_man_a3/bwd_000 test_2_col3/bwd_000
cp -r test_2_col_0_1_copy_fin2_2d_v_FIN_man_a3/int_000 test_2_col3/int_000

cp -r test_2_col_1_2_copy_fin2_2d_v_FIN_man_a3/fwd_000 test_2_col3/fwd_001
cp -r test_2_col_1_2_copy_fin2_2d_v_FIN_man_a3/bwd_000 test_2_col3/bwd_001
cp -r test_2_col_1_2_copy_fin2_2d_v_FIN_man_a3/int_000 test_2_col3/int_001

cp -r test_2_col_2_3_copy_fin2_2d_v_FIN_man_a3/fwd_000 test_2_col3/fwd_002
cp -r test_2_col_2_3_copy_fin2_2d_v_FIN_man_a3/bwd_000 test_2_col3/bwd_002
cp -r test_2_col_2_3_copy_fin2_2d_v_FIN_man_a3/int_000 test_2_col3/int_002

cp -r test_2_col_3_5_copy_fin2_2d_v_FIN_man_a3/fwd_000 test_2_col3/fwd_003
cp -r test_2_col_3_5_copy_fin2_2d_v_FIN_man_a3/bwd_000 test_2_col3/bwd_003
cp -r test_2_col_3_5_copy_fin2_2d_v_FIN_man_a3/int_000 test_2_col3/int_003

cp -r test_2_col_5_6_copy_fin2_2d_v_FIN_man_a3/fwd_000 test_2_col3/fwd_004
cp -r test_2_col_5_6_copy_fin2_2d_v_FIN_man_a3/bwd_000 test_2_col3/bwd_004
cp -r test_2_col_5_6_copy_fin2_2d_v_FIN_man_a3/int_000 test_2_col3/int_004

cp -r test_2_col_6_7_copy_fin2_2d_v_FIN_man_a3/fwd_000 test_2_col3/fwd_005
cp -r test_2_col_6_7_copy_fin2_2d_v_FIN_man_a3/bwd_000 test_2_col3/bwd_005
cp -r test_2_col_6_7_copy_fin2_2d_v_FIN_man_a3/int_000 test_2_col3/int_005

cp -r test_2_col_7_0_copy_fin2_2d_v_FIN_man_a3/fwd_000 test_2_col3/fwd_006
cp -r test_2_col_7_0_copy_fin2_2d_v_FIN_man_a3/bwd_000 test_2_col3/bwd_006
cp -r test_2_col_7_0_copy_fin2_2d_v_FIN_man_a3/int_000 test_2_col3/int_006

inpaint_method=lama
suffix=_${inpaint_method}_fin2_2d_v_FIN_a2
out_dir=test_19"${suffix}"
rm -rf "${out_dir}"
python src/inb.py \
    --inpaint-method "${inpaint_method}" \
    --out-dir "${out_dir}" \
    --img-paths \
        ./data/imgs/19/image_50388737.png \
        ./data/imgs/19/image_123650291.png \
    --animation-paths \
        ./animations/test_19_lama_fin2_2d_v_FIN_man_double2_skeleton_1.txt \


# git co multiple_layers
#inpaint_method=lama
#suffix=_${inpaint_method}_fin2_2d_v_FIN_man_double2
#out_dir=test_19"${suffix}"
#rm -rf "${out_dir}"
#python src/inb.py \
#    --inpaint-method "${inpaint_method}" \
#    --out-dir "${out_dir}" \
#    --img-paths \
#        ./data/imgs_m/19_m/image_123650291.png \
#        ./data/imgs_m/19_m/image_50349569.png \

mkdir -p test_19

cp -r test_19_lama_fin2_2d_v_FIN_a2/fwd_000 test_19/fwd_000
cp -r test_19_lama_fin2_2d_v_FIN_a2/bwd_000 test_19/bwd_000
cp -r test_19_lama_fin2_2d_v_FIN_a2/int_000 test_19/int_000

#cp -r test_19_lama_fin2_2d_v_FIN_man_double_a2/fwd_000 test_19/fwd_001
#cp -r test_19_lama_fin2_2d_v_FIN_man_double_a2/bwd_000 test_19/bwd_001
#cp -r test_19_lama_fin2_2d_v_FIN_man_double_a2/int_000 test_19/int_001

inpaint_method=lama
suffix=_${inpaint_method}_fin2_2d_v_FIN_a
out_dir=test_23"${suffix}"
rm -rf "${out_dir}"
python src/inb.py \
    --inpaint-method "${inpaint_method}" \
    --out-dir "${out_dir}" \
    --img-paths \
        ./data/imgs/23/image_2.png \
        ./data/imgs/23/image_1.png \
    --animation-paths \
        ./animations/test_23_lama_fin2_2d_v_FIN_skeleton.txt \

inpaint_method=lama
suffix=_"${inpaint_method}"_FIN_man
#dilate=1
out_dir=test_dvor_wman_hr"${suffix}"
rm -rf "${out_dir}"
python src/inb.py \
    --inpaint-method "${inpaint_method}" \
    --out-dir "${out_dir}" \
    --touch-pixels 1 \
    --img-paths \
        ./data/imgs_m/ts_old/2.png \
        ./data/imgs_m/ts_old/4.png \
    --guidance-paths \
        ./data/imgs_m/ts_old/3.png \


inpaint_method=lama
suffix=_"${inpaint_method}"_FIN_man
# dilate 3
out_dir=test_dvor_girl"${suffix}"
rm -rf "${out_dir}"
python src/inb.py \
    --inpaint-method "${inpaint_method}" \
    --out-dir "${out_dir}" \
    --touch-pixels 1 \
    --img-paths \
        ./data/imgs_m/toonsynth/toonsynth-002.jpg \
        ./data/imgs_m/toonsynth/toonsynth-008.jpg \
    --guidance-paths \
        ./data/imgs_m/toonsynth/toonsynth-004.jpg \
        ./data/imgs_m/toonsynth/toonsynth-006.jpg \


inpaint_method=lama
suffix=_"${inpaint_method}"_FIN_man
out_dir=test_dvor_lady"${suffix}"
rm -rf "${out_dir}"
python src/inb.py \
    --inpaint-method "${inpaint_method}" \
    --out-dir "${out_dir}" \
    --touch-pixels 1 \
    --img-paths \
        ./data/imgs_m/dvor/lady2/2/1.png \
        ./data/imgs_m/dvor/lady2/2/7.png \
    --guidance-paths \
        ./data/imgs_m/dvor/lady2/2/2.png \
        ./data/imgs_m/dvor/lady2/2/3.png \
        ./data/imgs_m/dvor/lady2/2/4.png \
        ./data/imgs_m/dvor/lady2/2/5.png \
        ./data/imgs_m/dvor/lady2/2/6.png \


inpaint_method=copy
suffix=_"${inpaint_method}"_FIN_man
out_dir=test_tooncap"${suffix}"
rm -rf "${out_dir}"
python src/inb.py \
    --inpaint-method "${inpaint_method}" \
    --out-dir "${out_dir}" \
    --img-paths \
        ./data/imgs_m/tooncap/1.png \
        ./data/imgs_m/tooncap/2.png \


inpaint_method=copy
suffix=_"${inpaint_method}"_FIN_man
out_dir=test_tooncap2"${suffix}"
rm -rf "${out_dir}"
python src/inb.py \
    --inpaint-method "${inpaint_method}" \
    --out-dir "${out_dir}" \
    --img-paths \
        ./data/imgs_m/tooncap2/1.png \
        ./data/imgs_m/tooncap2/2.png \


inpaint_method=copy
suffix=_"${inpaint_method}"_FIN_man
out_dir=test_aladin_p3_0_6"${suffix}"
rm -rf "${out_dir}"
python src/inb.py \
    --inpaint-method "${inpaint_method}" \
    --out-dir "${out_dir}" \
    --img-paths \
        ./data/imgs_m/aladin/pose_3_new/0.png \
        ./data/imgs_m/aladin/pose_3_new/6.png \
    --guidance-paths \
        ./data/imgs_m/aladin/pose_3_new/1.png \
        ./data/imgs_m/aladin/pose_3_new/2.png \
        ./data/imgs_m/aladin/pose_3_new/3.png \
        ./data/imgs_m/aladin/pose_3_new/4.png \
        ./data/imgs_m/aladin/pose_3_new/5.png \


inpaint_method=copy
suffix=_${inpaint_method}_fin2_2d_v_FIN
out_dir=test_2_1_2"${suffix}"
rm -rf "${out_dir}"
python src/inb.py \
    --inpaint-method "${inpaint_method}" \
    --out-dir "${out_dir}" \
    --touch-pixels 2 \
    --img-paths \
        ./data/imgs_m/2_w/out-1.png \
        ./data/imgs_m/2_w/out-2.png \


inpaint_method=lama
suffix=_${inpaint_method}_fin2_2d_v_FIN
out_dir=test_18_0_1"${suffix}"
rm -rf "${out_dir}"
python src/inb.py \
    --inpaint-method "${inpaint_method}" \
    --out-dir "${out_dir}" \
    --touch-pixels 2 \
    --img-paths \
        ./data/imgs/18/kA8na5ZF.png \
        ./data/imgs/18/MVyi_6ih.png \


inpaint_method=lama
suffix=_${inpaint_method}_fin2_2d_v_FIN_man
# nopad in center!
out_dir=test_20"${suffix}"
rm -rf "${out_dir}"
python src/inb.py \
    --inpaint-method "${inpaint_method}" \
    --out-dir "${out_dir}" \
    --touch-pixels 2 \
    --img-paths \
        ./data/imgs_m/20/santa1.png \
        ./data/imgs_m/20/santa2.png \


inpaint_method=lama
suffix=_${inpaint_method}_fin2_2d_v_FIN
out_dir=test_21_1_2"${suffix}"
rm -rf "${out_dir}"
python src/inb.py \
    --inpaint-method "${inpaint_method}" \
    --out-dir "${out_dir}" \
    --touch-pixels 1 \
    --img-paths \
        ./data/imgs/21/frog_1.png \
        ./data/imgs/21/frog_2.png \

inpaint_method=lama
suffix=_${inpaint_method}_fin2_2d_v_FIN_man2
#assert img.shape[0] == img.shape[1] == wh_img, (img.shape, wh_img)
out_dir=test_21_0_1"${suffix}"
rm -rf "${out_dir}"
python src/inb.py \
    --inpaint-method "${inpaint_method}" \
    --out-dir "${out_dir}" \
    --touch-pixels 1 \
    --img-paths \
        ./data/imgs_m/21/frog_0.png \
        ./data/imgs_m/21/frog_1.png \


mkdir -p test_21

cp -r test_21_1_2_lama_fin2_2d_v_FIN/fwd_000 test_21/fwd_000
cp -r test_21_1_2_lama_fin2_2d_v_FIN/bwd_000 test_21/bwd_000
cp -r test_21_1_2_lama_fin2_2d_v_FIN/int_000 test_21/int_000

cp -r test_21_0_1_lama_fin2_2d_v_FIN_man2/fwd_000 test_21/fwd_001
cp -r test_21_0_1_lama_fin2_2d_v_FIN_man2/bwd_000 test_21/bwd_001
cp -r test_21_0_1_lama_fin2_2d_v_FIN_man2/int_000 test_21/int_001

inpaint_method=lama
suffix=_${inpaint_method}_fin2_2d_v_FIN_man2
out_dir=test_24"${suffix}"
rm -rf "${out_dir}"
python src/inb.py \
    --inpaint-method "${inpaint_method}" \
    --out-dir "${out_dir}" \
    --img-paths \
        ./data/imgs_m/24_nl/im1.png \
        ./data/imgs_m/24_nl/im2.png \


inpaint_method=lama
suffix=_${inpaint_method}_fin2_2d_v_FIN
out_dir=test_28"${suffix}"
rm -rf "${out_dir}"
python src/inb.py \
    --inpaint-method "${inpaint_method}" \
    --out-dir "${out_dir}" \
    --img-paths \
        ./data/imgs/28/bal1.png \
        ./data/imgs/28/bal2.png \


inpaint_method=copy
suffix=_"${inpaint_method}"_FIN
out_dir=test_jostc"${suffix}"
rm -rf "${out_dir}"
python src/inb.py \
    --inpaint-method "${inpaint_method}" \
    --out-dir "${out_dir}" \
    --img-paths \
        ./data/imgs/jostc/target-0.png \
        ./data/imgs/jostc/target-8.png \


inpaint_method=copy
suffix=_"${inpaint_method}"_FIN
# pad 16
out_dir=test_chaki_1_4"${suffix}"
rm -rf "${out_dir}"
python src/inb.py \
    --character-topology-path ./characters_topology/chain_topology.json \
    --out-dir "${out_dir}" \
    --inpaint-method "${inpaint_method}" \
    --img-paths \
        ./data/imgs_m/chaki/1.png \
        ./data/imgs_m/chaki/4.png \
    --guidance-paths \
        ./data/imgs_m/chaki/2.png \
        ./data/imgs_m/chaki/3.png \

