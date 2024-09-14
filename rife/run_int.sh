#!/usr/bin/env sh


set -euo pipefail

export HSA_OVERRIDE_GFX_VERSION=9.0.0


input_dir=$1
use_diag=${2-}
[ -n "${use_diag}" ] && use_diag="--diag"
#| xargs -n 2 python ./inference_img.py -n 13 --img
            #<(python -c 'import numpy as np; [print(int(i)) for i in np.linspace(0, 16, 20).round()]') | \

#for suf in 3; do
#    paste \
#            <(ls -1 $input_dir/fwd/img_deformed_0*.png) \
#            <(ls -1 $input_dir/bwd/img_deformed_0*.png | tac) \
#            <(python -c 'import numpy as np; [print(i) for i in np.linspace(0, 1, 20)]') | \
#    while read i; do
#        i1=$(cut -d ' ' -f 1 <(echo $i))
#        i2=$(cut -d ' ' -f 2 <(echo $i))
#        n=$(cut -d ' ' -f 3 <(echo $i))
#        echo $i1 $i2 $n
#        # python ./inference_img.py -n $n --img $i1 $i2 --out-dir output_$suf
#        python ./inference_img.py --img-start $i1 --img-end $i2 --out-dir output_$suf --ratio $n
#    done
#done

for j in {0..9}; do
    suf=_00$j
    echo $input_dir/int$suf
    # python ./inference_img.py -n $n --img $i1 $i2 --out-dir output_$suf
    if [ ! -d $input_dir/int$suf ]; then
        break
    fi

    img_start=$(ls -1 $input_dir/int$suf/fwd/img_deformed_dirichlet_N_N_0*.png)
    img_end=$(ls -1 $input_dir/int$suf/bwd/img_deformed_dirichlet_N_N_0*.png | tac)

    #ffmpeg \
    #   -r 25 \
    #   -pattern_type glob \
    #   -i $input_dir/int$suf/fwd/'img_deformed_dirichlet_N_N_0*.png' \
    #   -vb 20M \
    #   -vcodec mpeg4 \
    #   -y \
    #   "${input_dir}"/out_rife_"${suf}"_f.mp4

    #ffmpeg \
    #   -r 25 \
    #   -pattern_type glob \
    #   -i $input_dir/int$suf/bwd/'img_deformed_dirichlet_N_N_0*.png' \
    #   -vb 20M \
    #   -vcodec mpeg4 \
    #   -y \
    #   "${input_dir}"/out_rife_"${suf}"_b.mp4

    n=$(ls -1 $input_dir/int$suf/fwd/img_deformed_dirichlet_N_N_0*.png | wc -l)
    out_dir=$(basename $input_dir)/output$suf
    python ./inference_img.py \
        --img-start $img_start \
        --img-end $img_end \
        --out-dir "${out_dir}" \
        --ratio $(python -c "import numpy as np; [print(i) for i in np.linspace(0, 1, $n)]") \
        $(echo "${use_diag}")

    mkdir -p "${out_dir}"/skel_fwd
    cp $input_dir/int$suf/fwd/skeleton_img_deformed_dirichlet_N_N_0*.png "${out_dir}"/skel_fwd
    mkdir -p "${out_dir}"/skel_bwd
    cp $input_dir/int$suf/bwd/skeleton_img_deformed_dirichlet_N_N_0*.png "${out_dir}"/skel_bwd

    mkdir -p "${out_dir}"/tjun_fwd
    cp $input_dir/int$suf/fwd/t_jun_skeleton_img_deformed_dirichlet_N_N_0*.png "${out_dir}"/tjun_fwd
    mkdir -p "${out_dir}"/tjun_bwd
    cp $input_dir/int$suf/bwd/t_jun_skeleton_img_deformed_dirichlet_N_N_0*.png "${out_dir}"/tjun_bwd

done

#ffmpeg \
#   -r 25 \
#   -pattern_type glob \
#   -i $(basename $input_dir)'/output*/*.png' \
#   -vb 20M \
#   -vcodec mpeg4 \
#   -y \
#   "${input_dir}"/out_rife.mp4

exit 0
out_dir=$(basename $input_dir)
echo $out_dir
#sh ../inbetweening/crop.sh $out_dir
##exit 0

for j in {0..9}; do
    suf=_00$j
    out_path=$out_dir/output$suf
    if [ ! -d $out_path ]; then
        break
    fi
    echo $out_path

    ffmpeg \
        -r 25 \
        -i $out_path/tjun_fwd/t_jun_skeleton_img_deformed_dirichlet_N_N_000.png \
        -r 25 \
        -pattern_type glob \
        -i $out_path/skel_fwd/'skeleton_img_deformed_dirichlet_N_N_0*.png' \
        -r 25 \
        -pattern_type glob \
        -i $out_path/'img_deformed_dirichlet_N_N_0*.png' \
        -r 25 \
        -i $out_path/tjun_bwd/t_jun_skeleton_img_deformed_dirichlet_N_N_000.png \
        -filter_complex "\
            [0:v][1:v][2:v][3:v]hstack=inputs=4[v0]\
            ;[v0]scale=1920:-1\
            ,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=white[v1]\
            ;[v1]drawtext=fontfile=/home/kbrodt/.local/share/fonts/lin/LinBiolinum_R.ttf\
            :fontcolor=black\
            :fontsize=48\
            :x=5*(w-text_w)/16\
            :y=(h-text_h)/4\
            :text='input skeletal motion'\
            ,drawtext=fontfile=/home/kbrodt/.local/share/fonts/lin/LinBiolinum_R.ttf\
            :fontcolor=black\
            :fontsize=48\
            :x=11*(w-text_w)/16\
            :y=(h-text_h)/4\
            :text='our inbetweening result'\
            ,drawtext=fontfile=/home/kbrodt/.local/share/fonts/lin/LinBiolinum_R.ttf\
            :fontcolor=black\
            :fontsize=48\
            :x=1*(w-text_w)/16\
            :y=(h-text_h)/4\
            :text='input keyframe'\
            ,drawtext=fontfile=/home/kbrodt/.local/share/fonts/lin/LinBiolinum_R.ttf\
            :fontcolor=black\
            :fontsize=48\
            :x=15*(w-text_w)/16\
            :y=(h-text_h)/4\
            :text='input keyframe'[v]\
        "\
        -map "[v]" \
        -vcodec mpeg4 \
        -y \
        -vb 20M \
        $out_path/out_rife_ens.mp4 \

done

#ffmpeg \
#   -r 25 \
#   -pattern_type glob \
#   -i $(basename $input_dir)'/output*/*.png' \
#   -vb 20M \
#   -vcodec mpeg4 \
#   -y \
#   "${input_dir}"/out_rife.mp4

#ffmpeg -r 25 -pattern_type glob -i 'output_000/*.png' -vb 20M -vcodec mpeg4 -y out_rife.mp4
