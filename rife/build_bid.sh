#!/usr/bin/env sh


set -euo pipefail

export HSA_OVERRIDE_GFX_VERSION=9.0.0


input_dir=$1

#sh ../inbetweening/crop.sh $out_dir
##exit 0
for j in {0..9}; do
    suf=_00$j
    if [ ! -d $input_dir/int$suf ]; then
        break
    fi

    echo $input_dir/int$suf

    ffmpeg \
        -hide_banner \
        -r 25 \
        -pattern_type glob \
        -i $input_dir/int$suf/fwd/'img_deformed_dirichlet_N_N_0*.png' \
        -c:v libx264 \
        -pix_fmt yuv420p \
        -y \
        "${input_dir}"/out_rife_"${suf}"_f.mp4

    ffmpeg \
       -r 25 \
       -pattern_type glob \
       -i $input_dir/int$suf/bwd/'img_deformed_dirichlet_N_N_0*.png' \
       -c:v libx264 \
       -pix_fmt yuv420p \
       -y \
       "${input_dir}"/out_rife_"${suf}"_b.mp4

done

out_dir=$(basename $input_dir)
echo $out_dir

for j in {0..9}; do
    suf=_00$j
    out_path=$out_dir/output$suf
    if [ ! -d $out_path ]; then
        break
    fi
    echo $out_path

    cat $out_path/anim.txt | awk -v var="$out_path" '{print "file " var"/"$0}' > _an

    r=$(wc -l _an  | cut -d' ' -f 1)
    r=25
    ffmpeg \
        -hide_banner \
        -r $r \
        -i $out_path/tjun_fwd/t_jun_skeleton_img_deformed_dirichlet_N_N_000.png \
        -r $r \
        -pattern_type glob \
        -i $out_path/skel_fwd/'skeleton_img_deformed_dirichlet_N_N_0*.png' \
        -r $r \
        -i $out_path/tjun_bwd/t_jun_skeleton_img_deformed_dirichlet_N_N_000.png \
        -r $r \
        -f concat \
        -i _an \
        -filter_complex "\
            [0:v][1:v][2:v][3:v]xstack=inputs=4:fill=white:layout=0_0|0_h0|w0_0|w0_h0[v0]\
            ;[v0]scale=-1:1080\
            ,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=white[v1]\
            ;[v1]drawtext=fontfile=/home/kbrodt/.local/share/fonts/lin/LinBiolinum_R.ttf\
            :fontcolor=black\
            :fontsize=36\
            :x=1*(w-text_w)/4\
            :y=(h-text_h)/2\
            :text='input skeletal motion'\
            ,drawtext=fontfile=/home/kbrodt/.local/share/fonts/lin/LinBiolinum_R.ttf\
            :fontcolor=black\
            :fontsize=36\
            :x=3*(w-text_w)/4\
            :y=(h-text_h)/2\
            :text='our inbetweening result'\
            ,drawtext=fontfile=/home/kbrodt/.local/share/fonts/lin/LinBiolinum_R.ttf\
            :fontcolor=black\
            :fontsize=36\
            :x=1*(w-text_w)/2\
            :y=(h-text_h)/48\
            :text='input keyframes'\
            [v]\
        "\
        -map "[v]" \
        -c:v libx264 \
        -pix_fmt yuv420p \
        -y \
        $out_path/out_rife_ens_25.mp4 \

    #r=$(($r/2))
    #ffmpeg \
    #    -hide_banner \
    #    -r $r \
    #    -i $out_path/tjun_fwd/t_jun_skeleton_img_deformed_dirichlet_N_N_000.png \
    #    -r $r \
    #    -pattern_type glob \
    #    -i $out_path/skel_fwd/'skeleton_img_deformed_dirichlet_N_N_0*.png' \
    #    -r $r \
    #    -i $out_path/tjun_bwd/t_jun_skeleton_img_deformed_dirichlet_N_N_000.png \
    #    -r $r \
    #    -f concat \
    #    -i _an \
    #    -filter_complex "\
    #        [0:v][1:v][2:v][3:v]xstack=inputs=4:fill=white:layout=0_0|0_h0|w0_0|w0_h0[v0]\
    #        ;[v0]scale=-1:1080\
    #        ,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=white[v1]\
    #        ;[v1]drawtext=fontfile=/home/kbrodt/.local/share/fonts/lin/LinBiolinum_R.ttf\
    #        :fontcolor=black\
    #        :fontsize=36\
    #        :x=1*(w-text_w)/4\
    #        :y=(h-text_h)/2\
    #        :text='input skeletal motion'\
    #        ,drawtext=fontfile=/home/kbrodt/.local/share/fonts/lin/LinBiolinum_R.ttf\
    #        :fontcolor=black\
    #        :fontsize=36\
    #        :x=3*(w-text_w)/4\
    #        :y=(h-text_h)/2\
    #        :text='our inbetweening result'\
    #        ,drawtext=fontfile=/home/kbrodt/.local/share/fonts/lin/LinBiolinum_R.ttf\
    #        :fontcolor=black\
    #        :fontsize=36\
    #        :x=1*(w-text_w)/2\
    #        :y=(h-text_h)/48\
    #        :text='input keyframes'\
    #        [v]\
    #    "\
    #    -map "[v]" \
    #    -c:v libx264 \
    #    -pix_fmt yuv420p \
    #    -y \
    #    $out_path/out_rife_ens_50.mp4 \

    ffmpeg -i $out_path/out_rife_ens_25.mp4 -filter:v "setpts=2*PTS" -y $out_path/out_rife_ens_50.mp4
    #ffmpeg -i $out_path/out_rife_ens_25.mp4 -filter:v "setpts=0.5*PTS" -y $out_path/out_rife_ens_50.mp4
    #ffmpeg -i $out_path/out_rife_ens_25.mp4 -filter:v "setpts=1.5*PTS" -y $out_path/out_rife_ens_50.mp4
    #ffmpeg -i $out_path/out_rife_ens_25.mp4 -filter:v "setpts=0.9*PTS" -y _an.mp4
    #mv _an.mp4 $out_path/out_rife_ens_25.mp4
    #
    ffmpeg \
        -i $out_path/out_rife_ens_25.mp4 \
        -filter_complex "\
            [0:v]drawtext=fontfile=/home/kbrodt/.local/share/fonts/lin/LinBiolinum_R.ttf\
            :fontcolor=black\
            :fontsize=36\
            :x=47*(w-text_w)/48\
            :y=47*(h-text_h)/48\
            :text='1x'\
            [v]\
        "\
        -map "[v]" \
        -c:v libx264 \
        -pix_fmt yuv420p \
        -y \
        /tmp/a.mp4

    mv /tmp/a.mp4 $out_path/out_rife_ens_25.mp4

    ffmpeg \
        -i $out_path/out_rife_ens_50.mp4 \
        -filter_complex "\
            [0:v]drawtext=fontfile=/home/kbrodt/.local/share/fonts/lin/LinBiolinum_R.ttf\
            :fontcolor=black\
            :fontsize=36\
            :x=47*(w-text_w)/48\
            :y=47*(h-text_h)/48\
            :text='0.5x'\
            [v]\
        "\
        -map "[v]" \
        -c:v libx264 \
        -pix_fmt yuv420p \
        -y \
        /tmp/a.mp4

    mv /tmp/a.mp4 $out_path/out_rife_ens_50.mp4

done
