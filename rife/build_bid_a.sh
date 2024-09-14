#!/usr/bin/env sh


set -euo pipefail

input_dir=$1


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
    rife_path="${out_dir}"/rife$suf
    eisai_path="${out_dir}"/eisai$suf
    animeint_path="${out_dir}"/animeint$suf

    w=$(cat _an | cut -d' ' -f 2 | head -n1 | xargs -I{} magick {} -format "%w\n" info:)
    h=$(cat _an | cut -d' ' -f 2 | head -n1 | xargs -I{} magick {} -format "%h\n" info:)
    magick -size ${w}x${h} xc:rgb\(255,255,255\) /tmp/white.png

    r=$(wc -l _an  | cut -d' ' -f 1)
    r=25
    ffmpeg \
        -hide_banner \
        -r $r \
        -i /tmp/white.png \
        -r $r \
        -pattern_type glob \
        -i $eisai_path/'*.png' \
        -r $r \
        -i $out_path/tjun_fwd/t_jun_skeleton_img_deformed_dirichlet_N_N_000.png \
        -r $r \
        -pattern_type glob \
        -i $animeint_path/'*.png' \
        -r $r \
        -i $out_path/tjun_bwd/t_jun_skeleton_img_deformed_dirichlet_N_N_000.png \
        -r $r \
        -pattern_type glob \
        -i $rife_path/'img_deformed_dirichlet_N_N_000_*_000.png' \
        -r $r \
        -i /tmp/white.png \
        -r $r \
        -f concat \
        -i _an \
        -filter_complex "\
            [0:v][1:v][2:v][3:v][4:v][5:v][6:v][7:v]xstack=inputs=8:fill=white:layout=0_0|0_h0|w0_0|w0_h0|w0+w1_0|w0+w1_h0|w0+w1+w2_0|w0+w1+w2_h0[v0]\
            ;[v0]scale=1920:-1\
            ,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=white[v1]\
            ;[v1]drawtext=fontfile=/home/kbrodt/.local/share/fonts/lin/LinBiolinum_R.ttf\
            :fontcolor=black\
            :fontsize=36\
            :x=4*(w-text_w)/32\
            :y=(h-text_h)/2\
            :text='EISAI'\
            ,drawtext=fontfile=/home/kbrodt/.local/share/fonts/lin/LinBiolinum_R.ttf\
            :fontcolor=black\
            :fontsize=36\
            :x=12*(w-text_w)/32\
            :y=(h-text_h)/2\
            :text='AnimeInterp'\
            ,drawtext=fontfile=/home/kbrodt/.local/share/fonts/lin/LinBiolinum_R.ttf\
            :fontcolor=black\
            :fontsize=36\
            :x=20*(w-text_w)/32\
            :y=(h-text_h)/2\
            :text='RIFE'\
            ,drawtext=fontfile=/home/kbrodt/.local/share/fonts/lin/LinBiolinum_R.ttf\
            :fontcolor=black\
            :fontsize=36\
            :x=63*(w-text_w)/64\
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
        $animeint_path/out_animeint_ens_25.mp4 \

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

    ffmpeg -i $animeint_path/out_animeint_ens_25.mp4 -filter:v "setpts=2*PTS" -y $animeint_path/out_animeint_ens_50.mp4
    #ffmpeg -i $out_path/out_rife_ens_25.mp4 -filter:v "setpts=0.5*PTS" -y $out_path/out_rife_ens_50.mp4
    #ffmpeg -i $out_path/out_rife_ens_25.mp4 -filter:v "setpts=1.5*PTS" -y $out_path/out_rife_ens_50.mp4
    #ffmpeg -i $out_path/out_rife_ens_25.mp4 -filter:v "setpts=0.9*PTS" -y _an.mp4
    #mv _an.mp4 $out_path/out_rife_ens_25.mp4
    #
    ffmpeg \
        -i $animeint_path/out_animeint_ens_25.mp4 \
        -filter_complex "\
            [0:v]drawtext=fontfile=/home/kbrodt/.local/share/fonts/lin/LinBiolinum_RB.ttf\
            :fontcolor=black\
            :fontsize=36\
            :x=46*(w-text_w)/48\
            :y=44*(h-text_h)/48\
            :text='1x'\
            [v]\
        "\
        -map "[v]" \
        -c:v libx264 \
        -pix_fmt yuv420p \
        -y \
        /tmp/a.mp4

    mv /tmp/a.mp4 $animeint_path/out_animeint_ens_25.mp4

    ffmpeg \
        -i $animeint_path/out_animeint_ens_50.mp4 \
        -filter_complex "\
            [0:v]drawtext=fontfile=/home/kbrodt/.local/share/fonts/lin/LinBiolinum_RB.ttf\
            :fontcolor=black\
            :fontsize=36\
            :x=46*(w-text_w)/48\
            :y=44*(h-text_h)/48\
            :text='0.5x'\
            [v]\
        "\
        -map "[v]" \
        -c:v libx264 \
        -pix_fmt yuv420p \
        -y \
        /tmp/a.mp4

    mv /tmp/a.mp4 $animeint_path/out_animeint_ens_50.mp4

done
