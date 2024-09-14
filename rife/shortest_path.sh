#/usr/bin/env sh


set -euo pipefail


for d in $1; do
    for s in $d/output_0*; do
        i=$(echo $(basename $s) | cut -d_ -f 2)
        echo $d $s

        #python shortest_path.py $d $i 0

        #cd $s
        ##cat ../../diag.txt | xargs cat | ffmpeg  -f image2pipe -i -  -r 25  -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p  -y diag.mp4
        ##cat ../../anim.txt | xargs cat | ffmpeg  -f image2pipe -i -  -r 25  -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p  -y anim.mp4
        #cat ../../diag.txt | xargs cat | ffmpeg  -f image2pipe -i -  -r 25  -c:v libx264 -pix_fmt yuv420p  -y diag.mp4
        #cat ../../anim.txt | xargs cat | ffmpeg  -f image2pipe -i -  -r 25  -c:v libx264 -pix_fmt yuv420p  -y anim.mp4
        #ffmpeg -i diag.mp4 -i anim.mp4 -filter_complex "hstack=inputs=2" -y h.mp4
        #cd -

        #echo $d $s
        #python shortest_path.py $d $i 1
        python shortest_path.py $d $i 1

        cd $s
        #cat ../../diag.txt | xargs cat | ffmpeg  -f image2pipe -i -  -r 25  -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p  -y diag.mp4
        #cat ../../anim.txt | xargs cat | ffmpeg  -f image2pipe -i -  -r 25  -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p  -y anim.mp4
        cat diag.txt | xargs cat | ffmpeg  -f image2pipe -i -  -r 25  -c:v libx264 -pix_fmt yuv420p  -y diag.mp4
        cat anim.txt | xargs cat | ffmpeg  -f image2pipe -i -  -r 25  -c:v libx264 -pix_fmt yuv420p  -y anim.mp4
        ffmpeg -i diag.mp4 -i anim.mp4 -filter_complex "hstack=inputs=2" -y h.mp4

        #ffmpeg -i anim.mp4 -i anim_div.mp4 -filter_complex "hstack=inputs=2" -y h_div.mp4

        cd -

    done
done
