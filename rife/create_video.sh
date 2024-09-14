#!/usr/bin/env sh


input_dir=$1


ffmpeg \
    -r 25 \
    -f lavfi \
    -i color=size=1920x1080:duration=4:rate=25:color=white \
    -filter_complex "\
        [0:v]drawtext=fontfile=/home/kbrodt/.local/share/fonts/lin/LinBiolinum_R.ttf\
        :fontcolor=black\
        :fontsize=64\
        :x=(w-text_w)/2\
        :y=(h-text_h)/2\
        :textfile=paper.txt\
        :enable='between(t,0,2)'\
        ,drawtext=fontfile=/home/kbrodt/.local/share/fonts/lin/LinBiolinum_R.ttf\
        :fontcolor=black\
        :fontsize=48\
        :x=(w-text_w)/2\
        :y=h-th-80
        :text='The video contains no sound.'\
        :enable='between(t,0,2)'\
        ,drawtext=fontfile=/home/kbrodt/.local/share/fonts/lin/LinBiolinum_R.ttf\
        :fontcolor=black\
        :fontsize=64\
        :x=(w-text_w)/2\
        :y=(h-text_h)/2\
        :text='Our results'\
        :enable='between(t,2,4)'
    "\
    -c:v libx264 \
    -pix_fmt yuv420p \
    -y \
    our_results2.mp4 \


ffmpeg \
    -r 25 \
    -f lavfi \
    -i color=size=1920x1080:duration=2:rate=25:color=white \
    -filter_complex "\
        [0:v]drawtext=fontfile=/home/kbrodt/.local/share/fonts/lin/LinBiolinum_R.ttf\
        :fontcolor=black\
        :fontsize=64\
        :x=(w-text_w)/2\
        :y=(h-text_h)/2\
        :text='Effect of Blending Optimization'\
    "\
    -c:v libx264 \
    -pix_fmt yuv420p \
    -y \
    cmp_diag.mp4 \


ffmpeg \
    -r 25 \
    -f lavfi \
    -i color=size=1920x1080:duration=2:rate=25:color=white \
    -filter_complex "\
        [0:v]drawtext=fontfile=/home/kbrodt/.local/share/fonts/lin/LinBiolinum_R.ttf\
        :fontcolor=black\
        :fontsize=64\
        :x=(w-text_w)/2\
        :y=(h-text_h)/2\
        :textfile=comp.txt'\
    "\
    -c:v libx264 \
    -pix_fmt yuv420p \
    -y \
    cmp_rife.mp4 \


ffmpeg \
    -r 25 \
    -f lavfi \
    -i color=size=1920x1080:duration=2:rate=25:color=white \
    -filter_complex "\
        [0:v]drawtext=fontfile=/home/kbrodt/.local/share/fonts/lin/LinBiolinum_R.ttf\
        :fontcolor=black\
        :fontsize=64\
        :x=(w-text_w)/2\
        :y=(h-text_h)/2\
        :text='Comparisons with CharacterGAN [Hinz et al. 2022]'\
    "\
    -c:v libx264 \
    -pix_fmt yuv420p \
    -y \
    cmp_chargan.mp4 \

for i in $(cat anim_list.txt); do sh build_bid.sh ../inbetweening/$i; done
sh build_bid.sh ../inbetweening/test_2_col3
#for i in $(cat anim_list.txt); do sh build_bid_rife.sh ../inbetweening/$i; done
#sh build_bid_rife.sh ../inbetweening/test_2_col3
#for i in $(cat anim_list.txt); do sh build_bid_eisai.sh ../inbetweening/$i; done
#sh build_bid_eisai.sh ../inbetweening/test_2_col3
for i in $(cat anim_list.txt); do sh build_bid_a.sh ../inbetweening/$i; done
sh build_bid_a.sh ../inbetweening/test_2_col3

sh build_bid_chargan.sh ../inbetweening/test_dvor_lady_lama_FIN_man
sh build_bid_chargan.sh ../inbetweening/test_aladin_p3_0_6_copy_FIN_man
sh build_bid_chargan.sh ../inbetweening/test_dvor_wman_hr_lama_FIN_man

ffmpeg -f concat -safe 0 -i list2r.txt -c copy -y h.mp4
#ffmpeg -f concat -safe 0 -i list2r_rife.txt -c copy -y h_rife.mp4
#ffmpeg -f concat -safe 0 -i list2r_eisai.txt -c copy -y h_eisai.mp4
ffmpeg -f concat -safe 0 -i list2r_animeint.txt -c copy -y h_animeint.mp4
ffmpeg -f concat -safe 0 -i list2r_chargan.txt -c copy -y h_chargan.mp4
ffmpeg -f concat -safe 0 -i list_fin2.txt \
    -c copy \
    -y vid2.mp4
