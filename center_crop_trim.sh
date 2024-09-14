#!/usr/bin/ens sh


input_dir=$1
img_name=${2-img_deformed_dirichlet_N_N_}
nw=${3-8}

imgs=$(find "${input_dir}" -type f -name "*${img_name}*.png" -printf "%p\n")

W=$(echo "${imgs}" \
    | xargs -P"${nw}" -I{} magick {} -format "%w\n" info: \
)
Wmin=$(printf "%s\n" $W | sort | head -n 1)
Wmax=$(printf "%s\n" $W | sort -r | head -n 1)
echo $Wmin $Wmax
W=$Wmin
#W=$Wmax
if [ $Wmin -ne $Wmax ]; then
    echo "${imgs}" | xargs -P"${nw}" -I {} magick {} -gravity center -crop "${Wmin}x${Wmin}+0+0" +repage PNG24:{}
    #echo "${imgs}" | xargs -P"${nw}" -I {} magick {} -gravity center -extent "${Wmax}x${Wmax}" +repage PNG24:{}
fi

echo $W

res=$(\
    echo "${imgs}" \
    | sort \
    | xargs -P"${nw}" -I{} magick -fuzz 0.1% {} -format "%@\n" info: \
)

ws=$(printf "%s\n" $res | cut -d'x' -f 1)
hs=$(printf "%s\n" $res | cut -d'x' -f 2 | cut -d'+' -f 1)
xmins=$(printf "%s\n" $res | cut -d'+' -f 2)
ymins=$(printf "%s\n" $res | cut -d'+' -f 3)
xmaxs=$(paste <(printf "%s\n" $ws) <(printf "%s\n" $xmins) | awk '{ print $1 + $2 }')
ymaxs=$(paste <(printf "%s\n" $hs) <(printf "%s\n" $ymins) | awk '{ print $1 + $2 }')

xmin=$(printf "%s\n" $xmins | sort | head -n 1)
ymin=$(printf "%s\n" $ymins | sort | head -n 1)
xmax=$(printf "%s\n" $xmaxs | sort -r | head -n 1)
ymax=$(printf "%s\n" $ymaxs | sort -r | head -n 1)

echo $xmin $ymin $xmax $ymax

if [ "${ymin}" -lt "${xmin}" ]; then
    xmin=$ymin
fi

if [ "${ymax}" -gt "${xmax}" ]; then
    xmax=$ymax
fi
echo $xmin $xmax

xc=$((W/2))
wl=$((xc-xmin))
wr=$((xmax-xc))
if [ "${wl}" -gt "${wr}" ]; then
    w=$((2*wl))
else
    w=$((2*wr))
fi

h=$w
echo $w $h

echo "${imgs}" | xargs -P"${nw}" -I {} magick {} -gravity center -crop "${w}x${h}+0+0" +repage PNG24:{}
