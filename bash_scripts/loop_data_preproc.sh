monkey_names=(venus paul baby1 louie)
imec_list=(0 1)
dates=(19 20 21 22 23 24)
path=/Volumes/LivingstoneLab/Data/Data-Neuropixels-Preprocessed
path_plx=/Volumes/LivingstoneLab/Data/Data-Formatted
for name in ${monkey_names[@]}; do
    for day in ${dates[@]}; do
        if [[ "$name" = "paul" ]]; then
            realpath ${path_plx}/${name}_202509${day}-rasters.h5 #_plx
        fi            
        for imec in ${imec_list[@]}; do
            if [[ "$name" == "baby" && "$imec" == "1" ]]; then
                :
            else
                realpath ${path}/${name}_2509${day} #_npx${imec}            
            fi            
        done
    done
done

