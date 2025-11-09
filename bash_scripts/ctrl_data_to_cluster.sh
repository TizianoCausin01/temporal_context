monkey_names=(venus paul baby1 louie)
dates=(19 20 21 22 23 24)
loc_path=/n/files/Neurobio/LivingstoneLab/Data/Data-Neuropixels-Preprocessed
loc_path_plx=/n/files/Neurobio/LivingstoneLab/Data/Data-Formatted
cluster_path=/n/data2/hms/neurobio/livingstone/Data/Npx-Preprocessed
cluster_path_plx=/n/data2/hms/neurobio/livingstone/Data/Formatted

for name in ${monkey_names[@]}; do
    for day in ${dates[@]}; do
        ls -l ${cluster_path}/${name}_2509${day}/catgt_${name}_2509${day}_g0/${name}_2509${day}_g0_imec0/${name}_2509${day}-imec0-mua_cont.h5
        done
    done


