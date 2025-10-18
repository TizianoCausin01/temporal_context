monkey_names=(venus paul baby1 louie)
dates=(19 20 21 22 23 24)
loc_path=/n/files/Neurobio/LivingstoneLab/Data/Data-Neuropixels-Preprocessed
loc_path_plx=/n/files/Neurobio/LivingstoneLab/Data/Data-Formatted
cluster_path=/n/data2/hms/neurobio/livingstone/Data/Npx-Preprocessed
cluster_path_plx=/n/data2/hms/neurobio/livingstone/Data/Formatted

for name in ${monkey_names[@]}; do
    for day in ${dates[@]}; do
        if [[ "$name" = "paul" ]]; then
            cp ${loc_path_plx}/${name}_202509${day}-rasters.h5 ${cluster_path_plx}/${name}_202509${day}-rasters.h5 #_plx
        fi            
        cp -r ${loc_path}/${name}_2509${day} ${cluster_path}/${name}_2509${day}         
        ls -l ${cluster_path}/${name}_2509${day}
        cp ${loc_path_plx}/${name}_2509${day}_experiment.mat ${cluster_path_plx}/${name}_2509${day}_experiment.mat #_plx
        done
    done


