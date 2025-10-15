monkey_names=(venus paul baby)
imec_list=(0 1)
dates=(19 20 21 22 23 24)
for name in ${monkey_names[@]}; do
    for day in ${dates[@]}; do
        if [[ "$name" = "paul" ]]; then
            echo ${name}_202509${day}_plx
        fi            
        for imec in ${imec_list[@]}; do
            if [[ "$name" == "baby" && "$imec" == "1" ]]; then
                :
            else
                echo ${name}_2509${day}_npx${imec}            
            fi            
        done
    done
done

