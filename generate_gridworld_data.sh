declare -a arr=("up" "down" "border" "center")


for i in "${arr[@]}"
do
    CUDA_VISIBLE_DEVICES='' python3 run_gridworld_multipleintent.py --run_policy --num_exp 10 --demonstration 1000 --compute_gradient --filter_gradients --direction "$i"

done


