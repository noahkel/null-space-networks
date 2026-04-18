# Null-Space-Networks
Systematic investigation of robustness of Null Space Networks

For data creation run:

python -u create_ellipse_data.py --img_size 128 --noise 0.05 --min_angle 0 --max_angle 90 --num_thetas 180 --n_samples --matrix_mode 1

For training run something like: 

python -u train.py --type "ellipses"

For attack run something like:

python -u attack.py --type "ellipses" --init fbp --models resnet,nsn,dpnsn,dpnsn_res --attacks adam --norm l2 --eps 5.0 --alpha 0.5 --steps 500 --data-root "/scratch/noah/data/ellipses_out" --model-dir "/scratch/noah/models"

