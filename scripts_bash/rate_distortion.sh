python3 train_ours.py -s data/synthetic_dataset/cone_ntrain_25_angle_360/1_broccoli_cone --render_backend slang --wandb --densify_scale_threshold 0.05
python3 train_ours.py -s data/synthetic_dataset/cone_ntrain_25_angle_360/1_broccoli_cone --render_backend slang --wandb --densify_scale_threshold 0.08
python3 train_ours.py -s data/synthetic_dataset/cone_ntrain_25_angle_360/1_broccoli_cone --render_backend slang --wandb --densify_scale_threshold 0.15

python3 train_ours.py -s data/synthetic_dataset/cone_ntrain_25_angle_360/1_broccoli_cone --render_backend slang --wandb --density_lr_init 0.008 --density_lr_final 0.008
python3 train_ours.py -s data/synthetic_dataset/cone_ntrain_25_angle_360/1_broccoli_cone --render_backend slang --wandb --density_lr_init 0.03 --density_lr_final 0.003
python3 train_ours.py -s data/synthetic_dataset/cone_ntrain_25_angle_360/1_broccoli_cone --render_backend slang --wandb --density_lr_init 0.005 --density_lr_final 0.0005

python3 train_ours.py -s data/synthetic_dataset/cone_ntrain_25_angle_360/1_broccoli_cone --render_backend slang --wandb --densify_grad_threshold 3e-5  
python3 train_ours.py -s data/synthetic_dataset/cone_ntrain_25_angle_360/1_broccoli_cone --render_backend slang --wandb --densify_grad_threshold 7e-5  
python3 train_ours.py -s data/synthetic_dataset/cone_ntrain_25_angle_360/1_broccoli_cone --render_backend slang --wandb  --densify_grad_threshold 1e-4  



# python3 train_ours.py -s data/synthetic_dataset/cone_ntrain_25_angle_360/1_broccoli_cone --render_backend r2 --wandb --densify_scale_threshold 0.05
# python3 train_ours.py -s data/synthetic_dataset/cone_ntrain_25_angle_360/1_broccoli_cone --render_backend r2 --wandb --densify_scale_threshold 0.08
# python3 train_ours.py -s data/synthetic_dataset/cone_ntrain_25_angle_360/1_broccoli_cone --render_backend r2 --wandb --densify_scale_threshold 0.15

# python3 train_ours.py -s data/synthetic_dataset/cone_ntrain_25_angle_360/1_broccoli_cone --render_backend r2 --wandb --density_lr_init 0.008 --density_lr_final 0.008
# python3 train_ours.py -s data/synthetic_dataset/cone_ntrain_25_angle_360/1_broccoli_cone --render_backend r2 --wandb --density_lr_init 0.03 --density_lr_final 0.003
# python3 train_ours.py -s data/synthetic_dataset/cone_ntrain_25_angle_360/1_broccoli_cone --render_backend r2 --wandb --density_lr_init 0.005 --density_lr_final 0.0005

# python3 train_ours.py -s data/synthetic_dataset/cone_ntrain_25_angle_360/1_broccoli_cone --render_backend r2 --wandb --densify_grad_threshold 3e-5  
# python3 train_ours.py -s data/synthetic_dataset/cone_ntrain_25_angle_360/1_broccoli_cone --render_backend r2 --wandb --densify_grad_threshold 7e-5  
# python3 train_ours.py -s data/synthetic_dataset/cone_ntrain_25_angle_360/1_broccoli_cone --render_backend r2 --wandb  --densify_grad_threshold 1e-4  


## Later try some experiments wiht min_density_threshold

# python3 train_ours.py -s data/real_dataset/cone_ntrain_50_angle_360/walnut/ --render_backend slang --wandb -m output_rate_distortion/3_25k_ours --subsample_ratio 10 --max_pts 25000
# python3 train_ours.py -s data/real_dataset/cone_ntrain_50_angle_360/walnut/ --render_backend slang --wandb -m output_rate_distortion/3_35k_ours --subsample_ratio 10 --max_pts 35000
# python3 train_ours.py -s data/real_dataset/cone_ntrain_50_angle_360/walnut/ --render_backend slang --wandb -m output_rate_distortion/3_45k_ours --subsample_ratio 10 --max_pts 45000
# python3 train_ours.py -s data/real_dataset/cone_ntrain_50_angle_360/walnut/ --render_backend slang --wandb -m output_rate_distortion/3_55k_ours --subsample_ratio 10 --max_pts 55000

# python3 train_ours.py -s data/synthetic_dataset/cone_ntrain_25_angle_360/1_broccoli_cone --render_backend r2 --wandb 
# python3 train_ours.py -s data/real_dataset/cone_ntrain_50_angle_360/walnut/ --render_backend r2 --wandb -m output_rate_distortion/3_25k_r2 --subsample_ratio 10 --max_pts 25000
# python3 train_ours.py -s data/real_dataset/cone_ntrain_50_angle_360/walnut/ --render_backend r2 --wandb -m output_rate_distortion/3_35k_r2 --subsample_ratio 10 --max_pts 35000
# python3 train_ours.py -s data/real_dataset/cone_ntrain_50_angle_360/walnut/ --render_backend r2 --wandb -m output_rate_distortion/3_45k_r2 --subsample_ratio 10 --max_pts 45000
# python3 train_ours.py -s data/real_dataset/cone_ntrain_50_angle_360/walnut/ --render_backend r2 --wandb -m output_rate_distortion/3_55k_r2 --subsample_ratio 10 --max_pts 55000