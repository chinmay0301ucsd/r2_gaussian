# python3 train_ours.py -s data/synthetic_dataset/cone_ntrain_25_angle_360/0_chest_cone/ --render_backend slang --wandb --densify_grad_threshold 3.5e-5 #--densify_scale_threshold 0.1 --density_lr_init 0.01 --density_lr_final 0.001 --density_lr_max_steps 30_000 --scaling_lr_init 0.005 --scaling_lr_final 0.0005 --scaling_lr_max_steps 30_000 --rotation_lr_init 0.001 --rotation_lr_final 0.0001 --rotation_lr_max_steps 30_000 --lambda_dssim 0.25 --lambda_tv 0.05 --tv_vol_size 32 --densify_from_iter 500 --densify_until_iter 15000 --max_num_gaussians 500_000
# python3 train_ours.py -s data/synthetic_dataset/cone_ntrain_25_angle_360/0_foot_cone/ --render_backend slang --wandb --densify_grad_threshold 3.5e-5
# python3 train_ours.py -s data/synthetic_dataset/cone_ntrain_25_angle_360/0_head_cone/ --render_backend slang --wandb --densify_grad_threshold 3.5e-5
# python3 train_ours.py -s data/synthetic_dataset/cone_ntrain_25_angle_360/0_jaw_cone/ --render_backend slang --wandb --densify_grad_threshold 3.5e-5
# python3 train_ours.py -s data/synthetic_dataset/cone_ntrain_25_angle_360/0_pancreas_cone/ --render_backend slang --wandb --densify_grad_threshold 3.5e-5
# python3 train_ours.py -s data/synthetic_dataset/cone_ntrain_25_angle_360/1_beetle_cone/ --render_backend slang --wandb --densify_grad_threshold 3.5e-5
# python3 train_ours.py -s data/synthetic_dataset/cone_ntrain_25_angle_360/1_bonsai_cone/ --render_backend slang --wandb --densify_grad_threshold 3.5e-5
# python3 train_ours.py -s data/synthetic_dataset/cone_ntrain_25_angle_360/1_broccoli_cone/ --render_backend slang --wandb --densify_grad_threshold 3.5e-5
# python3 train_ours.py -s data/synthetic_dataset/cone_ntrain_25_angle_360/1_kingsnake_cone/ --render_backend slang --wandb --densify_grad_threshold 3.5e-5
# python3 train_ours.py -s data/synthetic_dataset/cone_ntrain_25_angle_360/1_pepper_cone/ --render_backend slang --wandb --densify_grad_threshold 3.5e-5
python3 train_ours.py -s data/synthetic_dataset/cone_ntrain_25_angle_360/2_backpack_cone/ --render_backend slang --wandb --densify_grad_threshold 3.5e-5
python3 train_ours.py -s data/synthetic_dataset/cone_ntrain_25_angle_360/2_engine_cone/ --render_backend slang --wandb --densify_grad_threshold 3.5e-5
python3 train_ours.py -s data/synthetic_dataset/cone_ntrain_25_angle_360/2_mount_cone/ --render_backend slang --wandb --densify_grad_threshold 3.5e-5
python3 train_ours.py -s data/synthetic_dataset/cone_ntrain_25_angle_360/2_present_cone/ --render_backend slang --wandb --densify_grad_threshold 3.5e-5
python3 train_ours.py -s data/synthetic_dataset/cone_ntrain_25_angle_360/2_teapot_cone/ --render_backend slang --wandb --densify_grad_threshold 3.5e-5

python3 train_ours.py -s data/synthetic_dataset/cone_ntrain_50_angle_360/0_chest_cone/ --render_backend slang --wandb --densify_grad_threshold 3.5e-5 -m chest-50#--densify_scale_threshold 0.1 --density_lr_init 0.01 --density_lr_final 0.001 --density_lr_max_steps 30_000 --scaling_lr_init 0.005 --scaling_lr_final 0.0005 --scaling_lr_max_steps 30_000 --rotation_lr_init 0.001 --rotation_lr_final 0.0001 --rotation_lr_max_steps 30_000 --lambda_dssim 0.25 --lambda_tv 0.05 --tv_vol_size 32 --densify_from_iter 500 --densify_until_iter 15000 --max_num_gaussians 500_000
python3 train_ours.py -s data/synthetic_dataset/cone_ntrain_50_angle_360/0_foot_cone/ --render_backend slang --wandb --densify_grad_threshold 3.5e-5 -m foot-50
python3 train_ours.py -s data/synthetic_dataset/cone_ntrain_50_angle_360/0_head_cone/ --render_backend slang --wandb --densify_grad_threshold 3.5e-5 -m head-50
python3 train_ours.py -s data/synthetic_dataset/cone_ntrain_50_angle_360/0_jaw_cone/ --render_backend slang --wandb --densify_grad_threshold 3.5e-5 -m jaw-50
python3 train_ours.py -s data/synthetic_dataset/cone_ntrain_50_angle_360/0_pancreas_cone/ --render_backend slang --wandb --densify_grad_threshold 3.5e-5 -m pancreas-50
python3 train_ours.py -s data/synthetic_dataset/cone_ntrain_50_angle_360/1_beetle_cone/ --render_backend slang --wandb --densify_grad_threshold 3.5e-5 -m beetle-50
python3 train_ours.py -s data/synthetic_dataset/cone_ntrain_50_angle_360/1_bonsai_cone/ --render_backend slang --wandb --densify_grad_threshold 3.5e-5 -m bonsai-50
python3 train_ours.py -s data/synthetic_dataset/cone_ntrain_50_angle_360/1_broccoli_cone/ --render_backend slang --wandb --densify_grad_threshold 3.5e-5 -m broccoli-50
python3 train_ours.py -s data/synthetic_dataset/cone_ntrain_50_angle_360/1_kingsnake_cone/ --render_backend slang --wandb --densify_grad_threshold 3.5e-5 -m kingsnake-50
python3 train_ours.py -s data/synthetic_dataset/cone_ntrain_50_angle_360/1_pepper_cone/ --render_backend slang --wandb --densify_grad_threshold 3.5e-5 -m pepper-50
python3 train_ours.py -s data/synthetic_dataset/cone_ntrain_50_angle_360/2_backpack_cone/ --render_backend slang --wandb --densify_grad_threshold 3.5e-5 -m backpack-50
python3 train_ours.py -s data/synthetic_dataset/cone_ntrain_50_angle_360/2_engine_cone/ --render_backend slang --wandb --densify_grad_threshold 3.5e-5 -m engine-50
python3 train_ours.py -s data/synthetic_dataset/cone_ntrain_50_angle_360/2_mount_cone/ --render_backend slang --wandb --densify_grad_threshold 3.5e-5 -m mount-50
python3 train_ours.py -s data/synthetic_dataset/cone_ntrain_50_angle_360/2_present_cone/ --render_backend slang --wandb --densify_grad_threshold 3.5e-5 -m present-50
python3 train_ours.py -s data/synthetic_dataset/cone_ntrain_50_angle_360/2_teapot_cone/ --render_backend slang --wandb --densify_grad_threshold 3.5e-5 -m teapot-50
