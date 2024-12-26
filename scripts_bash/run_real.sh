## Our Method
# python3 train_ours.py -s data/real_dataset/cone_ntrain_25_angle_360/pine/ --render_backend slang --wandb --densify_grad_threshold 3.5e-5 #--densify_scale_threshold 0.1 --density_lr_init 0.01 --density_lr_final 0.001 --density_lr_max_steps 30_000 --scaling_lr_init 0.005 --scaling_lr_final 0.0005 --scaling_lr_max_steps 30_000 --rotation_lr_init 0.001 --rotation_lr_final 0.0001 --rotation_lr_max_steps 30_000 --lambda_dssim 0.25 --lambda_tv 0.05 --tv_vol_size 32 --densify_from_iter 500 --densify_until_iter 15000 --max_num_gaussians 500_000
# python3 train_ours.py -s data/real_dataset/cone_ntrain_25_angle_360/seashell/ --render_backend slang --wandb --densify_grad_threshold 3.5e-5
# python3 train_ours.py -s data/real_dataset/cone_ntrain_25_angle_360/walnut/ --render_backend slang --wandb --densify_grad_threshold 3.5e-5

# python3 train_ours.py -s data/real_dataset/cone_ntrain_50_angle_360/pine/ --render_backend slang --wandb --densify_grad_threshold 3.5e-5 #--densify_scale_threshold 0.1 --density_lr_init 0.01 --density_lr_final 0.001 --density_lr_max_steps 30_000 --scaling_lr_init 0.005 --scaling_lr_final 0.0005 --scaling_lr_max_steps 30_000 --rotation_lr_init 0.001 --rotation_lr_final 0.0001 --rotation_lr_max_steps 30_000 --lambda_dssim 0.25 --lambda_tv 0.05 --tv_vol_size 32 --densify_from_iter 500 --densify_until_iter 15000 --max_num_gaussians 500_000
# python3 train_ours.py -s data/real_dataset/cone_ntrain_50_angle_360/seashell/ --render_backend slang --wandb --densify_grad_threshold 3.5e-5
# python3 train_ours.py -s data/real_dataset/cone_ntrain_50_angle_360/walnut/ --render_backend slang --wandb --densify_grad_threshold 3.5e-5

# python3 train_ours.py -s data/real_dataset/cone_ntrain_75_angle_360/pine/ --render_backend slang --wandb --densify_grad_threshold 3.5e-5 #--densify_scale_threshold 0.1 --density_lr_init 0.01 --density_lr_final 0.001 --density_lr_max_steps 30_000 --scaling_lr_init 0.005 --scaling_lr_final 0.0005 --scaling_lr_max_steps 30_000 --rotation_lr_init 0.001 --rotation_lr_final 0.0001 --rotation_lr_max_steps 30_000 --lambda_dssim 0.25 --lambda_tv 0.05 --tv_vol_size 32 --densify_from_iter 500 --densify_until_iter 15000 --max_num_gaussians 500_000
# python3 train_ours.py -s data/real_dataset/cone_ntrain_75_angle_360/seashell/ --render_backend slang --wandb --densify_grad_threshold 3.5e-5
# python3 train_ours.py -s data/real_dataset/cone_ntrain_75_angle_360/walnut/ --render_backend slang --wandb --densify_grad_threshold 3.5e-5

## Baseline
python3 train_ours.py -s data/real_dataset/cone_ntrain_25_angle_360/pine/ --render_backend slang --wandb -m pine-25-default
python3 train_ours.py -s data/real_dataset/cone_ntrain_25_angle_360/seashell/ --render_backend slang --wandb -m seashell-25-default
python3 train_ours.py -s data/real_dataset/cone_ntrain_25_angle_360/walnut/ --render_backend slang --wandb -m walnut-25-default

python3 train_ours.py -s data/real_dataset/cone_ntrain_50_angle_360/pine/ --render_backend slang --wandb -m pine-50-default #--densify_scale_threshold 0.1 --density_lr_init 0.01 --density_lr_final 0.001 --density_lr_max_steps 30_000 --scaling_lr_init 0.005 --scaling_lr_final 0.0005 --scaling_lr_max_steps 30_000 --rotation_lr_init 0.001 --rotation_lr_final 0.0001 --rotation_lr_max_steps 30_000 --lambda_dssim 0.25 --lambda_tv 0.05 --tv_vol_size 32 --densify_from_iter 500 --densify_until_iter 15000 --max_num_gaussians 500_000
python3 train_ours.py -s data/real_dataset/cone_ntrain_50_angle_360/seashell/ --render_backend slang --wandb -m seashell-50-default
python3 train_ours.py -s data/real_dataset/cone_ntrain_50_angle_360/walnut/ --render_backend slang --wandb -m walnut-50-default

python3 train_ours.py -s data/real_dataset/cone_ntrain_75_angle_360/pine/ --render_backend slang --wandb -m pine-75-default
python3 train_ours.py -s data/real_dataset/cone_ntrain_75_angle_360/seashell/ --render_backend slang --wandb -m seashell-75-default
python3 train_ours.py -s data/real_dataset/cone_ntrain_75_angle_360/walnut/ --render_backend slang --wandb -m walnut-75-default







        # self.density_min_threshold = 0.00001
        # self.densification_interval = 100
        # self.densify_from_iter = 500
        # self.densify_until_iter = 15000
        # self.densify_grad_threshold = 5.0e-5
        # self.densify_scale_threshold = 0.1  # percent of volume size
        # self.max_screen_size = None
        # self.max_scale = None  # percent of volume size
        # self.max_num_gaussians = 500_000

		#         self.iterations = 30_000
        # self.position_lr_init = 0.0002
        # self.position_lr_final = 0.00002
        # self.position_lr_max_steps = 30_000
        # self.density_lr_init = 0.01
        # self.density_lr_final = 0.001
        # self.density_lr_max_steps = 30_000
        # self.scaling_lr_init = 0.005
        # self.scaling_lr_final = 0.0005
        # self.scaling_lr_max_steps = 30_000
        # self.rotation_lr_init = 0.001
        # self.rotation_lr_final = 0.0001
        # self.rotation_lr_max_steps = 30_000
        # self.lambda_dssim = 0.25
        # self.lambda_tv = 0.05
        # self.tv_vol_size = 32
        # self.density_min_threshold = 0.00001
        # self.densification_interval = 100
        # self.densify_from_iter = 500
        # self.densify_until_iter = 15000
        # self.densify_grad_threshold = 5.0e-5
        # self.densify_scale_threshold = 0.1  # percent of volume size
        # self.max_screen_size = None
        # self.max_scale = None  # percent of volume size
        # self.max_num_gaussians = 500_000