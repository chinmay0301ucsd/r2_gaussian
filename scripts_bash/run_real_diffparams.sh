# python3 train_ours.py -s data/real_dataset/cone_ntrain_25_angle_360/pine --render_backend slang --wandb --densify_scale_threshold 0.05 --densify_grad_threshold 2.0e-5 --lambda_tv 0.08 -m out_newp_ours
# python3 train_ours.py -s data/real_dataset/cone_ntrain_25_angle_360/seashell --render_backend slang --wandb --densify_scale_threshold 0.1 --densify_grad_threshold 2.0e-5 --lambda_tv 0.08 -m out_newp_ours --detect_anomaly
# python3 train_ours.py -s data/real_dataset/cone_ntrain_25_angle_360/walnut --render_backend slang --wandb --densify_scale_threshold 0.05 --densify_grad_threshold 2.0e-5 --lambda_tv 0.08 -m out_newp_ours

# python3 train_ours.py -s data/real_dataset/cone_ntrain_25_angle_360/pine --render_backend r2 --wandb --densify_scale_threshold 0.05 --densify_grad_threshold 2.0e-5 --lambda_tv 0.08 -m out_newp_r2
# python3 train_ours.py -s data/real_dataset/cone_ntrain_25_angle_360/seashell --render_backend r2 --wandb --densify_scale_threshold 0.05 --densify_grad_threshold 2.0e-5 --lambda_tv 0.08 -m out_newp_r2
# python3 train_ours.py -s data/real_dataset/cone_ntrain_25_angle_360/walnut --render_backend r2 --wandb --densify_scale_threshold 0.05 --densify_grad_threshold 2.0e-5 --lambda_tv 0.08 -m out_newp_r2

# python3 train_ours.py -s data/real_dataset/cone_ntrain_25_angle_360/pine --render_backend r2 --wandb -m out_oldp_r2
# python3 train_ours.py -s data/real_dataset/cone_ntrain_25_angle_360/seashell --render_backend r2 --wandb  -m out_oldp_r2
# python3 train_ours.py -s data/real_dataset/cone_ntrain_25_angle_360/walnut --render_backend r2 --wandb  -m out_oldp_r2

python3 train_ours.py -s data/real_dataset/cone_ntrain_25_angle_360/pine --render_backend r2 --wandb --lambda_tv 0.08 -m out_newp_r2
python3 train_ours.py -s data/real_dataset/cone_ntrain_25_angle_360/pine --render_backend slang --wandb --lambda_tv 0.08 -m out_newp_ours

python3 train_ours.py -s data/real_dataset/cone_ntrain_25_angle_360/seashell --render_backend r2 --wandb --lambda_tv 0.08 -m out_newp_r2
python3 train_ours.py -s data/real_dataset/cone_ntrain_25_angle_360/seashell --render_backend slang --wandb --lambda_tv 0.08 -m out_newp_ours
# python3 train_ours.py -s data/real_dataset/cone_ntrain_25_angle_360/walnut --render_backend slang --wandb --densify_scale_threshold 0.08 --densify_grad_threshold 2.0e-5 --lambda_tv 0.08 -m out_newp_ours
