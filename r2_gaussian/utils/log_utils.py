import os
import sys
import uuid
import os.path as osp
from argparse import Namespace
import yaml

try:
    from tensorboardX import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    import wandb
    WANDB_FOUND = True
except ImportError:
    WANDB_FOUND = False

sys.path.append("./")
from r2_gaussian.utils.cfg_utils import args2string


def prepare_output_and_logger(args, logger_type="tensorboard", project_name="r2_gaussian"):
    """
    Create output directory and initialize logger.
    Args:
        args: Arguments containing model_path and other configs
        logger_type: Type of logger to use - "tensorboard" or "wandb"
    Returns:
        logger: Tensorboard writer or Wandb logger
    """
    assert logger_type in ["tensorboard", "wandb"], "Logger type must be tensorboard or wandb"
    
    # Initialize logger first
    logger = None
    if logger_type == "tensorboard" and TENSORBOARD_FOUND:
        if args.model_path:
            logger = SummaryWriter(args.model_path)
    elif logger_type == "wandb" and WANDB_FOUND:
        logger = wandb.init(
            project=project_name,
            config=vars(args),
        )
    else:
        print(f"{logger_type} not available: not logging progress")

    # Update model path if not specified
    # if not args.model_path:
    if os.getenv("OAR_JOB_ID"):
        unique_str = os.getenv("OAR_JOB_ID")
    else:
        unique_str = str(uuid.uuid4())
    args.model_path = osp.join(args.model_path, unique_str[0:10])
    if logger_type == "tensorboard" and logger:
        logger = SummaryWriter(args.model_path)

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(osp.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Save to yaml
    args_dict = vars(args)
    with open(osp.join(args.model_path, "cfg_args.yml"), "w") as f:
        yaml.dump(args_dict, f, default_flow_style=False, sort_keys=False)

    # Add args text for tensorboard
    if logger_type == "tensorboard" and logger:
        logger.add_text("args", args2string(args_dict), global_step=0)

    return logger
