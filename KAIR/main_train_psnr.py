import os.path
import math
import argparse
import random
import numpy as np
import logging

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import torch

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist

from data.select_dataset import define_Dataset
from models.select_model import define_Model


# --------------------------------------------
# Training code for SwinIR / KAIR
# Modified for KLA dataset
# --------------------------------------------


def main(json_path='options/train_msrresnet_psnr.json'):

    # ----------------------------------------
    # Step 1: Prepare options
    # ----------------------------------------

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--opt',
        type=str,
        default=json_path,
        help='Path to option JSON file.'
    )

    parser.add_argument(
        '--launcher',
        default='pytorch',
        help='job launcher'
    )

    parser.add_argument(
        '--local_rank',
        type=int,
        default=0
    )

    parser.add_argument(
        '--dist',
        default=False
    )

    # ----------------------------------------
    # NEW:
    # Maximum total training iterations
    # ----------------------------------------

    parser.add_argument(
        '--max_iter',
        type=int,
        default=100000,
        help='Maximum total training iterations.'
    )

    args = parser.parse_args()

    # ----------------------------------------
    # Parse options
    # ----------------------------------------

    opt = option.parse(
        args.opt,
        is_train=True
    )

    opt['dist'] = args.dist

    # ----------------------------------------
    # Distributed settings
    # ----------------------------------------

    if opt['dist']:
        init_dist('pytorch')

    opt['rank'], opt['world_size'] = get_dist_info()

    # ----------------------------------------
    # Create required directories
    # ----------------------------------------

    if opt['rank'] == 0:

        util.mkdirs(
            (
                path
                for key, path in opt['path'].items()
                if 'pretrained' not in key
            )
        )

    # ----------------------------------------
    # GPU / CUDA CHECK
    # ----------------------------------------

    print()
    print("==========================================")
    print("GPU / CUDA CHECK")
    print("==========================================")

    print("PyTorch version :", torch.__version__)
    print("CUDA version    :", torch.version.cuda)
    print("CUDA available  :", torch.cuda.is_available())

    if torch.cuda.is_available():

        print(
            "GPU count       :",
            torch.cuda.device_count()
        )

        for gpu_index in range(torch.cuda.device_count()):

            print(
                "GPU",
                gpu_index,
                ":",
                torch.cuda.get_device_name(gpu_index)
            )

    else:

        print("WARNING: CUDA is NOT available.")

    print("==========================================")
    print()

    # ----------------------------------------
    # Find last checkpoints
    # ----------------------------------------

    init_iter_G, init_path_G = option.find_last_checkpoint(
        opt['path']['models'],
        net_type='G'
    )

    init_iter_E, init_path_E = option.find_last_checkpoint(
        opt['path']['models'],
        net_type='E'
    )

    opt['path']['pretrained_netG'] = init_path_G
    opt['path']['pretrained_netE'] = init_path_E

    init_iter_optimizerG, init_path_optimizerG = option.find_last_checkpoint(
        opt['path']['models'],
        net_type='optimizerG'
    )

    opt['path']['pretrained_optimizerG'] = init_path_optimizerG

    # ----------------------------------------
    # Resume from latest checkpoint
    # ----------------------------------------

    current_step = max(
        init_iter_G,
        init_iter_E,
        init_iter_optimizerG
    )

    print()
    print("==========================================")
    print("CHECKPOINT / RESUME INFORMATION")
    print("==========================================")

    print("Latest G checkpoint iteration       :", init_iter_G)
    print("Latest E checkpoint iteration       :", init_iter_E)
    print("Latest optimizer checkpoint        :", init_iter_optimizerG)
    print("Starting training iteration        :", current_step)
    print("Target training iteration          :", args.max_iter)

    if init_path_G is not None:
        print("G checkpoint                      :", init_path_G)

    if init_path_optimizerG is not None:
        print("Optimizer checkpoint              :", init_path_optimizerG)

    print("==========================================")
    print()

    # ----------------------------------------
    # Stop immediately if already completed
    # ----------------------------------------

    if current_step >= args.max_iter:

        print()
        print("==========================================")
        print("TRAINING TARGET ALREADY REACHED")
        print("==========================================")
        print("Current iteration:", current_step)
        print("Target iteration :", args.max_iter)
        print("Nothing more to train.")
        print("==========================================")
        print()

        return

    # ----------------------------------------
    # Border
    # ----------------------------------------

    border = opt['scale']

    # ----------------------------------------
    # Save options
    # ----------------------------------------

    if opt['rank'] == 0:
        option.save(opt)

    # ----------------------------------------
    # Convert dictionary to NoneDict
    # ----------------------------------------

    opt = option.dict_to_nonedict(opt)

    # ----------------------------------------
    # Configure logger
    # ----------------------------------------

    if opt['rank'] == 0:

        logger_name = 'train'

        utils_logger.logger_info(
            logger_name,
            os.path.join(
                opt['path']['log'],
                logger_name + '.log'
            )
        )

        logger = logging.getLogger(logger_name)

        logger.info(
            option.dict2str(opt)
        )

    # ----------------------------------------
    # Random seed
    # ----------------------------------------

    seed = opt['train']['manual_seed']

    if seed is None:
        seed = random.randint(1, 10000)

    print(
        'Random seed: {}'.format(seed)
    )

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ----------------------------------------
    # Step 2: Create datasets and dataloaders
    # ----------------------------------------

    print()
    print("==========================================")
    print("CREATING DATASETS")
    print("==========================================")

    for phase, dataset_opt in opt['datasets'].items():

        if phase == 'train':

            train_set = define_Dataset(
                dataset_opt
            )

            train_size = int(
                math.ceil(
                    len(train_set)
                    /
                    dataset_opt['dataloader_batch_size']
                )
            )

            if opt['rank'] == 0:

                logger.info(
                    'Number of train images: {:,d}, iters: {:,d}'.format(
                        len(train_set),
                        train_size
                    )
                )

            if opt['dist']:

                train_sampler = DistributedSampler(
                    train_set,
                    shuffle=dataset_opt['dataloader_shuffle'],
                    drop_last=True,
                    seed=seed
                )

                train_loader = DataLoader(
                    train_set,
                    batch_size=(
                        dataset_opt['dataloader_batch_size']
                        //
                        opt['num_gpu']
                    ),
                    shuffle=False,
                    num_workers=(
                        dataset_opt['dataloader_num_workers']
                        //
                        opt['num_gpu']
                    ),
                    drop_last=True,
                    pin_memory=True,
                    sampler=train_sampler
                )

            else:

                train_loader = DataLoader(
                    train_set,
                    batch_size=dataset_opt['dataloader_batch_size'],
                    shuffle=dataset_opt['dataloader_shuffle'],
                    num_workers=dataset_opt['dataloader_num_workers'],
                    drop_last=True,
                    pin_memory=True
                )

        elif phase == 'test':

            test_set = define_Dataset(
                dataset_opt
            )

            test_loader = DataLoader(
                test_set,
                batch_size=1,
                shuffle=False,
                num_workers=0,
                drop_last=False,
                pin_memory=True
            )

        else:

            raise NotImplementedError(
                "Phase [%s] is not recognized." % phase
            )

    print()
    print("==========================================")
    print("DATASET / DATALOADER READY")
    print("==========================================")

    print(
        "Training samples :",
        len(train_set)
    )

    print(
        "Training batches  :",
        train_size
    )

    print(
        "Batch size        :",
        opt['datasets']['train']['dataloader_batch_size']
    )

    print(
        "Workers           :",
        opt['datasets']['train']['dataloader_num_workers']
    )

    print("==========================================")
    print()

    # ----------------------------------------
    # Step 3: Initialize model
    # ----------------------------------------

    print()
    print("==========================================")
    print("CREATING SWINIR MODEL")
    print("==========================================")

    model = define_Model(opt)

    print("Model object created.")

    model.init_train()

    print("Model training initialization completed.")

    if opt['rank'] == 0:

        logger.info(
            model.info_network()
        )

        logger.info(
            model.info_params()
        )

    print()
    print("==========================================")
    print("MODEL READY")
    print("==========================================")
    print()

    # ----------------------------------------
    # Step 4: Main training
    #
    # IMPORTANT:
    # Training is controlled by total iterations,
    # not by a fixed number of epochs.
    # ----------------------------------------

    epoch = current_step // train_size

    print()
    print("==========================================")
    print("TRAINING STARTED")
    print("==========================================")
    print("Starting iteration :", current_step)
    print("Target iteration   :", args.max_iter)
    print("Approx. epoch      :", epoch)
    print("==========================================")
    print()

    # ----------------------------------------
    # Continue until target iteration
    # ----------------------------------------

    while current_step < args.max_iter:

        if opt['dist']:

            train_sampler.set_epoch(
                epoch + seed
            )

        for i, train_data in enumerate(train_loader):

            # --------------------------------
            # Stop if target has been reached
            # --------------------------------

            if current_step >= args.max_iter:
                break

            current_step += 1

            # --------------------------------
            # ITERATION START
            # --------------------------------

            print()
            print("------------------------------------------")
            print(
                "STARTING ITERATION:",
                current_step
            )
            print("------------------------------------------")

            # --------------------------------
            # 1. Update learning rate
            # --------------------------------

            print(
                "[1/4] Updating learning rate..."
            )

            model.update_learning_rate(
                current_step
            )

            print(
                "[1/4] Learning rate update finished."
            )

            # --------------------------------
            # 2. Inspect batch
            # --------------------------------

            print(
                "[2/4] Inspecting training batch..."
            )

            if isinstance(train_data, dict):

                if 'L' in train_data:

                    print(
                        "L shape :",
                        train_data['L'].shape
                    )

                    print(
                        "L dtype:",
                        train_data['L'].dtype
                    )

                if 'H' in train_data:

                    print(
                        "H shape :",
                        train_data['H'].shape
                    )

                    print(
                        "H dtype:",
                        train_data['H'].dtype
                    )

            print(
                "[2/4] Batch inspection finished."
            )

            # --------------------------------
            # 3. Feed data
            # --------------------------------

            print(
                "[3/4] Feeding data to model..."
            )

            model.feed_data(
                train_data
            )

            print(
                "[3/4] Data successfully fed to model."
            )

            # --------------------------------
            # 4. Optimize
            # --------------------------------

            print(
                "[4/4] Starting forward + backward + optimizer..."
            )

            model.optimize_parameters(
                current_step
            )

            print(
                "[4/4] Optimization finished."
            )

            # --------------------------------
            # Training information
            # --------------------------------

            if (
                current_step
                %
                opt['train']['checkpoint_print']
                ==
                0
                and
                opt['rank'] == 0
            ):

                logs = model.current_log()

                message = (
                    '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '
                    .format(
                        epoch,
                        current_step,
                        model.current_learning_rate()
                    )
                )

                for k, v in logs.items():

                    message += (
                        '{:s}: {:.3e} '
                        .format(
                            k,
                            v
                        )
                    )

                logger.info(
                    message
                )

                print(
                    message
                )

            # --------------------------------
            # Save model
            # --------------------------------

            if (
                current_step
                %
                opt['train']['checkpoint_save']
                ==
                0
                and
                opt['rank'] == 0
            ):

                logger.info(
                    'Saving the model.'
                )

                print()
                print("==========================================")
                print("CHECKPOINT SAVING")
                print("==========================================")
                print(
                    "Saving checkpoint at iteration:",
                    current_step
                )

                model.save(
                    current_step
                )

                print(
                    "Checkpoint saved."
                )

                print("==========================================")
                print()

            # --------------------------------
            # Validation
            # --------------------------------

            if (
                current_step
                %
                opt['train']['checkpoint_test']
                ==
                0
                and
                opt['rank'] == 0
            ):

                print()
                print("==========================================")
                print("VALIDATION STARTED")
                print("==========================================")

                avg_psnr = 0.0

                idx = 0

                for test_data in test_loader:

                    idx += 1

                    image_name_ext = os.path.basename(
                        test_data['L_path'][0]
                    )

                    img_name, ext = os.path.splitext(
                        image_name_ext
                    )

                    img_dir = os.path.join(
                        opt['path']['images'],
                        img_name
                    )

                    util.mkdir(
                        img_dir
                    )

                    model.feed_data(
                        test_data
                    )

                    model.test()

                    visuals = model.current_visuals()

                    E_img = util.tensor2uint(
                        visuals['E']
                    )

                    H_img = util.tensor2uint(
                        visuals['H']
                    )

                    # --------------------------------
                    # Save restored image
                    # --------------------------------

                    save_img_path = os.path.join(
                        img_dir,
                        '{:s}_{:d}.png'.format(
                            img_name,
                            current_step
                        )
                    )

                    util.imsave(
                        E_img,
                        save_img_path
                    )

                    # --------------------------------
                    # Calculate PSNR
                    # --------------------------------

                    current_psnr = util.calculate_psnr(
                        E_img,
                        H_img,
                        border=border
                    )

                    logger.info(
                        '{:->4d}--> {:>10s} | {:<4.2f}dB'.format(
                            idx,
                            image_name_ext,
                            current_psnr
                        )
                    )

                    avg_psnr += current_psnr

                if idx > 0:

                    avg_psnr = avg_psnr / idx

                else:

                    avg_psnr = 0.0

                # --------------------------------
                # Validation result
                # --------------------------------

                logger.info(
                    '<epoch:{:3d}, iter:{:8,d}, Average PSNR : {:<.2f}dB\n'.format(
                        epoch,
                        current_step,
                        avg_psnr
                    )
                )

                print()
                print(
                    "Average PSNR:",
                    avg_psnr
                )

                print(
                    "=========================================="
                )

                print(
                    "VALIDATION FINISHED"
                )

                print(
                    "=========================================="
                )

                print()

            # --------------------------------
            # Target reached
            # --------------------------------

            if current_step >= args.max_iter:

                print()
                print("==========================================")
                print("TARGET ITERATION REACHED")
                print("==========================================")
                print(
                    "Training completed at iteration:",
                    current_step
                )
                print(
                    "Target iteration:",
                    args.max_iter
                )
                print("==========================================")
                print()

                return

        # ----------------------------------------
        # One complete epoch finished
        # ----------------------------------------

        epoch += 1

        print()
        print("==========================================")
        print("EPOCH COMPLETED")
        print("==========================================")
        print(
            "Epoch:",
            epoch
        )
        print(
            "Current iteration:",
            current_step
        )
        print(
            "Target iteration:",
            args.max_iter
        )
        print("==========================================")
        print()


# --------------------------------------------
# Main
# --------------------------------------------

if __name__ == '__main__':
    main()