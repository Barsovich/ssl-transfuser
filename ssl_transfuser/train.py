import inspect
from pathlib import Path
import argparse
import json
from multiprocessing.sharedctypes import Value
import os
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import wandb
import torch.nn.functional as F

torch.backends.cudnn.benchmark = True
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ssl_transfuser.config import GlobalConfig
from ssl_transfuser.model import TransFuser
from ssl_transfuser.data import CARLA_Data


def get_config_dict(config):
    attributes = inspect.getmembers(config, lambda a: not (inspect.isroutine(a)))
    attributes = [
        a for a in attributes if not (a[0].startswith("__") and a[0].endswith("__"))
    ]
    return {k: v for k, v in attributes}


class Engine(object):
    """Engine that runs training and inference.
    Args
            - cur_epoch (int): Current epoch.
            - print_every (int): How frequently (# batches) to print loss.
            - validate_every (int): How frequently (# epochs) to run validation.

    """

    def __init__(
        self,
        model,
        optimizer,
        config,
        dataloader_train,
        dataloader_val,
        writer,
        cur_epoch=0,
        cur_iter=0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val
        self.writer = writer
        self.cur_epoch = cur_epoch
        self.cur_iter = cur_iter
        self.bestval_epoch = cur_epoch
        self.train_loss = []
        self.val_loss = []
        self.bestval = 1e10

    @staticmethod
    def compute_losses(
        pred_wp,
        gt_waypoints,
        next_frame_image_prediction,
        gt_next_frame_image,
        next_frame_lidar_prediction,
        gt_next_frame_lidar,
        lidar_to_image_prediction,
        gt_curr_frame_image,
        image_to_lidar_prediction,
        gt_curr_frame_lidar,
    ):
        waypoint_loss = F.l1_loss(pred_wp, gt_waypoints, reduction="none").mean()

        next_frame_image_prediction_loss = (
            args.next_frame_prediction_loss_coef
            * args.image_loss_coef
            * F.l1_loss(next_frame_image_prediction, gt_next_frame_image)
        )
        next_frame_lidar_prediction_loss = (
            args.next_frame_prediction_loss_coef
            * F.l1_loss(next_frame_lidar_prediction, gt_next_frame_lidar)
        )
        image_to_lidar_prediction_loss = (
            args.cross_modal_prediction_loss_coef
            * F.l1_loss(image_to_lidar_prediction, gt_curr_frame_lidar)
        )
        lidar_to_image_prediction_loss = (
            args.cross_modal_prediction_loss_coef
            * args.image_loss_coef
            * F.l1_loss(lidar_to_image_prediction, gt_curr_frame_image)
        )

        return (
            waypoint_loss,
            next_frame_image_prediction_loss,
            next_frame_lidar_prediction_loss,
            image_to_lidar_prediction_loss,
            lidar_to_image_prediction_loss,
        )

    def log_losses(
        self,
        loss,
        waypoint_loss,
        next_frame_image_prediction_loss,
        next_frame_lidar_prediction_loss,
        image_to_lidar_prediction_loss,
        lidar_to_image_prediction_loss,
    ):
        def log_to_wandb_and_tensorboard(key, value, iter):
            self.writer.add_scalar(key, value, iter)
            wandb.log({key: value}, step=iter)

        log_to_wandb_and_tensorboard("train_loss", loss.item(), self.cur_iter)
        log_to_wandb_and_tensorboard(
            "original_loss", waypoint_loss.item(), self.cur_iter
        )
        log_to_wandb_and_tensorboard(
            "next_frame_image_prediction_loss",
            next_frame_image_prediction_loss.item(),
            self.cur_iter,
        )
        log_to_wandb_and_tensorboard(
            "next_frame_image_prediction_loss",
            next_frame_image_prediction_loss.item(),
            self.cur_iter,
        )
        log_to_wandb_and_tensorboard(
            "next_frame_lidar_prediction_loss",
            next_frame_lidar_prediction_loss.item(),
            self.cur_iter,
        )
        log_to_wandb_and_tensorboard(
            "image_to_lidar_prediction_loss",
            image_to_lidar_prediction_loss.item(),
            self.cur_iter,
        )
        log_to_wandb_and_tensorboard(
            "lidar_to_image_prediction_loss",
            lidar_to_image_prediction_loss.item(),
            self.cur_iter,
        )

    def train(self):
        loss_epoch = 0.0
        num_batches = 0
        self.model.train()

        # Train loop
        for data in tqdm(self.dataloader_train):
            if self.cur_iter >= args.max_steps:
                break
            # efficiently zero gradients
            for p in self.model.parameters():
                p.grad = None

            # create batch and move to GPU
            fronts_in = data["fronts"]
            lefts_in = data["lefts"]
            rights_in = data["rights"]
            rears_in = data["rears"]
            lidars_in = data["lidars"]
            fronts = []
            lefts = []
            rights = []
            rears = []
            lidars = []
            for i in range(self.config.seq_len):
                fronts.append(fronts_in[i].to(args.device, dtype=torch.float32))
                if not self.config.ignore_sides:
                    lefts.append(lefts_in[i].to(args.device, dtype=torch.float32))
                    rights.append(rights_in[i].to(args.device, dtype=torch.float32))
                if not self.config.ignore_rear:
                    rears.append(rears_in[i].to(args.device, dtype=torch.float32))
                lidars.append(lidars_in[i].to(args.device, dtype=torch.float32))

            # same for the next frame
            next_fronts_in = data["next_fronts"]
            next_lefts_in = data["next_lefts"]
            next_rights_in = data["next_rights"]
            next_rears_in = data["next_rears"]
            next_lidars_in = data["next_lidars"]
            next_fronts = []
            next_lefts = []
            next_rights = []
            next_rears = []
            next_lidars = []
            for i in range(self.config.seq_len):
                next_fronts.append(
                    next_fronts_in[i].to(args.device, dtype=torch.float32)
                )
                if not self.config.ignore_sides:
                    next_lefts.append(
                        next_lefts_in[i].to(args.device, dtype=torch.float32)
                    )
                    next_rights.append(
                        next_rights_in[i].to(args.device, dtype=torch.float32)
                    )
                if not self.config.ignore_rear:
                    next_rears.append(
                        next_rears_in[i].to(args.device, dtype=torch.float32)
                    )
                next_lidars.append(
                    next_lidars_in[i].to(args.device, dtype=torch.float32)
                )

            # driving labels
            command = data["command"].to(args.device)
            gt_velocity = data["velocity"].to(args.device, dtype=torch.float32)
            gt_steer = data["steer"].to(args.device, dtype=torch.float32)
            gt_throttle = data["throttle"].to(args.device, dtype=torch.float32)
            gt_brake = data["brake"].to(args.device, dtype=torch.float32)

            # target point
            target_point = torch.stack(data["target_point"], dim=1).to(
                args.device, dtype=torch.float32
            )

            (
                pred_wp,
                image_to_lidar_prediction,
                lidar_to_image_prediction,
                next_frame_lidar_prediction,
                next_frame_image_prediction,
            ) = self.model(
                fronts + lefts + rights + rears, lidars, target_point, gt_velocity
            )

            gt_waypoints = [
                torch.stack(data["waypoints"][i], dim=1).to(
                    args.device, dtype=torch.float32
                )
                for i in range(self.config.seq_len, len(data["waypoints"]))
            ]
            gt_waypoints = torch.stack(gt_waypoints, dim=1).to(
                args.device, dtype=torch.float32
            )

            # Here we assume that the the sequence length of input is one
            logging.debug("Calculating loss")
            logging.debug("gt_waypoints.shape: {}".format(gt_waypoints.shape))
            logging.debug("pred_wp.shape: {}".format(pred_wp.shape))

            (
                waypoint_loss,
                next_frame_image_prediction_loss,
                next_frame_lidar_prediction_loss,
                image_to_lidar_prediction_loss,
                lidar_to_image_prediction_loss,
            ) = self.compute_losses(
                pred_wp,
                gt_waypoints,
                next_frame_image_prediction,
                next_fronts[0],
                next_frame_lidar_prediction,
                next_lidars[0],
                lidar_to_image_prediction,
                fronts[0],
                image_to_lidar_prediction,
                lidars[0],
            )

            loss = (
                waypoint_loss
                + next_frame_image_prediction_loss
                + next_frame_lidar_prediction_loss
                + image_to_lidar_prediction_loss
                + lidar_to_image_prediction_loss
            )

            loss.backward()
            loss_epoch += float(loss.item())

            num_batches += 1
            self.optimizer.step()

            self.log_losses(
                loss,
                waypoint_loss,
                next_frame_image_prediction_loss,
                next_frame_lidar_prediction_loss,
                image_to_lidar_prediction_loss,
                lidar_to_image_prediction_loss,
            )

            self.cur_iter += 1

            if self.cur_iter % args.val_every == 0:
                logging.info("Running validation")
                self.validate()
                self.save()

        if num_batches > 0:
            loss_epoch = loss_epoch / num_batches
        else:
            loss_epoch = 0.0

        self.train_loss.append(loss_epoch)
        self.cur_epoch += 1

    def validate(self):
        self.model.eval()

        with torch.no_grad():
            num_batches = 0
            wp_epoch = 0.0

            # Validation loop
            for batch_num, data in enumerate(tqdm(self.dataloader_val), 0):

                # create batch and move to GPU
                fronts_in = data["fronts"]
                lefts_in = data["lefts"]
                rights_in = data["rights"]
                rears_in = data["rears"]
                lidars_in = data["lidars"]
                fronts = []
                lefts = []
                rights = []
                rears = []
                lidars = []
                for i in range(self.config.seq_len):
                    fronts.append(fronts_in[i].to(args.device, dtype=torch.float32))
                    if not self.config.ignore_sides:
                        lefts.append(lefts_in[i].to(args.device, dtype=torch.float32))
                        rights.append(rights_in[i].to(args.device, dtype=torch.float32))
                    if not self.config.ignore_rear:
                        rears.append(rears_in[i].to(args.device, dtype=torch.float32))
                    lidars.append(lidars_in[i].to(args.device, dtype=torch.float32))

                # driving labels
                command = data["command"].to(args.device)
                gt_velocity = data["velocity"].to(args.device, dtype=torch.float32)
                gt_steer = data["steer"].to(args.device, dtype=torch.float32)
                gt_throttle = data["throttle"].to(args.device, dtype=torch.float32)
                gt_brake = data["brake"].to(args.device, dtype=torch.float32)

                # target point
                target_point = torch.stack(data["target_point"], dim=1).to(
                    args.device, dtype=torch.float32
                )

                (
                    pred_wp,
                    image_to_lidar_prediction,
                    lidar_to_image_prediction,
                    next_frame_lidar_prediction,
                    next_frame_image_prediction,
                ) = self.model(
                    fronts + lefts + rights + rears, lidars, target_point, gt_velocity
                )

                gt_waypoints = [
                    torch.stack(data["waypoints"][i], dim=1).to(
                        args.device, dtype=torch.float32
                    )
                    for i in range(self.config.seq_len, len(data["waypoints"]))
                ]
                gt_waypoints = torch.stack(gt_waypoints, dim=1).to(
                    args.device, dtype=torch.float32
                )
                logging.debug("Calculating loss")
                logging.debug("gt_waypoints.shape: {}".format(gt_waypoints.shape))
                logging.debug("pred_wp.shape: {}".format(pred_wp.shape))
                wp_epoch += float(
                    F.l1_loss(pred_wp, gt_waypoints, reduction="none").mean()
                )

                num_batches += 1

            wp_loss = wp_epoch / float(num_batches)
            tqdm.write(
                f"Epoch {self.cur_epoch:03d}, Batch {batch_num:03d}:"
                + f" Wp: {wp_loss:3.3f}"
            )

            self.writer.add_scalar("val_loss", wp_loss, self.cur_epoch)
            wandb.log({"val_loss": wp_loss}, step=self.cur_iter)

            self.val_loss.append(wp_loss)

    def save(self):

        save_best = False
        if self.val_loss[-1] <= self.bestval:
            self.bestval = self.val_loss[-1]
            self.bestval_epoch = self.cur_epoch
            save_best = True

        # Create a dictionary of all data to save
        log_table = {
            "epoch": self.cur_epoch,
            "iter": self.cur_iter,
            "bestval": self.bestval,
            "bestval_epoch": self.bestval_epoch,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
        }

        # Save ckpt for every epoch
        torch.save(
            self.model.state_dict(),
            os.path.join(args.logdir, "model_%d.pth" % self.cur_epoch),
        )

        # Save the recent model/optimizer states
        torch.save(self.model.state_dict(), os.path.join(args.logdir, "model.pth"))
        torch.save(
            self.optimizer.state_dict(), os.path.join(args.logdir, "recent_optim.pth")
        )

        # Log other data corresponding to the recent model
        with open(os.path.join(args.logdir, "recent.log"), "w") as f:
            f.write(json.dumps(log_table))

        tqdm.write("====== Saved recent model ======>")

        if save_best:
            torch.save(
                self.model.state_dict(), os.path.join(args.logdir, "best_model.pth")
            )
            torch.save(
                self.optimizer.state_dict(), os.path.join(args.logdir, "best_optim.pth")
            )
            tqdm.write("====== Overwrote best model ======>")


# Data
def load_data(config, args):
    train_set = CARLA_Data(root=config.train_data, config=config)
    val_set = CARLA_Data(root=config.val_data, config=config)

    if args.debug:
        logging.info("Debug mode: subsampling data")
        train_set = Subset(train_set, range(100))
        val_set = Subset(val_set, range(100))

    logging.info("Creating train dataloader")
    dataloader_train = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )

    logging.info("Creating val dataloader")
    dataloader_val = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    return dataloader_train, dataloader_val


def load_model(config, args, dataloader_train, dataloader_val, writer):
    # Model
    logging.info("Initializing model")
    logging.info("Args.resume: {}".format(args.resume))
    model = TransFuser(config, args.device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    trainer = Engine(model, optimizer, config, dataloader_train, dataloader_val, writer)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    logging.info(f"Total trainable parameters: {str(params)}")

    # Create logdir
    if not os.path.isdir(args.logdir):
        if args.resume:
            raise ValueError("Cannot resume training, logdir does not exist")
        os.makedirs(args.logdir)
        logging.info("Created dir:", args.logdir)

    elif os.path.isfile(os.path.join(args.logdir, "recent.log")):
        if args.resume:
            logging.info("Loading checkpoint from " + args.logdir)
            with open(os.path.join(args.logdir, "recent.log"), "r") as f:
                log_table = json.load(f)

            # Load variables
            trainer.cur_epoch = log_table["epoch"]
            if "iter" in log_table:
                trainer.cur_iter = log_table["iter"]

            trainer.bestval = log_table["bestval"]
            trainer.train_loss = log_table["train_loss"]
            trainer.val_loss = log_table["val_loss"]

            # Load checkpoint
            model.load_state_dict(torch.load(os.path.join(args.logdir, "model.pth")))
            optimizer.load_state_dict(
                torch.load(os.path.join(args.logdir, "recent_optim.pth"))
            )
        else:
            # remove old logdir
            logging.info("Removing old logdir")
            os.system("rm -rf " + args.logdir)
            os.makedirs(args.logdir)
            logging.info("Created dir:", args.logdir)

    return model, optimizer, trainer


def log_args(args):
    with open(os.path.join(args.logdir, "args.txt"), "w") as f:
        logging.info("Save args to " + str(os.path.join(args.logdir, "args.txt")))
        json.dump(args.__dict__, f, indent=2)


def run_training(args):
    torch.cuda.empty_cache()
    writer = SummaryWriter(log_dir=args.logdir)

    config = GlobalConfig()

    logging.info(f"config.root_dir: {config.root_dir}")
    if not Path(config.root_dir).is_dir():
        raise ValueError(f"{config.root_dir} is not a valid directory")

    wandb.init(project="transfuser", config=get_config_dict(config))
    wandb.config.update(args.__dict__)

    logging.info(f"config.train_data: {config.train_data}")
    logging.info(f"config.val_data: {config.val_data}")

    dataloader_train, dataloader_val = load_data(config, args)
    model, optimizer, trainer = load_model(
        config, args, dataloader_train, dataloader_val, writer
    )
    log_args(args)

    wandb.config.update({"len_train_data": len(dataloader_train)})
    wandb.config.update({"len_val_data": len(dataloader_val)})

    logging.info("Start training")
    for epoch in range(trainer.cur_epoch, args.epochs):
        trainer.train()


def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--id", type=str, default="transfuser", help="Unique experiment identifier."
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument(
        "--epochs", type=int, default=101, help="Number of train epochs."
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument(
        "--val_every", type=int, default=5, help="Validation frequency (epochs)."
    )
    parser.add_argument("--batch_size", type=int, default=24, help="Batch size")
    parser.add_argument(
        "--logdir", type=str, default="log", help="Directory to log data to."
    )

    parser.add_argument(
        "--cross_modal_prediction_loss_coef",
        type=float,
        default=1,
        help="Coefficient of the image-to-lidar-prediction loss and vice versa.",
    )
    parser.add_argument(
        "--next_frame_prediction_loss_coef",
        type=float,
        default=1,
        help="Coefficient of the next frame prediction loss.",
    )
    parser.add_argument(
        "--image_loss_coef",
        type=float,
        default=0.1,
        help="Coefficient to scale image losses",
    )
    parser.add_argument(
        "--sweep_id",
        type=str,
        default=None,
        help="Used to group runs from the same sweep",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Debug mode (subsamples data for development)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume from checkpoint",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=2000,
        help="Maximum number of gradient updates for training",
    )
    parser.set_defaults(debug=False)

    args = parser.parse_args()
    args.logdir = os.path.join(args.logdir, args.id)
    args.config_str = f"nf_{args.next_frame_prediction_loss_coef}_cm_{args.cross_modal_prediction_loss_coef}_img_{args.image_loss_coef}"
    return args


if __name__ == "__main__":
    args = setup_parser()
    run_training(args)
