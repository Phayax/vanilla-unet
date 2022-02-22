import logging
import math
from pathlib import Path
from sys import stdout as sys_stdout
from typing import Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from unet import UNET
from utils import create_run_log_dir, PathLike
from datasets import KvasirSEGDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_result_images(model: torch.nn.Module, test_dataloader: DataLoader, tb_logger: SummaryWriter):
    logging.info("Evaluating with best test model")
    # TODO: somehow derive the length from the dataloader
    length = 20
    model.eval()
    with torch.no_grad():
        # values = np.zeros(shape=(length, 3, 512, 1536))
        # counter = 0
        idx_counter = 1
        for data in test_dataloader:
            images, targets = data
            images = images.to(device)
            targets = targets.to(device)
            preds = model(images)
            preds = (torch.sigmoid(preds) > 0.5).float()
            for image, target, pred in zip(images, targets, preds):
                plt.rcParams.update({
                    "figure.facecolor": (0.2, 0.2, 0.2, 1.0),  # figure background
                    "axes.facecolor": (0.2, 0.2, 0.2, 1.0),  # actual axis background
                    "savefig.facecolor": (0.2, 0.2, 0.2, 1.0),  # figure background used by savefig
                })
                fig, ax = plt.subplots(1, 3, figsize=(9, 3))
                im = image.cpu().detach().numpy()
                im_pred = pred[0].cpu().detach().numpy()
                im_target = target.cpu().numpy()
                ax[0].imshow(np.transpose(im, (1, 2, 0)))
                ax[1].imshow(im_pred.squeeze(), cmap='gray')
                ax[2].imshow(im_target.squeeze(), cmap='gray')
                ax[0].set_axis_off()
                ax[1].set_axis_off()
                ax[2].set_axis_off()
                fig.tight_layout()

                #plt.show()
                tb_logger.add_figure(f'seg/test-{idx_counter}', fig)
                idx_counter += 1
                # values[counter, :, :, 0:512] = im
                # values[counter, :, :, 512:1024] = im_pred
                # values[counter, :, :, 1024:] = im_target
                # tb_logger.add_images('seg/test', values)
                # counter += 1


def evaluate(model: torch.nn.Module, test_dataloader: DataLoader) -> float:
    # necessary for models with batchnorm

    logging.info("Evaluating on test data")
    batch_losses = []
    model.eval()

    with torch.no_grad():
        for data in test_dataloader:
            images, targets = data
            images = images.to(device)
            targets = targets.to(device)
            pred = model(images)
            loss = BCEWithLogitsLoss()(pred, targets)
            batch_losses.append(loss.item())
    model.train()
    val_loss = sum(batch_losses) / len(batch_losses)
    logging.info(f"Validation Loss: {val_loss:.6f}")
    return val_loss


def train(train_dataloader: DataLoader, test_dataloader: DataLoader, target_path: Path):

    learning_rate = 1e-5
    batch_size = train_dataloader.batch_size
    epochs = 200
    model = UNET(in_channels=3)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # loss_fn = BCELoss()
    # loss_fn = CrossEntropyLoss()
    loss_fn = BCEWithLogitsLoss()
    tb_logger = SummaryWriter(log_dir=str(target_path / "log"))

    best_test_model: Tuple[float, PathLike] = float('inf'), ""

    losses = []
    epoch_losses = []
    val_losses = []
    (target_path / "checkpoints").mkdir(exist_ok=True, parents=False)
    for epoch in range(epochs):
        batch_losses = []
        logging.info(f"Epoch {epoch:>2d}")
        for batch_no, (images, targets) in enumerate(train_dataloader):

            images = images.to(device)
            targets = targets.to(device)

            # logging.debug(f"{batch_no:>3d}: Forward Pass...  ")
            pred = model(images)
            # logging.debug("Done")
            # logging.debug("Loss...  ")

            loss = loss_fn(pred, targets)
            loss.backward()
            # plt.imshow(np.transpose(pred[0].detach().numpy() - target[0].detach().numpy(), (1, 2, 0))*255)
            # plt.show()
            # logging.debug("Done")
            # logging.debug("Backprop...  ")
            optimizer.step()
            optimizer.zero_grad()
            # logging.debug("Done")
            losses.append(loss.item())
            batch_losses.append(loss.item())
            logging.info(f"Epoch {epoch:>2d}    Batch {batch_no*batch_size:>3d}/{len(train_dataloader.dataset):>3d} -> Loss: {loss.item()}")

        epoch_loss = sum(batch_losses) / len(batch_losses)
        epoch_losses.append(epoch_loss)
        val_loss = evaluate(model, test_dataloader)
        val_losses.append(val_loss)
        tb_logger.add_scalar("loss/train", epoch_loss, global_step=epoch)
        tb_logger.add_scalar("loss/test", val_loss, global_step=epoch)

        # save the model every epoch
        model_save_path = target_path / f"checkpoints/model_ep_{epoch:03d}.pth"
        torch.save(model, model_save_path)
        # check if the model is the new best on the test dataset
        if val_loss < best_test_model[0]:
            best_test_model = val_loss, model_save_path

    torch.save(model, target_path / "model.pth")

    # after we have finished save images of the test_dataset
    best_model = torch.load(best_test_model[1])
    val_loss = evaluate(best_model, test_dataloader)
    tb_logger.add_hparams(
        {
            'lr': learning_rate,
            'batch_size': batch_size,
            'epochs': epochs,
            'loss': str(loss_fn),
        },
        {
            'hparam/test_loss': val_loss,
        },
        run_name=target_path.resolve().stem  # if path is not resolved a Path of './' will return nothing as stem
    )
    create_result_images(best_model, test_dataloader, tb_logger=tb_logger)


if __name__ == '__main__':
    torch.set_num_threads(6)
    target_path = create_run_log_dir("logging+high epoch test")
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys_stdout), logging.FileHandler(str(target_path / "train.log"))])

    ds = KvasirSEGDataset(
        images_path="./Kvasir-SEG/images",
        masks_path="./Kvasir-SEG/masks",
        # images_path="/home/sam/Desktop/Kvasir-SEG/images",
        # masks_path="/home/sam/Desktop/Kvasir-SEG/masks",
        target_size=(512, 512),
        keep_aspect_ratio=True,
        preload=True
    )
    train_ds, test_ds = torch.utils.data.random_split(dataset=ds, lengths=[800, 200])
    train_dl = DataLoader(dataset=train_ds, batch_size=12, shuffle=True, num_workers=0)
    test_dl = DataLoader(dataset=test_ds, batch_size=12, shuffle=True, num_workers=0)

    train(train_dl, test_dl, target_path=target_path)
