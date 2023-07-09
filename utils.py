import pandas as pd
import matplotlib.pyplot as plt
import string, random, csv
from cleanfid import fid
import os

from torchvision import datasets
import torchvision.transforms as transforms
import torch, torchvision
import torchvision.utils as vutils
import numpy as np

# Generate random id
def random_id(size, chars=string.ascii_uppercase + string.digits):
    return "".join(random.choice(chars) for x in range(size))

# Generate plots for loss, scores, fid
def generate_plot(data_summary, graphic, color_x, color_y, img_folder, epoch_end=-1):
    fig, ax = plt.subplots(figsize=(10, 5))
    if graphic == "loss":
        title = "Pérdida"
        plt.plot(
            data_summary["g"][:epoch_end], label="Generador", color=color_x, marker="s", ms=2
        )
        plt.plot(
            data_summary["d"][:epoch_end], label="Discriminador", color=color_y, marker="s", ms=2
        )
        plt.legend()
    elif graphic == "scores":
        title = "Precisión del discriminador"
        plt.plot(
            data_summary["real"][:epoch_end] * 100,
            label="IMG Real",
            color=color_x,
            marker="s",
            ms=2,
        )
        plt.plot(
            data_summary["fake"][:epoch_end] * 100,
            label="IMG Falsa",
            color=color_y,
            marker="s",
            ms=2,
        )
        # plt.plot(data_summary["fake2"][:]*100, label="Falsa2", color=color_x, marker='s', ms=2)
        plt.legend()
    elif graphic == "fid-clean":
        title = "FID"
        plt.plot(data_summary["fid-clean"][:epoch_end].dropna(), label="Clean", marker="o", color=color_x, ms=5)
        #plt.plot(data_summary["fid-tensor"][:].dropna(), label="Tensor", marker="o", color=color_y, ms=5)

    # ax.set_title(title)
    ax.set_xlabel("Épocas")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(color="gray", linestyle="-", linewidth=0.5, alpha=0.5)
    plt.savefig(img_folder + graphic + ".jpg", bbox_inches="tight")
    plt.show()
    plt.close()
    print()

# Create numpy custom stats for FID
def create_stats(val, folder1, device, mode, batch_size):
    if not fid.test_stats_exists(val, mode=mode):
        fid.make_custom_stats(val, folder1, mode=mode, device=device, batch_size=batch_size)
        print("Datos para %s creados." % mode)
    else:
        print("Datos para %s disponibles." % mode)

# Get dataloader
def get_dataset(data_path, image_size, batch_size, workers):
    transform = transforms.Compose(
        [
            transforms.Resize([image_size, image_size]),
            transforms.ToTensor(),
            transforms.Grayscale(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    datareference = datasets.ImageFolder(root=data_path, transform=transform)
    datatrain = torch.utils.data.DataLoader(
        datareference,
        batch_size=batch_size,
        num_workers=workers,
        shuffle=True,
        pin_memory=True,
    )
    train_size = int(len(datareference))
    print("Train size: " + str(train_size))
    return datatrain

# Generate csv with data
def create_data_summary(img_folder, summary):
    with open(img_folder + "data_summary.csv", "w") as w:
        writer = csv.writer(w)
        writer.writerow(("d", "g", "real", "fake", "fake2", "fid-clean", "fid-tensor"))
        writer.writerows(summary)

# Save training information
def create_savelog(type_net, img_folder, name_coded, parm_list):
    file = open(img_folder + "%s.txt" % name_coded, "w")
    parameters = (
        "beta1: %f, beta2: %f\nbatch_size: %d\nlr: %f\nepochs: %d\nlatent_dimen: %d\nimg_size: %d\ngenerator model\n%s \ndiscriminator model\n%s"
        % (
            parm_list[0],
            parm_list[1],
            parm_list[2],
            parm_list[3],
            parm_list[4],
            parm_list[5],
            parm_list[6],
            parm_list[7],
            parm_list[8],
        )
    )
    if type_net == "sngan":
        parameters += str(
            "disc_iters: %d" % (disc_iters)
        )
    file.write(parameters)
    file.close()
    print("Parametros guardados correctamente.")

# Get values of epochs
def check_epochs_values(epoch, num_epochs, errD, errG, D_x, D_G_z1, D_G_z2, score_clean, score_tensor):
    s = "%d|%d\tLOSS(D):%.4f / LOSS(G):%.4f || D(x):%.4f / D(G(z)):%.4f/ %.4f" % (
        epoch,
        num_epochs,
        errD,
        errG,
        D_x,
        D_G_z1,
        D_G_z2,
    )
    if score_clean is not None:
        s += str(" || FID-C: %.4f" % (score_clean))
    if score_tensor is not None:
        s += str(" || FID-T: %.4f" % (score_tensor))
        s += str("\n________________________________________________________________________________________________")
    return s

# Show and make grid of 64 elements
def show_grid_64(data, device, title, save_fig=False, path_save=None):
    plt.figure(figsize=(12,12))
    plt.axis("off")
    plt.title(title)
    grid = np.transpose(vutils.make_grid(data.to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0))
    plt.imshow(grid)
    if save_fig: 
      plt.savefig(path_save, bbox_inches="tight")
      #plt.show(block=False)
      plt.close()
      return grid
    else:
      plt.show()
      plt.close()

# Saving model
def save_checkpoint(states, checkpoint_dir, filename='checkpoint.pth'):
    torch.save(states, os.path.join(checkpoint_dir, filename))