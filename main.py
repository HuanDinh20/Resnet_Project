import torch
from module import RestNet18, ResNet
from train import per_epoch_activity
import utils
from datetime import datetime
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

# 1. define some parameters
BATCH_SIZE = 16
EPOCH = 16
device = utils.get_device()
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
root = r"A:\huan_shit\Study_Shit\Deep_Learning\Side_Projects\LeNet_project\data"
model_path = r"A:\huan_shit\Study_Shit\Deep_Learning\Side_Projects\Residual_Networks\saved_model\full_model" \
             r"\RestNet18_01.pth "
summary_writer = SummaryWriter()

# 2. dataset
train_aug, val_aug = utils.image_augmentation()
train_set, val_set = utils.get_FashionMNIST(train_aug, val_aug, root)
train_loader, val_loader = utils.create_dataloader(train_set, val_set, BATCH_SIZE)

# 3.
model = ResNet((64, 64, 128, 128, 256, 256, 512, 512), 10)
model.to(device)

# 4 .losss
loss_fn = nn.CrossEntropyLoss()

# 6. optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

if __name__ == "__main__":
    per_epoch_activity(train_loader, val_loader, device, optimizer, model, loss_fn,
                       summary_writer, EPOCH, timestamp)
    torch.save(model, model_path)