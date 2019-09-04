import argparse, os
import torch
import random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model_tsr_12 import model_tsr
from dataset_tsr import DatasetFromHdf5
import time, math
import numpy as np
from torchsummary import summary
from tensorboardX import SummaryWriter

# Training settings
parser = argparse.ArgumentParser(description="PyTorch Temporal Super Resolution for Dynamic Real Dataset")
parser.add_argument("--batchSize", type=int, default=1, help="Training batch size. Default: 16")
parser.add_argument("--nEpochs", type=int, default=100, help="Number of epochs to train for. Default: 100")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--step", type=int, default=2,
                    help="Halves the learning rate for every n epochs. Default: n=3")
parser.add_argument("--cuda", action="store_true", help="Use cuda? Default: True")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint for resume. Default: None")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts). Default: 1")
parser.add_argument("--threads", type=int, default=0, help="Number of threads for data loader to use, Default: 0")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="weight decay, Default: 1e-4")
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained model. Default: None")
parser.add_argument("--train_data0", default="./tsr_train_data0.h5", type=str, help="training set path.")
parser.add_argument("--train_data1", default="./tsr_train_data1.h5", type=str, help="training set path.")
parser.add_argument("--train_label", default="./tsr_train_label.h5", type=str, help="training set path.")
parser.add_argument("--valid_data0", default="./tsr_val_data0.h5", type=str, help="validation set path.")
parser.add_argument("--valid_data1", default="./tsr_val_data1.h5", type=str, help="validation set path.")
parser.add_argument("--valid_label", default="./tsr_val_label.h5", type=str, help="validation set path.")
parser.add_argument("--gpu", default='0', help="GPU number to use when training. ex) 0,1 Default: 0")
parser.add_argument("--checkpoint", default="./checkpoint", type=str,
                    help="Checkpoint path. Default: ./checkpoint ")


def main():
    global opt, model, board
    opt = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

    print(opt)

    cuda = opt.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True

    print("===> Loading datasets")
    train_path = []
    train_path.append(opt.train_data0)
    train_path.append(opt.train_data1)
    train_path.append(opt.train_label)
    train_set = DatasetFromHdf5(train_path)

    valid_path = []
    valid_path.append(opt.valid_data0)
    valid_path.append(opt.valid_data1)
    valid_path.append(opt.valid_label)
    valid_set = DatasetFromHdf5(valid_path)

    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize,
                                      shuffle=True)
    validation_data_loader = DataLoader(dataset=valid_set, num_workers=opt.threads, batch_size=opt.batchSize,
                                        shuffle=True)

    print("===> Building model")
    model = model_tsr()

    L1_loss = nn.L1Loss()
    #L2_loss = nn.MSELoss()

    print("===> Setting GPU")
    if cuda:
        model = nn.DataParallel(model).cuda()
        L1_loss = L1_loss.cuda()

    summary(model, [(3, 640, 720), (3, 640, 720)])

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["model"].state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    # optionally copy weights from a checkpoint
    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("=> loading model '{}'".format(opt.pretrained))
            weights = torch.load(opt.pretrained)
            model.load_state_dict(weights['model'].state_dict())
        else:
            print("=> no model found at '{}'".format(opt.pretrained))

    print("===> Setting Optimizer")
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    # optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay, nesterov=True)

    board = SummaryWriter()

    print("===> Training")
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        train(training_data_loader, optimizer, model, L1_loss, epoch, validation_data_loader)


def adjust_learning_rate(epoch):
    lr = opt.lr
    for i in range(epoch // opt.step):
        lr = lr / 2
    return lr


def train(training_data_loader, optimizer, model, L1_loss, epoch, validation_data_loader):
    lr = adjust_learning_rate(epoch - 1)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    print("epoch =", epoch, "lr =", optimizer.param_groups[0]["lr"])

    model.train()

    train_psnr = 0
    loss_sum = 0
    st = time.time()

    for iteration, batch in enumerate(training_data_loader, 1):

        input0, input1, label = Variable(batch[0]/255.), Variable(batch[1]/255.), Variable(batch[2]/255.)
        [num_bat, num_c, patch_h, patch_w] = input0.shape
        input0 = input0.numpy()
        input1 = input1.numpy()
        label = label.numpy()

        a = np.random.randint(4, size=1)[0]
        if a % 2 == 0:
            for i in range(num_bat):
                for j in range(num_c):
                    input0[i, j, :, :] = np.rot90(input0[i, j, :, :], a).copy()
                    input1[i, j, :, :] = np.rot90(input1[i, j, :, :], a).copy()
                    label[i, j, :, :] = np.rot90(label[i, j, :, :], a).copy()
        else:
            temp0 = np.zeros((num_bat, num_c, patch_w, patch_h))
            temp1 = np.zeros((num_bat, num_c, patch_w, patch_h))
            temp2 = np.zeros((num_bat, num_c, patch_w, patch_h))
            [num_bat, num_c, patch_h, patch_w] = temp0.shape
            for i in range(num_bat):
                for j in range(num_c):
                    temp0[i, j, :, :] = np.rot90(input0[i, j, :, :], a).copy()
                    temp1[i, j, :, :] = np.rot90(input1[i, j, :, :], a).copy()
                    temp2[i, j, :, :] = np.rot90(label[i, j, :, :], a).copy()

            del input0
            del input1
            del label

            input0 = temp0
            input1 = temp1
            label = temp2

            del temp0
            del temp1
            del temp2

        if np.random.randint(2, size=1)[0] == 1:
            for i in range(num_bat):
                for j in range(num_c):
                    input0[i, j, :, :] = np.flip(input0[i, j, :, :], axis=1).copy()
                    input1[i, j, :, :] = np.flip(input1[i, j, :, :], axis=1).copy()
                    label[i, j, :, :] = np.flip(label[i, j, :, :], axis=1).copy()

        if np.random.randint(2, size=1)[0] == 1:
            for i in range(num_bat):
                for j in range(num_c):
                    input0[i, j, :, :] = np.flip(input0[i, j, :, :], axis=0).copy()
                    input1[i, j, :, :] = np.flip(input1[i, j, :, :], axis=0).copy()
                    label[i, j, :, :] = np.flip(label[i, j, :, :], axis=0).copy()

        input0 = Variable(torch.from_numpy(input0).float()).view(num_bat, num_c, patch_h, patch_w)
        input1 = Variable(torch.from_numpy(input1).float()).view(num_bat, num_c, patch_h, patch_w)
        label = Variable(torch.from_numpy(label).float()).view(num_bat, num_c, patch_h, patch_w)

        if opt.cuda:
            input0 = input0.cuda()
            input1 = input1.cuda()
            label = label.cuda()

        output = model(input0, input1)

        loss = L1_loss(output, label)

        train_psnr += output_psnr_mse(label.cpu().detach().numpy(), output.cpu().detach().numpy())

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        loss_sum += loss.item()

        if iteration % int(len(training_data_loader)/10.) == 0:

            model.eval()

            val_psnr = 0

            for it, batch in enumerate(validation_data_loader, 1):

                input0, input1, label = Variable(batch[0]/255.), Variable(batch[1]/255.), Variable(batch[2]/255.)

                if opt.cuda:
                    input0 = input0.cuda()
                    input1 = input1.cuda()

                with torch.no_grad():
                    val_out = model(input0, input1)

                val_out = val_out.cpu().data[0].numpy()
                label = label.data[0].numpy()

                val_psnr += output_psnr_mse(label, val_out)

            val_psnr /= len(validation_data_loader)

            avg_loss = loss_sum / iteration
            print(
                "===> Epoch[{}]({}/{}): Train_Loss: {:.10f} Val_PSNR: {:.4f} Train_PSNR: {:.4f}".format(
                                        epoch, iteration, len(training_data_loader), avg_loss, val_psnr, train_psnr / iteration))
            board.add_scalar('avg_loss', avg_loss, iteration + len(training_data_loader) * (epoch - 1))
            board.add_scalar('val_PSNR', val_psnr, iteration + len(training_data_loader) * (epoch - 1))
            board.add_scalar('train_PSNR', train_psnr / iteration, iteration + len(training_data_loader) * (epoch - 1))

            model.train()

            save_checkpoint(model, epoch, iteration, train_psnr / iteration, val_psnr, avg_loss)

    print("training_time: ", time.time() - st)


def save_checkpoint(model, epoch, iteration, tpsnr, vpsnr, loss):
    model_folder = opt.checkpoint
    model_out_path = model_folder + "/model_epoch_{}_iter_{}_TPSNR1_{:.4f}_VPSNR_{:.4f}_loss_{:.8f}.pth".format(
                                                                            epoch, iteration, tpsnr, vpsnr, loss)
    state = {"epoch": epoch, "model": model}
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    torch.save(state, model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))


def output_psnr_mse(img_orig, img_out):
    squared_error = np.square(img_orig - img_out)
    mse = np.mean(squared_error)
    psnr = 10 * np.log10(1.0 / mse)
    return psnr


if __name__ == "__main__":
    main()
