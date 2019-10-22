from datetime import datetime
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from torch import nn, optim
from torch.utils.data import DataLoader
from fast_dataset import THODataset
from data_process import *
from networks.baseline2d_lstm import Baseline2D_LSTM
from networks.baseline2path import Baseline2Path
from config import Config
torch.backends.cudnn.benchmark = True

torch.cuda.manual_seed(43)
config = Config()

def train(model, criterion):
    # loss
    # choose optimizer
    if config.set_optim == 'Adam':
        optimizer = optim.Adam(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    elif config.set_optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=config.lr,
                              momentum=0.9, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=config.milestones, gamma=0.1)

    # prepare data
    data = np.load(os.path.join(data_root, 'train.npy'))
    train_set = THODataset(
        data, config.data_root, config.label_path, config.dataset, split='submit_train')
    train_loader = DataLoader(
        train_set, config.batch_size, shuffle=True, num_workers=config.num_workers)

    # starting training
    for epoch in range(0, config.epoch_num):
        print("current lr is :", optimizer.param_groups[0]['lr'])
        model.train()
        ############# trainning ###############
        train_loss = 0.0
        train_mae_a = 0.0
        train_mae_s = 0.0
        i = 1
        for inputs, labels in tqdm(train_loader):
            inputs = (inputs.cuda().permute(0, 1, 4, 2, 3).float() - 128)/128  # normalize
            labels = labels.cuda()
            optimizer.zero_grad()
            model.path_a.lstm.reset_hidden_state()
            model.path_s.lstm.reset_hidden_state()
            outputs_a, outputs_s = model(inputs)
            loss_a = criterion(outputs_a, labels[:, 0])
            loss_s = criterion(outputs_s, labels[:, 1])
            loss = loss_a+loss_s
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*inputs.size(0)
            train_mae_a += (labels[:, 0]-outputs_a).abs().sum()
            train_mae_s += (labels[:, 1]-outputs_s).abs().sum()
            if i % 50 == 0:
                print('training loss: %3f,mae_angle:%3f,mae_speed:%3f' %
                      (train_loss / i / config.batch_size, train_mae_a / i / config.batch_size, train_mae_s / i / config.batch_size))
            i += 1
        train_loss = train_loss / len(train_loader) / config.batch_size
        print("[train] Epoch: {}/{} Loss: {}".format(epoch, config.epoch_num, train_loss))

        scheduler.step(epoch)


def eval(model, criterion):
    data = np.load(os.path.join(data_root, 'test.npy'))
    eval_set = THODataset(data, config.data_root, None,config.dataset, split='submit_eval')
    eval_loader = DataLoader(eval_set, config.batch_size,shuffle=False, num_workers=config.num_workers)

    model.eval()

    predict_s = []
    predict_a = []
    for inputs in tqdm(eval_loader):
        inputs = (inputs.cuda().permute(0, 1, 4, 2, 3).float() - 128)/128  # normalize
        with torch.no_grad():
            model.path_a.lstm.reset_hidden_state()
            model.path_s.lstm.reset_hidden_state()
            outputs_a, outputs_s = model(inputs)
            predict_a += outputs_a.cpu().numpy().tolist()
            predict_s += outputs_s.cpu().numpy().tolist()

    timeline = list(range(len(predict_a)))
    plot_line_single(timeline, predict_s, label="speed_pred",
                     path="submit_speed.png")
    plot_line_single(timeline, predict_a, label="angle_pred",
                     path="submit_angle.png")

    with open("submit.txt",'w') as f:
        res = []
        for i in range(len(timeline)):
            label = timeline[i]
            angle = predict_a[i]
            speed = predict_s[i]
            res.append("{:d} {:.6f} {:.6f}".format(label,angle,speed))
        f.write("\n".join(res))

def submit():
    model = Baseline2Path()
    model = model.cuda()

    criterion = nn.MSELoss()
    train(model,criterion)
    eval(model, criterion)

def show_result():
    label_path = "submit.txt"
    labels = get_labels(label_path,submit=True)
    img_paths = get_image_paths(config.data_root,split='test',ext = 'jpg')
    show_image(img_paths,labels,size=400,wait=50)

if __name__ == "__main__":
    #submit()
    show_result()