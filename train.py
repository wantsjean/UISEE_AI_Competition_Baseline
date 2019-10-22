from datetime import datetime
import os
# from tqdm import tqdm
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


date = datetime.now().strftime('%d-%H-%M')
if not os.path.exists('models'):
    os.mkdir('models')
model_dir = os.path.join('models', config.model_name)
save_dir = os.path.join(model_dir, date)
save_mode_lame = config.model_name + '-' + config.dataset

if not os.path.exists(model_dir):
    os.mkdir(model_dir)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
writer = SummaryWriter(log_dir=save_dir)


def train_model(save_dir=save_dir, lr=config.lr, num_epochs=config.epoch_num, save_epoch=config.save_freq, val_interval=config.val_freq):

    # prepare model
    model = Baseline2Path()
    # model = model.cuda()
    model = model.to(config.device_ids[0])
    if len(config.device_ids) > 1:
        if torch.cuda.device_count() >= len(config.device_ids):
            model = nn.DataParallel(model, device_ids=config.device_ids)
        else:
            # raise ValueError("the machine don't have {} gpus".format(str(len(config.device_ids))))
            pass

    # #loss
    criterion = nn.MSELoss()

    # choose optimizer
    if config.set_optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr,
                               weight_decay=config.weight_decay)
    elif config.set_optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr,
                              momentum=0.9, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=config.milestones, gamma=0.1)

    # prepare data

    data = np.load(os.path.join(data_root, 'train.npy'))
    train_set = THODataset(data, config.data_root,
                           config.label_path, config.dataset, split='train')
    train_loader = DataLoader(
        train_set, config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_set = THODataset(data, config.data_root,
                         config.label_path, config.dataset, split='val')
    val_loader = DataLoader(val_set, config.val_batch_size,
                            shuffle=False, num_workers=config.num_workers)

    # load pretrained model if needed
    if config.resume_epoch_num != 0:
        checkpoint = torch.load(config.resume_model_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['opt_dict'])

    # start trainning
    global_min_error = 1e5

    for epoch in range(config.resume_epoch_num, num_epochs):

        print("current lr is :", optimizer.param_groups[0]['lr'])
        model.train()
        ############# trainning ###############
        train_loss = 0.0
        train_mae_a = 0.0
        train_mae_s = 0.0
        i = 1
        for inputs, labels in tqdm(train_loader):
            inputs = (inputs.cuda().permute(
                0, 1, 4, 2, 3).float() - 128)/128  # normalize
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
            if i % 20 == 0:
                print('training loss: %3f,mae_angle:%3f,mae_speed:%3f' %
                      (train_loss / i / config.batch_size, train_mae_a / i / config.batch_size, train_mae_s / i / config.batch_size))
            i += 1
        train_loss = train_loss / len(train_loader)/config.batch_size
        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_mae_speed', train_mae_s /
                          len(train_loader)/config.batch_size, epoch)
        writer.add_scalar('train_mae_angle', train_mae_a /
                          len(train_loader)/config.batch_size, epoch)
        print("[train] Epoch: {}/{} Loss: {}".format(epoch,
                                                     config.epoch_num, train_loss))

        scheduler.step(epoch)

        ############# testing ###############
        if epoch % val_interval == (val_interval - 1):
            model.eval()
            val_loss = 0.0
            val_mae_a = 0.0
            val_mae_s = 0.0
            gt_s = []
            gt_a = []
            predict_s = []
            predict_a = []
            for inputs, labels in tqdm(val_loader):
                inputs = (inputs.cuda().permute(
                    0, 1, 4, 2, 3).float() - 128)/128  # normalize
                labels = labels.cuda()
                with torch.no_grad():
                    model.path_a.lstm.reset_hidden_state()
                    model.path_s.lstm.reset_hidden_state()
                    outputs_a, outputs_s = model(inputs)
                    loss_a = criterion(outputs_a, labels[:,  0])
                    loss_s = criterion(outputs_s, labels[:,  1])
                    loss = loss_a + loss_s
                    gt_a += labels[:, 0].cpu().numpy().tolist()
                    gt_s += labels[:, 1].cpu().numpy().tolist()
                    predict_a += outputs_a.cpu().numpy().tolist()
                    predict_s += outputs_s.cpu().numpy().tolist()
                    val_mae_a += (labels[:, 0]-outputs_a).abs().sum()
                    val_mae_s += (labels[:, 1]-outputs_s).abs().sum()
                    val_loss += loss.item() * inputs.size(0)

            # save fig of speed and angle at models/data/ epoch_speed.png & epoch_angle.png

            timeline = list(range(len(gt_a)))
            plot_line(timeline, gt_s, predict_s, label="speed_gt", label1="speed_pred",
                      path=os.path.join(save_dir, str(epoch)+"_speed.png"))
            plot_line(timeline, gt_a, predict_a, label="angle_gt", label1="angle_pred",
                      path=os.path.join(save_dir, str(epoch)+"_angle.png"))

            val_loss = val_loss / len(val_loader)/config.val_batch_size
            val_mae_s = val_mae_s / len(val_loader)/config.val_batch_size
            val_mae_a = val_mae_a / len(val_loader)/config.val_batch_size
            writer.add_scalar('val_loss', val_loss, epoch)
            writer.add_scalar('val_mae_speed', val_mae_s, epoch)
            writer.add_scalar('val_mae_angle', val_mae_a, epoch)
            print('val loss: %3f,angle_mae:%3f,speed_mae:%3f' %
                  (val_loss, val_mae_a, val_mae_s))

            if global_min_error > val_mae_a+val_mae_s:
                global_min_error = val_mae_a+val_mae_s
                torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'opt_dict': optimizer.state_dict(),
                            }, os.path.join(save_dir, 'best.pth.tar'))
                print("Save model at {}\n".format(
                    os.path.join(save_dir, 'best.pth.tar')))

    writer.close()


if __name__ == "__main__":
    train_model()
