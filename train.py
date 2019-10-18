from datetime import datetime
import os
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from torch import nn, optim
from torch.utils.data import DataLoader
from dataset import THODataset
from data_process import *
from networks.baseline import Baseline3D
from networks.baseline2d_lstm import Baseline2D_LSTM
from config import Config
torch.backends.cudnn.benchmark = True

torch.cuda.manual_seed(43)
config = Config()



date = datetime.now().strftime('%m-%d-%H')
if not os.path.exists('models'):
    os.mkdir('models')
model_dir = os.path.join('models',config.model_name)
save_dir = os.path.join(model_dir,date)
save_mode_lame = config.model_name + '-' + config.dataset

if not os.path.exists(model_dir):
    os.mkdir(model_dir)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
writer = SummaryWriter(log_dir=save_dir)

global_min_error = 1e5 


def train_model(save_dir=save_dir, lr=config.lr,num_epochs=config.epoch_num, save_epoch=config.save_freq, test_interval=config.test_freq):

    #prepare model
    model = Baseline2D_LSTM()
    model = model.cuda()
    model = model.to(config.device_ids[0])
    if len(config.device_ids) > 1:
        if torch.cuda.device_count() >= len(config.device_ids):
            model = nn.DataParallel(model, device_ids=config.device_ids)
        else:
            # raise ValueError("the machine don't have {} gpus".format(str(len(config.device_ids))))
            pass
    
    #loss
    criterion = nn.MSELoss()

    #choose optimizer
    if config.set_optim=='Adam':
        optimizer = optim.Adam(model.parameters(),lr = lr,weight_decay=config.weight_decay)
    elif config.set_optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.milestones, gamma=0.1)

    #prepare data
    train_set = THODataset(config.data_root, config.label_path, config.dataset, split='train')
    train_loader = DataLoader(train_set, config.batch_size, shuffle=True, num_workers=config.num_workers)
    test_set = THODataset(config.data_root, config.label_path, config.dataset, split='test')
    test_loader = DataLoader(test_set, config.test_batch_size, shuffle=False, num_workers=config.num_workers)

    #load pretrained model if needed
    if config.resume_epoch_num != 0:
        checkpoint = torch.load(config.resume_model_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['opt_dict'])


    # start trainning
    for epoch in range(config.resume_epoch_num, num_epochs):
        
        print("current lr is :",optimizer.param_groups[0]['lr'])
        model.train()
        ############# trainning ###############
        train_loss = 0.0
        train_mae_a = 0.0
        train_mae_s = 0.0
        i = 0
        for inputs, labels in tqdm(train_loader):
            inputs = inputs.cuda()
            labels = labels.cuda()
            optimizer.zero_grad()
            model.lstm_a.reset_hidden_state()
            model.lstm_s.reset_hidden_state()
            i += 1
            outputs_a,outputs_s = model(inputs)
            loss_a = criterion(outputs_a, labels[:,0])
            loss_s = criterion(outputs_s, labels[:,1])
            loss = loss_a+loss_s
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*inputs.size(0)
            train_mae_a+=  (labels[:,0]-outputs_a).abs().sum()
            train_mae_s+=  (labels[:,1]-outputs_s).abs().sum()
            if i % 20 == 0:
                print('training loss: %3f,mae_angle:%3f,mae_speed:%3f' % 
                (train_loss / i /config.batch_size,train_mae_a/ i /config.batch_size,train_mae_s/ i /config.batch_size))

        train_loss = train_loss / len(train_loader)/config.batch_size
        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_mae_speed', train_mae_s, epoch)
        writer.add_scalar('train_mae_angle', train_mae_a, epoch)
        print("[train] Epoch: {}/{} Loss: {}".format(epoch, config.epoch_num, train_loss))

        scheduler.step(epoch)

        ############# testing ###############
        if epoch % test_interval == (test_interval - 1):
            model.eval()
            test_loss = 0.0
            test_mae_a = 0.0
            test_mae_s = 0.0
            gt_s = []
            gt_a = []
            predict_s = []
            predict_a = []
            for inputs, labels in tqdm(test_loader):
                inputs = inputs.cuda()
                labels = labels.cuda()
                with torch.no_grad():
                    model.lstm_a.reset_hidden_state()
                    model.lstm_s.reset_hidden_state()
                    outputs_a, outputs_s = model(inputs)
                    loss_a = criterion(outputs_a, labels[:,  0])
                    loss_s = criterion(outputs_s, labels[:,  1])
                    loss = loss_a + loss_s
                    gt_a+= labels[:,0].cpu().numpy().tolist()
                    gt_s+= labels[:,1].cpu().numpy().tolist()
                    predict_a+= outputs_a.cpu().numpy().tolist()
                    predict_s+= outputs_s.cpu().numpy().tolist()
                    test_mae_a+=  (labels[:,0]-outputs_a).abs().sum()
                    test_mae_s+=  (labels[:,1]-outputs_s).abs().sum()
                    test_loss += loss.item() * inputs.size(0)

            #save fig of speed and angle at models/data/ epoch_speed.png & epoch_angle.png
            
            timeline = list(range(len(gt_a)))
            plot_line(timeline,gt_s,predict_s,label="speed_gt",label1="speed_pred",path = os.path.join(save_dir,str(epoch)+"_speed.png"))
            plot_line(timeline,gt_a,predict_a,label="angle_gt",label1="angle_pred",path = os.path.join(save_dir,str(epoch)+"_angle.png"))

            test_loss = test_loss / len(test_loader)/config.batch_size
            test_mae_s = test_mae_s / len(test_loader)/config.batch_size
            test_mae_a = test_mae_a / len(test_loader)/config.batch_size
            writer.add_scalar('test_loss', test_loss, epoch)
            writer.add_scalar('test_mae_speed', test_mae_s, epoch)
            writer.add_scalar('test_mae_angle', test_mae_a, epoch)
            print('test loss: %3f,angle_mae:%3f,speed_mae:%3f' % 
                (test_loss,test_mae_a,test_mae_s))

            if global_min_error>test_mae_a+test_mae_s:
                global_min_error = test_mae_a+test_mae_s


    writer.close()


if __name__ == "__main__":
    train_model()
