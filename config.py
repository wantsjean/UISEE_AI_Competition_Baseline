class Config:
    def __init__(self):
        self.dataset = 'uisee'
        self.batch_size = 8
        self.test_batch_size = 8
        self.seq_len = 4
        self.data_root,self.label_path = self.get_path_by_dataset(self.dataset)
        self.set_optim = 'Adam'
        self.multiplier = 10
        self.weight_decay = 5e-4
        self.device_ids = [0,1,2,3]
        self.resume_epoch_num = 0
        self.resume_model_path = None
        self.model_name = 'base2d_lstm'
        self.epoch_num = 200
        self.save_freq = 1
        self.test_freq = 1
        self.lr = 1e-4
        self.num_workers = 4
        self.milestones = [50,100]

    def get_path_by_dataset(self,dataset):
        if dataset=='uisee':
            return 'data/uisee/','data/uisee/train/label.txt'