class config:
    def __init__(self):
        # train config
        self.stage = 'attention'
        self.ft_epochs = 30
        self.atte_epochs = 30
        self.lr = 1e-4
        self.optim = 'adam'
        self.batch_size = 12
        self.num_workers = 8
        self.ncls = 10000
        self.lr_decay = 0.5
        self.weight_decay = 0.0001
        self.lr_stepsize = 10

        # dataset config 
        self.train_data_root = '/home/workspace/chencheng/libc++/gcc-lib-so6.0/dataset/sub_dataset/train/'
        self.val_data_root = '/home/workspace/chencheng/libc++/gcc-lib-so6.0/dataset/sub_dataset/val/'

        # checkpoints save
        self.ckpt_root = '/home/workspace/chencheng/Learning/ImageRetrieval/delf-pytorch/repository/'
        self.log_root = '/home/workspace/chencheng/Learning/ImageRetrieval/delf-pytorch/logs/'
        
        # load from path 
        self.ft_ckpt = "/home/workspace/chencheng/Learning/ImageRetrieval/delf-pytorch/repository/finetune/epoch_10_loss_0.28494.pth"
