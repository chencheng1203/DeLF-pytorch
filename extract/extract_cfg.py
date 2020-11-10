class extract_config:
    def __init__(self):
        self.stage = "delf"
        self.index_img = '/home/workspace/chencheng/Learning/ImageRetrieval/delf-pytorch/datasets/my_index/'
        self.kp_path = "/home/workspace/chencheng/Learning/ImageRetrieval/delf-pytorch/repository/attention/epoch_26_loss_1.84812.pth"
        self.log_root = '/home/workspace/chencheng/Learning/ImageRetrieval/delf-pytorch/logs/'
        # extract config
        self.scales = [0.25, 0.5, 1.0, 1.4142, 2.0]
        self.iou_thres = 0.98
        self.atte_thres = 0.16
        self.topk = 1000
        self.delf_rf = 291.0
        self.delf_stride = 16.0
        self.delf_padding = 145.0

        # pca config
        self.pca_saved = '/home/workspace/chencheng/Learning/ImageRetrieval/delf-pytorch/repository/pca/'
        self.pca_dims = 40

        self.delf_saved = '/home/workspace/chencheng/Learning/ImageRetrieval/delf-pytorch/repository/delf/'
        
        
        