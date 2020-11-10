class match_config:
    def __init__(self):
        self.delf_saved_path = '/home/workspace/chencheng/Learning/ImageRetrieval/delf-pytorch/repository/delf/index.delf'
        self.query_img_path = '/home/workspace/chencheng/Learning/ImageRetrieval/delf-pytorch/datasets/test_query/test4.jpg'
        self.indx_img_root = '/home/workspace/chencheng/Learning/ImageRetrieval/delf-pytorch/datasets/my_index/index_images/'
        self.saved_path = '/home/workspace/chencheng/Learning/ImageRetrieval/delf-pytorch/results/'
        self.is_visulize = True
        
        # match config
        self.pca_dims = 40
        self.topk_features = 500
        self.topk_retrival_imgs = 3


        # visulize config
        
        