class Config:
    gpu = 1

    epochs = 60

    learn_rate = 0.01
    learn_rate_seg = 0.01
    learn_rate_dec = 0.01
    momentum = 0.9
    batch_size = 1
    train_part = "test"
    train_mode = "total"
    mode = "testing"
    dataset = "ksdd2"

    max_val = 3
    p = 2
    dilation = 15
    delta_cls_loss = 1

    is_seg_loss_weighted = False
    test_during_train = True
    WRITE_TENSORBOARD = False
    FREQUENCY_SAMPLING = True
    GRADIENT_ADJUSTMENT = True
    sava_imgs = True
    dyn_balance_loss = True

    test_frequency = 4
    save_frequency = 2
    vis_frequency = 15

    test_ratio = 1
    test_data_dir = "../ksdd2-test--"
    train_data_dir = "../ksdd2-train--"
    ksdd2_Dataset_dir = ""
    vis_dir = "./visualization/test"
    ckp_dir = "ckp"
    feature_dir = "feature_map"
    log_dir = "LOG"
    tensorboard_dir = ""
    ret_dir = "results"

    input_w = None
    input_h = None
    input_c = None

    def init_extra(self):
        if self.is_seg_loss_weighted and (self.p is None or self.max_val is None):
            raise Exception("p 和 max是None")

        if self.dataset == 'ksdd2':
            self.input_w = 240
            self.input_h = 640
            self.input_c = 3
        else:
            raise Exception("数据集不存在，名字打错了")

    def merge_from_args(self, args):
        self.gpu = args.gpu
        self.epochs = args.epochs
        if args.train_seg:
            self.train_mode = "seg"
        elif args.train_dec:
            self.train_mode = "dec"
        elif args.train_total:
            self.train_mode = "total"
        else:
            raise Exception("训练哪？")

    def get_as_dict(self):
        params = {
            'gpu': self.gpu,
            'epochs': self.epochs,
            'learn_rate': self.learn_rate,
            'learn_rate_seg': self.learn_rate_seg,
            'learn_rate_dec': self.learn_rate_dec,
            'batch_size': self.batch_size,
            'momentum': self.momentum,
            'train_part': self.train_part,
            'train_mode': self.train_mode,
            'mode': self.mode,
            'dataset': self.dataset,
            'max_val': self.max_val,
            'p': self.p,
            'dilation': self.dilation,
            'is_seg_loss_weighted': self.is_seg_loss_weighted,
            'test_during_train': self.test_during_train,
            'sava_imgs': self.sava_imgs,
            'test_frequency': self.test_frequency,
            'save_frequency': self.save_frequency,
            'vis_frequency': self.vis_frequency,
            'delta_cls_loss': self.delta_cls_loss,
            "dyn_balance_loss":self.dyn_balance_loss,
            'GRADIENT_ADJUSTMENT':self.GRADIENT_ADJUSTMENT,
            'test_ratio': self.test_ratio,
            'test_data_dir': self.test_data_dir,
            'train_data_dir': self.train_data_dir,
            'ksdd2_Dataset_dir': self.ksdd2_Dataset_dir,
            'ret_dir': self.ret_dir,
            'vis_dir': self.vis_dir,
            'ckp_dir': self.ckp_dir,
            'feature_dir': self.feature_dir,
            'log_dir': self.log_dir,
            'input_w': self.input_w,
            'input_h': self.input_h,
            'input_c': self.input_c,
        }
