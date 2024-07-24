##################################################
# Training Config
##################################################
GPU = '0'  # '0,1,2,3'
workers = 4  # number of Dataloader workers
epochs = 160  # number of epochs
batch_size = 16  # batch size
learning_rate = 0.001  # initial learning rate
internalFeature = 1024
# allFeature = 1536
# allFeature = 2048
##################################################
# Model Config
##################################################
image_size = (448, 448)  # size of training images
net = 'inception_mixed_6e'  # feature extractor,'inception_mixed_6e'
num_attentions = 32  # number of attention maps
beta = 5e-2  # param for update feature centers

##################################################
# Dataset/Path Config
##################################################
tag = 'nabirds'  # 'aircraft', 'bird', 'car', or 'dog','nabirds'

# saving directory of .ckpt models
# save_dir = './FGVC/CUB-200-2011/ckpt/'
save_dir = './FGVC/nabirds/ckpt/'
# save_dir = './FGVC/dog/ckpt/'
model_name = 'model.ckpt'
log_name = 'train.log'
testLog_name = 'test.log'

# checkpoint model for resume training
ckpt = False
# ckpt = save_dir + model_name
# best_ckpt = './FGVC/dog/ckpt/model_bestacc.pth'
best_ckpt = './FGVC/nabirds/ckpt/model_bestacc.pth'
# best_ckpt = './FGVC/CUB-200-2011/ckpt/model_bestacc.pth'
##################################################
# Eval Config
##################################################
visual_path = None  # './FGVC/CUB-200-2011/visualize/'
# eval_ckpt = save_dir + model_name
# eval_savepath = './FGVC/CUB-200-2011/visualize/'