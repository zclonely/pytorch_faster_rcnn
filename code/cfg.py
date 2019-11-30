batch_size = 9
num_classes = 2
epoch = 800
lr = 0.005
momentum = 0.9
weight_decay = 0.0005

img_path = '..\dataset\img'
train_path = '..\dataset\json_train'
test_path = '..\dataset\json_test'
checkpoint_path = '..\checkpoint\\'
log_path = '..\log\\'
model_name = 'model_at_epoch_42'
model_path = checkpoint_path + model_name