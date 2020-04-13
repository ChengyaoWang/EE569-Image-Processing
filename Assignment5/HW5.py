import numpy 
import torch
import torch.nn as nn 
import torchvision.transforms as transforms
# Other Library
from datetime import datetime
# Self-Defined Files
import Utils
import Modules
import Config

'''
    Different Variation Settings:
        Weight Decay: 0.0 => 0.01
        Learning Rate: 0.001 => 0.01
        Initialization: Uniform(0, 0.1) => Xavier_gaussian_
        Batch Size: 64 => 32
        Normalize Std: 1 => 0.1
'''

# Define some Hyperparameters
# model = Modules.LeNet()
model = Modules.ResNetv1('ResNet18v1')
batch_size = 128
total_epoch = 300
device = Utils.check_device()


# Model Init & Copy
model.apply(Utils.weight_init)
model.to(device)
# Loss Function
lossFunc = nn.CrossEntropyLoss()
# Optimizer & Learning Rate Scheduler
optim_dict = Config.Optimizer['SGD']
scheduler_dict = Config.lr_Scheduler['MultiStepLR']
optim, scheduler = Utils.optim_init(model, optim_dict, scheduler_dict)

# DataSet Fetch & Augmentation
Train_transform = transforms.Compose([transforms.RandomCrop(32, padding = 4),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.1, 0.1, 0.1))])
Test_transform = transforms.Compose([ transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.1, 0.1, 0.1))])

DataLoader, label = Utils.DatasetLoader('CIFAR10', Train_transform, Test_transform, batch_size)

# Initialize the Output Data 
Config_JSON =  {'Model': 'LeNet5',
                'Model_Size_Trainable': sum(p.numel() for p in model.parameters() if p.requires_grad),
                'Model_Size_Total': sum(p.numel() for p in model.parameters()),
                'Batch_Size': batch_size,
                'weight_init': 'Default',
                'total_epoch': total_epoch,
                'device': str(device),
                'train_transform': [trans[4:] for trans in str(Train_transform).split('\n')[1:-1]],
                'test_transform': [trans[4:] for trans in str(Test_transform).split('\n')[1:-1]],
                'Model Structure': [layer[2:] for layer in str(model).split('\n')[1:-1]],
                'Loss Function': str(lossFunc)[:-2],
                'Optimizer': { 'Type': optim_dict['optim_TYPE'],
                               'State': optim.state_dict()['state'],
                               'param_groups': optim.state_dict()['param_groups'][0]},
                'lr_Scheduler': { 'Type': scheduler_dict['schedule_TYPE'],
                                  'State': scheduler.state_dict()},
                'Dataset': 'CIFAR10'}

# Starts Training
# Record Start Time, this serves as time stamp for all the files stored
TimeStamp = str(datetime.now())
StartTime = datetime.now()
train_acc, test_acc, loss = Utils.train(model, DataLoader, lossFunc, optim, device, total_epoch, scheduler)
Config_JSON['TrainingTime'], StartTime = str(datetime.now() - StartTime), datetime.now()
class_prob, ConfuMx = Utils.fine_validate(model, DataLoader[1], device, label)
Config_JSON['InferenceTime'] = str(datetime.now() - StartTime)
Utils.visualize_plt('./save/' + TimeStamp, train_acc, test_acc, loss)
Utils.Save_Model(model, './save/' + TimeStamp)
# Save Config into JSON
Config_JSON['Performance'] = { 'Best_Train': max(train_acc),
                               'Final_Train': train_acc[-1],
                               'Best_Test' : max(test_acc),
                               'Final_Test': test_acc[-1],
                               'Best_Loss':  max(loss),
                               'Final_Loss': loss[-1]}
Config_JSON['Class Performance'] = {label[i]: class_prob[i] for i in range(len(label))}
Config_JSON['Confusion Matrix'] = {label[i]: ConfuMx[i] for i in range(len(label))}
Utils.Save_JSON(Config_JSON, './save/' + TimeStamp)
torch.cuda.empty_cache()
