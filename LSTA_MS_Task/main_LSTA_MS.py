from __future__ import print_function, division
from LSTA_MS_Task.attentionModel import *
from spatial_transforms import (Compose, ToTensor, CenterCrop, Scale, Normalize, MultiScaleCornerCrop,
                                RandomHorizontalFlip)
from tensorboardX import SummaryWriter
from LSTA_MS_Task.makeDataset import *
import sys
import argparse
# from LSTA_MS_Task.gen_splits import *
import os
import torch.nn as nn
from torchvision.transforms import Resize

#TODO: create separate dirs for stage1 and stage 2

def main_run(dataset, stage, root_dir, out_dir, stage1_dict,seqLen, trainBatchSize, numEpochs, lr1, decay_factor,
             decay_step, memSize, outPool_size, split, evalInterval, regression, rloss, debug):
    if debug:
        n_workers = 0
        n_workers_test = 0
        device = 'cpu'
    else:
        n_workers = 4
        n_workers_test = 2
        device = 'cuda'
    # Train/Validation/Test split
    train_splits = ["S1", "S3", "S4"]
    val_splits = ["S2"]

    test_split = split

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    normalize = Normalize(mean=mean, std=std)

    stage = stage
    #test_split = split
    seqLen = seqLen
    memSize = memSize
    c_cam_classes = outPool_size
    dataset = dataset
    best_acc = 0

    if stage == 1:
        trainBatchSize = trainBatchSize
        testBatchSize = trainBatchSize
        lr1 = lr1
        decay_factor = decay_factor
        decay_step = decay_step
        numEpochs = numEpochs
    elif stage == 2:
        trainBatchSize = trainBatchSize
        testBatchSize = trainBatchSize
        lr1 = lr1
        decay_factor = decay_factor
        decay_step = decay_step
        numEpochs = numEpochs

    num_classes = 61

    dataset_dir = root_dir

    #model_folder = os.path.join('.', out_dir, dataset, str(test_split))
    model_folder = os.path.join('./', out_dir, 'stage' + str(stage))
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    else:
        print('Directory {} exists!'.format(model_folder))
        sys.exit()

    note_fl = open(model_folder + '/note.txt', 'w')
    note_fl.write('Number of Epochs = {}\n'
                  'lr = {}\n'
                  'Train Batch Size = {}\n'
                  'Sequence Length = {}\n'
                  'Decay steps = {}\n'
                  'Decay factor = {}\n'
                  'Memory size = {}\n'
                  'Memory cam classes = {}\n'.format(numEpochs, lr1, trainBatchSize, seqLen, decay_step, decay_factor,
                                                   memSize, c_cam_classes))

    note_fl.close()

    # Log files
    writer = SummaryWriter(model_folder)
    train_log_loss = open((model_folder + '/train_log_loss.txt'), 'w')
    train_log_acc = open((model_folder + '/train_log_acc.txt'), 'w')
    train_log_loss_batch = open((model_folder + '/train_log_loss_batch.txt'), 'w')
    test_log_loss = open((model_folder + '/test_log_loss.txt'), 'w')
    test_log_acc = open((model_folder + '/test_log_acc.txt'), 'w')

    # Dataloaders
    spatial_transform = Compose([Scale(256), RandomHorizontalFlip(), MultiScaleCornerCrop([1, 0.875, 0.75, 0.65625], 224),
                                 ToTensor(), normalize])


    print('Preparing dataset...')
    # Data loader
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    spatial_transform = Compose([Scale(256),
                                 RandomHorizontalFlip(),
                                 MultiScaleCornerCrop([1, 0.875, 0.75, 0.65625], 224),
                                 # ToTensor(),
                                 # normalize
                                 ])
    transform_rgb = Compose([ToTensor(), normalize])
    transform_MS = Compose([Resize((7, 7)), ToTensor()])

    vid_seq_train = makeDataset(dataset_dir, splits=train_splits,
                                spatial_transform=spatial_transform,
                                transform_rgb=transform_rgb,
                                transform_MS=transform_MS,
                                seqLen=seqLen,
                                fmt='.png',
                                regression=regression)

    train_loader = torch.utils.data.DataLoader(vid_seq_train, batch_size=trainBatchSize,
                                               shuffle=True, num_workers=n_workers, pin_memory=True)

    vid_seq_test = makeDataset(dataset_dir, splits=val_splits,
                              spatial_transform=Compose([Scale(256), CenterCrop(224)]),
                              transform_rgb=transform_rgb,
                              transform_MS=transform_MS,
                              seqLen=seqLen,
                              fmt='.png',
                              regression=regression,
                              verbose=False)

    test_loader = torch.utils.data.DataLoader(vid_seq_test, batch_size=testBatchSize,
                                             shuffle=False, num_workers=n_workers, pin_memory=True)


    print('Number of train samples = {}'.format(vid_seq_train.__len__()))

    print('Number of test samples = {}'.format(vid_seq_test.__len__()))


    train_params = []
    if stage == 1:
        if regression:
            model = attentionModel(num_classes=num_classes, mem_size=memSize, n_channels=1)
        else:
            model = attentionModel(num_classes=num_classes, mem_size=memSize)
        model.train(False)
        for params in model.parameters():
            params.requires_grad = False
    elif stage == 2:
        if regression:
            model = attentionModel(num_classes=num_classes, mem_size=memSize, n_channels=1)
        else:
            model = attentionModel(num_classes=num_classes, mem_size=memSize)

        model = attentionModel(num_classes=num_classes, mem_size=memSize, c_cam_classes=c_cam_classes)
        checkpoint_path = os.path.join(stage1_dict, 'last_checkpoint_stage' + str(1) + '.pth.tar')
        if os.path.exists(checkpoint_path):
                print('Loading weights from checkpoint file {}'.format(checkpoint_path))
        else:
                print('Checkpoint file {} does not exist'.format(checkpoint_path))
                sys.exit()
        last_checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(last_checkpoint['model_state_dict'])
        model.train(False)
        for params in model.parameters():
            params.requires_grad = False

        for params in model.resNet.layer4[0].conv1.parameters():
            params.requires_grad = True
            train_params += [params]

        for params in model.resNet.layer4[0].conv2.parameters():
            params.requires_grad = True
            train_params += [params]

        for params in model.resNet.layer4[1].conv1.parameters():
            params.requires_grad = True
            train_params += [params]

        for params in model.resNet.layer4[1].conv2.parameters():
            params.requires_grad = True
            train_params += [params]

        for params in model.resNet.layer4[2].conv1.parameters():
            params.requires_grad = True
            train_params += [params]

        for params in model.resNet.layer4[2].conv2.parameters():
            params.requires_grad = True
            train_params += [params]

        for params in model.resNet.fc.parameters():
            params.requires_grad = True
            train_params += [params]

        # Add params from ms_module
        for params in model.ms_module.parameters():
            params.requires_grad = True
            train_params += [params]

    for params in model.lsta_cell.parameters():
        params.requires_grad = True
        train_params += [params]

    for params in model.classifier.parameters():
        params.requires_grad = True
        train_params += [params]

    model.classifier.train(True)

    model.ms_module.train(True)
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()

    if regression:
        if rloss == 'MSE':
            # Mean Squared Error loss
            loss_ms_fn = nn.MSELoss()  # it should work
        elif rloss == 'L1':
            # L1 loss
            loss_ms_fn = nn.L1Loss()
        elif rloss == 'SmoothL1':
            # Huber Loss or Smooth L1 Loss
            loss_ms_fn = nn.SmoothL1Loss()
        elif rloss == 'KLdiv':
            # Kullback-Leiber Loss
            loss_ms_fn = nn.KLDivLoss()
    else:
        # classification
        loss_ms_fn = nn.CrossEntropyLoss()  # TODO: check paper Planamente

    optimizer_fn = torch.optim.Adam(train_params, lr=lr1, weight_decay=5e-4, eps=1e-4)

    optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_fn, milestones=decay_step, gamma=decay_factor)

    train_iter = 0

    for epoch in range(numEpochs):
        #optim_scheduler.step()
        epoch_loss = 0
        numCorrTrain = 0
        trainSamples = 0
        iterPerEpoch = 0
        # model.classifier.train(True)
        model.lsta_cell.train(True)
        model.classifier.train(True)
        if stage == 2:
            model.resNet.layer4[0].conv1.train(True)
            model.resNet.layer4[0].conv2.train(True)
            model.resNet.layer4[1].conv1.train(True)
            model.resNet.layer4[1].conv2.train(True)
            model.resNet.layer4[2].conv1.train(True)
            model.resNet.layer4[2].conv2.train(True)
            model.resNet.fc.train(True)
            model.ms_module.train(True)

        writer.add_scalar('lr', optimizer_fn.param_groups[0]['lr'], epoch+1)
        for i, (inputs, inputsMS, targets) in enumerate(train_loader):
            # Inputs:
            #   - inputsRGB : the rgb frame input
            # Labels :
            #   - inputsMS  : the motion task label
            #   - targets   : output

            train_iter += 1
            iterPerEpoch += 1
            optimizer_fn.zero_grad()
            inputVariable = inputs.permute(1, 0, 2, 3, 4).to(device)
            labelVariable = targets.to(device)
            msVariable = inputsMS.to(device)
            trainSamples += inputs.size(0)
            output_label, _, output_ms = model(inputVariable, device)
            loss_c = loss_fn(output_label, labelVariable)
            if regression:
                msVariable = torch.reshape(msVariable, (seqLen * 7 * 7, msVariable.size(0)))
                output_ms = torch.sigmoid(output_ms)
                output_ms = torch.reshape(output_ms, (seqLen * 7 * 7, output_ms.size(0)))
            else:
                # classification task
                msVariable = torch.reshape(msVariable, (seqLen * 7 * 7, msVariable.size(0))).long()
                output_ms = torch.reshape(output_ms, (seqLen * 7 * 7, 2, output_ms.size(0)))
            loss_ms = loss_ms_fn(output_ms, msVariable)

            loss = loss_c + loss_ms
            loss.backward()
            optimizer_fn.step()
            _, predicted = torch.max(output_label.data, 1)
            numCorrTrain += (predicted == targets.to(device)).sum()
            #print('Training loss after {} iterations = {} '.format(train_iter, loss.data.item()))
            #train_log_loss_batch.write('Training loss after {} iterations = {}\n'.format(train_iter, loss.data.item()))
            #writer.add_scalar('train/iter_loss', loss.data.item(), train_iter)
            epoch_loss += loss.data.item()
        avg_loss = epoch_loss/iterPerEpoch
        trainAccuracy = (numCorrTrain / trainSamples) * 100
        print('Average training loss after {} epoch = {} '.format(epoch+1, avg_loss))
        print('Training accuracy after {} epoch = {}% '.format(epoch+1, trainAccuracy))
        writer.add_scalar('train/epoch_loss', avg_loss, epoch+1)
        writer.add_scalar('train/accuracy', trainAccuracy, epoch+1)
        train_log_loss.write('Training loss after {} epoch = {}\n'.format(epoch+1, avg_loss))
        train_log_acc.write('Training accuracy after {} epoch = {}\n'.format(epoch+1, trainAccuracy))

        save_path_model = os.path.join(model_folder, 'last_checkpoint_stage' + str(stage) + '.pth.tar')
        save_file = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer_fn.state_dict(),
                'best_acc': best_acc,
            }
        torch.save(save_file, save_path_model)

        if (epoch+1) % evalInterval == 0:
            #print('Testing...')
            model.train(False)
            test_loss_epoch = 0
            test_iter = 0
            test_samples = 0
            numCorr = 0
            for j, (inputs, inputsMS, targets) in enumerate(test_loader):
                #print('testing inst = {}'.format(j))
                test_iter += 1
                test_samples += inputs.size(0)
                inputVariable = inputs.permute(1, 0, 2, 3, 4).to(device)
                labelVariable = targets.to(device)
                msVariable = inputsMS.to(device)

                output_label, _, output_ms = model(inputVariable, device)
                test_loss_c = loss_fn(output_label, labelVariable)

                if regression:
                    msVariable = torch.reshape(msVariable, (seqLen * 7 * 7, msVariable.size(0)))
                    output_ms = torch.sigmoid(output_ms)
                    output_ms = torch.reshape(output_ms, (seqLen * 7 * 7, output_ms.size(0)))
                else:
                    # classification task
                    msVariable = torch.reshape(msVariable, (seqLen * 7 * 7, msVariable.size(0))).long()
                    output_ms = torch.reshape(output_ms, (seqLen * 7 * 7, 2, output_ms.size(0)))
                test_loss_ms = loss_ms_fn(output_ms, msVariable)

                test_loss = test_loss_c + test_loss_ms
                test_loss_epoch += test_loss.data.item()
                _, predicted = torch.max(output_label.data, 1)
                numCorr += (predicted == targets.to(device)).sum()
            test_accuracy = (numCorr / test_samples) * 100
            avg_test_loss = test_loss_epoch / test_iter
            print('Test Loss after {} epochs, loss = {}'.format(epoch + 1,avg_test_loss))
            print('Test Accuracy after {} epochs = {}%'.format(epoch + 1, test_accuracy))
            writer.add_scalar('test/epoch_loss', avg_test_loss, epoch + 1)
            writer.add_scalar('test/accuracy', test_accuracy, epoch + 1)
            test_log_loss.write('Test Loss after {} epochs = {}\n'.format(epoch + 1, avg_test_loss))
            test_log_acc.write('Test Accuracy after {} epochs = {}%\n'.format(epoch + 1, test_accuracy))

            if test_accuracy > best_acc:
                best_acc = test_accuracy
                save_path_model = os.path.join(model_folder, 'best_checkpoint_stage' + str(stage) + '.pth.tar')
                save_file = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer_fn.state_dict(),
                    'best_acc': best_acc,
                }
                torch.save(save_file, save_path_model)
        optim_scheduler.step()
    train_log_loss.close()
    train_log_acc.close()
    test_log_acc.close()
    train_log_loss_batch.close()
    test_log_loss.close()
    writer.export_scalars_to_json(model_folder + "/all_scalars.json")
    writer.close()


def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='gtea61', help='Dataset')
    parser.add_argument('--stage', type=int, default=1, help='Training stage')
    parser.add_argument('--root_dir', type=str, default='dataset',
                        help='Dataset directory')
    parser.add_argument('--stage1dict', type=str, default='dataset',
                        help='Dataset directory')
    parser.add_argument('--outDir', type=str, default='experiments', help='Directory to save results')
    parser.add_argument('--seqLen', type=int, default=25, help='Length of sequence')
    parser.add_argument('--trainBatchSize', type=int, default=32, help='Training batch size')
    parser.add_argument('--numEpochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--stepSize', type=float, default=[25, 75, 150], nargs="+", help='Learning rate decay step')
    parser.add_argument('--decayRate', type=float, default=0.1, help='Learning rate decay rate')
    parser.add_argument('--memSize', type=int, default=512, help='ConvLSTM hidden state size')
    parser.add_argument('--outPoolSize', type=int, default=100, help='Output pooling size')
    parser.add_argument('--evalInterval', type=int, default=5, help='Evaluation interval')
    parser.add_argument('--split', type=int, default=1, help='Split')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--regression', action='store_true')
    parser.add_argument('--rloss', type=str, default='MSE', help='Regression Loss')

    args = parser.parse_args()

    dataset = args.dataset
    stage = args.stage
    root_dir = args.root_dir
    stage1_dict = args.stage1dict
    outDir = args.outDir
    seqLen = args.seqLen
    trainBatchSize = args.trainBatchSize
    numEpochs = args.numEpochs
    lr1 = args.lr
    stepSize = args.stepSize
    decayRate = args.decayRate
    memSize = args.memSize
    outPool_size = args.outPoolSize
    evalInterval = args.evalInterval
    split = args.split
    debug = args.debug
    regression = args.regression
    rloss = args.rloss

    main_run(dataset=dataset, stage=stage, root_dir=root_dir, stage1_dict=stage1_dict, out_dir=outDir, seqLen=seqLen,
             trainBatchSize=trainBatchSize, numEpochs=numEpochs, lr1=lr1, decay_factor=decayRate,
             decay_step=stepSize, memSize=memSize, outPool_size=outPool_size, evalInterval=evalInterval,
             split=split, regression=regression, rloss=rloss, debug=debug)

__main__()