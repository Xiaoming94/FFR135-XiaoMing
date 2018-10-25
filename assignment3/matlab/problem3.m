%% Problem3.m

% AUTHOR: HENRY YANG (940503-1056)


train_data_path = 'train-images-idx3-ubyte';
train_labels_path = 'train-labels-idx1-ubyte';
test_data_path = 't10k-images-idx3-ubyte';
test_labels_path = 't10k-labels-idx1-ubyte';

inputSize = [28 28 1]
labels = 10;

[train_data,train_targets,val_data,val_targets,test_data,test_targets] = LoadMNIST(3);

num_data = size(data_points,2);

indexpermutation = randperm(num_data);

net_layers1 = [
    imageInputLayer(inputSize)
    fullyConnectedLayer(100)
    reluLayer
    fullyConnectedLayer(100)
    reluLayer
    fullyConnectedLayer(labels)
    softmaxLayer
    classificationLayer
]

net_layers2 = [
    imageInputLayer(inputSize)
    fullyConnectedLayer(100)
    reluLayer
    fullyConnectedLayer(100)
    reluLayer
    fullyConnectedLayer(100)
    reluLayer
    fullyConnectedLayer(labels)
    softmaxLayer
    classificationLayer
]

validationPatience = 5
validationFreq = 30

options1 = trainingOptions('sgdm', ...
    'Momentum', 0.9, ...
    'InitialLearnRate', 0.01, ...
    'MiniBatchSize', 8192, ...
    'MaxEpochs', 200, ...
    'ExecutionEnvironment','parallel',...
    'L2Regularization',0,...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {val_data,val_targets},...
    'ValidationPatience', validationPatience,...
    'ValidationFrequency',validationFreq,...
    'Plots', 'training-progress')

options2 = trainingOptions('sgdm', ...
    'Momentum', 0.9, ...
    'InitialLearnRate', 0.01, ...
    'MiniBatchSize', 8192, ...
    'MaxEpochs', 200, ...
    'ExecutionEnvironment','parallel',...
    'L2Regularization',0.03,...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {val_data,val_targets},...
    'ValidationPatience', validationPatience,...
    'ValidationFrequency',validationFreq,...
    'Plots', 'training-progress')


net1 = trainNetwork(train_data,train_targets,net_layers1,options1);

net2 = trainNetwork(train_data,train_targets,net_layers2,options1);

net3 = trainNetwork(train_data,train_targets,net_layers1,options2);

YPred1 = classify(net1,test_data);
YPred2 = classify(net2,test_data);
YPred3 = classify(net3,test_data);

[acc1,cerror1] = classification_error(YPred1,test_targets);
[acc2,cerror2] = classification_error(YPred2,test_targets);
[acc3,cerror3] = classification_error(YPred3,test_targets);
disp('==========')
disp('Performance of Net1')
disp('Accuracy:')
disp(acc1)
disp('Classification error:')
disp(cerror1)
disp('==========')
disp('Performance of Net2')
disp('Accuracy:')
disp(acc2)
disp('Classification error:')
disp(cerror2)
disp('==========')
disp('Performance of Net3')
disp('Accuracy:')
disp(acc3)
disp('Classification error:')
disp(cerror3)