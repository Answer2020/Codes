function deepnet=ThreeAE(xTrainImages,hiddenUnitNumbers,pretrainingIterations,fineTuningIterations,tTrain1, tTrain)
%ThreeAE() build a SAE with three layers and train it with the training data.

%First AE
hiddenSize1 = hiddenUnitNumbers;
autoenc1 = trainAutoencoder(xTrainImages,hiddenSize1, ...
    'MaxEpochs',pretrainingIterations, ...
    'L2WeightRegularization',0.004, ...
    'SparsityRegularization',2, ...
    'SparsityProportion',0.8, ...
    'ScaleData', true);
% view(autoenc1)

feat1 = encode(autoenc1,xTrainImages);

%%

%Second AE
hiddenSize2 = hiddenUnitNumbers;
autoenc2 = trainAutoencoder(feat1,hiddenSize2, ...
    'MaxEpochs',pretrainingIterations, ...
    'L2WeightRegularization',0.002, ...
    'SparsityRegularization',2, ...
    'SparsityProportion',0.85, ...
    'ScaleData', true);
% view(autoenc2)
feat2 = encode(autoenc2,feat1);
%%
%Third AE
hiddenSize3 = hiddenUnitNumbers;
autoenc3 = trainAutoencoder(feat2,hiddenSize3, ...
    'MaxEpochs',pretrainingIterations, ...
    'L2WeightRegularization',0.002, ...
    'SparsityRegularization',2, ...
    'SparsityProportion',0.85, ...
    'ScaleData', true);
% view(autoenc2)
feat3 = encode(autoenc3,feat2);

%%
%Training the softmax layer
softnet = trainSoftmaxLayer(feat3,tTrain1,'MaxEpochs',fineTuningIterations);

%Stacking the AEs one after another to build the SAE
deepnet = stack(autoenc1,autoenc2,autoenc3,softnet);

%Training the SAE using our training data
deepnet = train(deepnet,xTrainImages,tTrain);
