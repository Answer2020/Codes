function deepnet=OneAE(xTrainImages,hiddenUnitNumbers,pretrainingIterations,fineTuningIterations,tTrain1, tTrain)
%OneAE() build a SAE with one layer and train it with the training data.

%First AE
hiddenSize1 = hiddenUnitNumbers;
autoenc1 = trainAutoencoder(xTrainImages,hiddenSize1, ...
    'MaxEpochs',pretrainingIterations, ...
    'L2WeightRegularization',0.004, ...
    'SparsityRegularization',2, ...
    'SparsityProportion',0.8, ...
    'ScaleData', true);


feat1 = encode(autoenc1,xTrainImages);

%Training the softmax layer
softnet = trainSoftmaxLayer(feat1,tTrain1,'MaxEpochs',fineTuningIterations);
% view(softnet)
%%
%Stacking the AEs one after another to build the SAE
deepnet = stack(autoenc1,softnet);

%Training the SAE using our training data
deepnet = train(deepnet,xTrainImages,tTrain);