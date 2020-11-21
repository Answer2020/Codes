
accArray=zeros(1,10);

clc;
clear all;
close all;
close all;
load 'PaviaU';
load 'PaviaU_gt';
%%
%INITIALIZATION
trainPercen=0.10;
numClasses=9;
%numFolds=10;

depth=2;
hiddenUnitNumbers=60;
neighbourCount=7;
PCAnumComp=5;
%tol=0.001;
MagThresh=0.31
P=50
sigma=0.5
pretrainingEpochs=1000;
fineTuningEpochs=2000;


cvACC=zeros(1,1);
%OA=zeros(1,numFolds);
eachClassTestNumber=zeros(1,numClasses);
eachClass=zeros(1,numClasses);



OATestMean=zeros(5,5,4);
OATestStd=zeros(5,5,4);
KappaTestMean=zeros(5,5,4);
KappaTestStd=zeros(5,5,4);
AATestMean=zeros(5,5,4);
AATestStd=zeros(5,5,4);
OATestMeanEachclass=zeros(5,5,4,numClasses);
OATestStdEachclass=zeros(5,5,4,numClasses);

%%
%In this section, we extract all the labeled pixels (TotTrainingData,
%TotTrainingDataLabels) from the input HSI
%along with their corresponding (row, column) index (ijindex).
%count1 to count9 represent number of available labeled samples for each
%class.
%neighbourCountIndex=0;
[TotTrainingData, TotTrainingDataLabels,ijindex,count1,count2,count3,count4,count5,count6,count7,count8,count9,c1,c2,c3,c4,c5,c6,c7,c8,c9]=buildLabeledData();

%%
OATestvector=zeros(1,10);
AATestvector=zeros(1,10);
KappaTestvector=zeros(1,10);
OATestEachclassMatrix=zeros(10,numClasses);
for iterCounter=1:10
    iterCounter
    %Here, we divide the labeled samples in two groups of train
    %and test based on the given percentage.
    [trainInd, testInd]=divideData(trainPercen,count1,count2,count3,count4,count5,count6,count7,count8,count9,c1,c2,c3,c4,c5,c6,c7,c8,c9);
    trainInd=trainInd';
    testInd=testInd';
    
    
    %Here, we create feature vectors for both the train and the
    %test pixels.
    [xTrainImages, tTrain,ijTrain,trainLabels,TestData, tTest,ijTest,testlabels,row,col]=buildfeatures(neighbourCount,PCAnumComp,MagThresh,P,sigma,TotTrainingData, TotTrainingDataLabels,ijindex,trainInd,testInd);
    
    tTrain1=zeros(numClasses,size(xTrainImages,2));
    for k=1:size(xTrainImages,2)
        tTrain1(trainLabels(k,1),k)=1;
    end
    
    %%
    %Here, we build the stacked sparse autoencoder and train it using our
    %training data. 
    %Depending on how many layers you want in your network, you initialize
    %the parameter "depth" in the initialization step.
    %Here, based on the value of depth, program switches to the appropriate 
    %case, builds and train the corresponding network using the training
    %data.
    switch depth
        case 1
            %call OneAE
            deepnet=OneAE(xTrainImages,hiddenUnitNumbers,pretrainingEpochs,fineTuningEpochs,tTrain1,tTrain);
        case 2
            %call TowAE
            deepnet=TwoAE(xTrainImages,hiddenUnitNumbers,pretrainingEpochs,fineTuningEpochs,tTrain1,tTrain);
            
        case 3
            %call ThreeAE
            deepnet=ThreeAE(xTrainImages,hiddenUnitNumbers,pretrainingEpochs,fineTuningEpochs,tTrain1,tTrain);
            
        case 4
            %call FourAE
            deepnet=FourAE(xTrainImages,hiddenUnitNumbers,pretrainingEpochs,fineTuningEpochs,tTrain1,tTrain);
        case 5
            %call FiveAE
            deepnet=FiveAE(xTrainImages,hiddenUnitNumbers,pretrainingEpochs,fineTuningEpochs,tTrain1,tTrain);   
    end
    %%
    %Testing the deepnet on training data
    yTraindata=deepnet(xTrainImages);  
    yTraindata2=zeros(1,size(yTraindata,2));
    for icountTrain=1:size(yTraindata,2)
        [valTrain, idxTrain] = max(yTraindata(:,icountTrain));
        yTraindata2(1,icountTrain)=idxTrain;
    end
    correctcvCounterTrain=0;
    for ft=1:size(trainLabels,1)
        if trainLabels(ft,1)==yTraindata2(1,ft)
            correctcvCounterTrain=correctcvCounterTrain+1;
        end
    end
    OATrain=(correctcvCounterTrain/size(trainLabels,1))*100

    %%
     %Testing the deepnet on test data
    y = deepnet(TestData);
    y2=zeros(1,size(y,2));
    for icount=1:size(y,2)
        [val, idx] = max(y(:,icount));
        y2(1,icount)=idx;
    end
    
    testlabels=testlabels';
    correctcvCounter=0;
    eachClass=zeros(1,numClasses);
    for f=1:size(testlabels,2)
        if testlabels(1,f)==y2(1,f)
            correctcvCounter=correctcvCounter+1;
        end
        if testlabels(1,f)==1 && y2(1,f)==1
            eachClass(1,1)=eachClass(1,1)+1;
        end
        if testlabels(1,f)==2 && y2(1,f)==2
            eachClass(1,2)=eachClass(1,2)+1;
        end
        if testlabels(1,f)==3 && y2(1,f)==3
            eachClass(1,3)=eachClass(1,3)+1;
        end
        if testlabels(1,f)==4 && y2(1,f)==4
            eachClass(1,4)=eachClass(1,4)+1;
        end
        if testlabels(1,f)==5 && y2(1,f)==5
            eachClass(1,5)=eachClass(1,5)+1;
        end
        if testlabels(1,f)==6 && y2(1,f)==6
            eachClass(1,6)=eachClass(1,6)+1;
        end
        if testlabels(1,f)==7 && y2(1,f)==7
            eachClass(1,7)=eachClass(1,7)+1;
        end
        if testlabels(1,f)==8 && y2(1,f)==8
            eachClass(1,8)=eachClass(1,8)+1;
        end
        if testlabels(1,f)==9 && y2(1,f)==9
            eachClass(1,9)=eachClass(1,9)+1;
        end
    end
    OATest=(correctcvCounter/size(testlabels,2))*100
    OATestvector(1,iterCounter)=OATest;
    
    
%%
%Building the classification map
    RGBGT = label2rgb(paviaU_gt,'hsv');
    %figure, imshow(RGBGT)
    
    map=zeros(row,col);
    
    for k=1:size(TestData,2)
        map(ijTest(k,1),ijTest(k,2))=y2(1,k);
    end
    
    
    for k=1:size(xTrainImages,2)
        map(ijTrain(k,1),ijTrain(k,2))=trainLabels(k,1);
    end
    
    RGB = label2rgb(map,'hsv');
    
    for i=1:size(paviaU_gt,1)
        
        for j=1:size(paviaU_gt,2)
            
            if(RGBGT(i,j,1)==255 & RGBGT(i,j,2)==255 & RGBGT(i,j,3)==255)
                for m=1:3
                    RGBGT(i,j,m)=0;
                    RGB(i,j,m)=0;
                end
            end
            
        end
    end   
    %figure, imshow(RGBGT)
    %print('PaviUGT','-depsc','-r1000');
    figure, imshow(RGB)
    name=sprintf('CMapPaviaUDiatance%d',iterCounter)
    print(name,'-depsc','-r1000');
    print(name,'-dpng','-r1000');
    %%
    %COMPUTE KAPPA COEFFICIENT FOR THE TEST DATA
    ConfMatrixTest = confusionmat(testlabels,y2);  %(known, predicted)
    kappaTest=ComputeKappa(ConfMatrixTest,OATest ,size(TestData,2));
    KappaTestvector(1,iterCounter)=kappaTest;
    %%
    %COMPUTING AVERAGE ACCURACY:
    eachClassAccuracy=zeros(1,numClasses);
    eachClassTestNumber=zeros(1,numClasses);
    for sampleCounter=1:size(testlabels,2)
        if testlabels(1,sampleCounter)==1
            eachClassTestNumber(1,1)=eachClassTestNumber(1,1)+1;
        elseif testlabels(1,sampleCounter)==2
            eachClassTestNumber(1,2)=eachClassTestNumber(1,2)+1;
        elseif testlabels(1,sampleCounter)==3
            eachClassTestNumber(1,3)=eachClassTestNumber(1,3)+1;
        elseif testlabels(1,sampleCounter)==4
            eachClassTestNumber(1,4)=eachClassTestNumber(1,4)+1;
        elseif testlabels(1,sampleCounter)==5
            eachClassTestNumber(1,5)=eachClassTestNumber(1,5)+1;
        elseif testlabels(1,sampleCounter)==6
            eachClassTestNumber(1,6)=eachClassTestNumber(1,6)+1;
        elseif testlabels(1,sampleCounter)==7
            eachClassTestNumber(1,7)=eachClassTestNumber(1,7)+1;
        elseif testlabels(1,sampleCounter)==8
            eachClassTestNumber(1,8)=eachClassTestNumber(1,8)+1;
        else
            eachClassTestNumber(1,9)=eachClassTestNumber(1,9)+1;
        end
    end
    
    for class=1:numClasses
        eachClassAccuracy(1,class)=(eachClass(1,class)/eachClassTestNumber(1,class))*100;
    end
    
    OATestEachclassMatrix(iterCounter,:)=eachClassAccuracy(1,:);  
    AATest=sum(eachClassAccuracy(:))/numClasses;
    AATestvector(1,iterCounter)=AATest;      
end


OATestMean = mean(OATestvector);
OATestStd = std(OATestvector);
KappaTestMean = mean(KappaTestvector);
KappaTestStd = std(KappaTestvector);
AATestMean = mean(AATestvector);
AATestStd = std(AATestvector);

OATestMeanEachclass=mean(OATestEachclassMatrix,1);
OATestStdEachclass=std(OATestEachclassMatrix,1);

%SAVING THE OUTPUTS:
save('OATestMean.mat','OATestMean');
save('OATestStd.mat','OATestStd');
save('KappaTestMean.mat','KappaTestMean');
save('KappaTestStd.mat','KappaTestStd');
save('AATestMean.mat','AATestMean');
save('AATestStd.mat','AATestStd');
save('OATestMeanEachclass.mat','OATestMeanEachclass');
save('OATestStdEachclass.mat','OATestStdEachclass');















