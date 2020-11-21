function [trainData,tTrain,ijTrain,trainLabels,TestData, tTest,ijTest,testlabels,r,c]=buildfeatures(neighbourSize,numComp,Data,labels,ijindex,trainInd,testInd)
%buildfeatures() builds feature vectors of both training and the test data.
%neighbourSize: size of the neighbourhood area around each target pixel.
%numComp: number of retained PCs in the dimensionality reduction step
%sigma: parameter of the Gaussian filter, used for smoothing the images at
%       each spectral band.
load 'PaviaU';
load 'PaviaU_gt';
%figure(1), imshow(paviaU(:,:,50),[]);
%figure(2), imshow(paviaU_gt,[]);
%  pavia=pavia(:,224:end,:);


%P: the threshold for removing the connected components having fewer than P
%pixels in the binary mask image
P=30;
%MagThreshold: the threshold above which pixels in the gradient image are
%considered strong edges
MagThreshold=0.25;
numThresh=4;
numAttributes=2;

%Computing the EMAP features
EMAPmatrix=PaviaUEMAPmaker(4,numThresh,numAttributes);
EMAPmatrix=double(EMAPmatrix);
x=floor(neighbourSize/2);
%B=ones(1,(neighbourSize*neighbourSize*numComp));

%Here, we calculate the gradient image of the input HSI
magImage=GradientImage(paviaU);
[r,c,lam]=size(paviaU);
%%
%figure(3), imshow(magImage, []);
%%
%Masking magImage with the ground truth image:
for f=1:size(paviaU_gt,1)
    for ff=1:size(paviaU_gt,2)
        if paviaU_gt(f,ff)>0
            paviaU_gt(f,ff)=1;
        end
    end
end
maskedWithGT=double(paviaU_gt).*magImage;
%Thresholding the gradient image using the MagThreshold parameter
mask=magImage>MagThreshold;
%figure(4), imshow(mask,[]);
%Performing morphological opening with thershold P on the mask image
mask = bwareaopen(mask,P);
%figure(5), imshow(mask);title("bwareaopen")
T=bwdist(mask);
%figure(6), imshow(T, []);
mi=min(T(:));
minimum = ones(r,c)*mi;
distanceImage=(T-minimum)/(max(T(:))-min(T(:)));
%figure(7), imshow(distanceImage, []);
distanceImage=double(distanceImage);
T=double(T);
distanceImage=T;
%%
%Applying PCA algorithm on the input HSI to reduce its dimension
paviaUReshaped=reshape(paviaU,[r*c,lam]);
[coeff,score,latent] = pca(paviaUReshaped,'NumComponents',numComp);
paviaPCA=reshape(score,[r,c,size(score,2)]);
for q=1:numComp
    pddedPaviaPCA(:,:,q)=padarray(paviaPCA(:,:,q),[x,x]) ;
end
dim3=size(EMAPmatrix,3);
for q2=1:dim3
    PaddedEMAPmatrix(:,:,q2)=padarray(EMAPmatrix(:,:,q2),[x,x]) ;
end
%%
%Building train features using the values in the distance transform image
numClasses=9;
for k1=1:size(trainInd,1)
    spatial=[];
    spectral=Data(trainInd(k1,1),:);
    for ii=ijindex(trainInd(k1,1))-x:ijindex(trainInd(k1,1))+x
        for jj=ijindex(trainInd(k1,1),2)-x:ijindex(trainInd(k1,1),2)+x
            if ((ii<1) || (jj<1)||(ii>r)||(jj>c))
                distance=0;
            else
                distance=distanceImage(ii,jj);
            end
            temp=[reshape(pddedPaviaPCA(ii+x,jj+x,:),[1,numComp]) reshape(PaddedEMAPmatrix(ii+x,jj+x,:),[1,dim3]) distance];
            spatial=[spatial  temp];
        end
    end
    trainData(k1,:)=[spectral spatial];
    trainLabels(k1,:)=labels(trainInd(k1,1),1);
    ijTrain(k1,1)=ijindex(trainInd(k1,1),1);
    ijTrain(k1,2)=ijindex(trainInd(k1,1),2);
end

trainData=trainData';
tTrain=zeros(numClasses,size(trainData,2));
for k=1:size(trainData,2)
    tTrain(trainLabels(k,1),k)=1;
end
%%
%Building test features using the values in the distance transform image
for k3=1:size(testInd,1)
    spatial=[];
    spectral=Data(testInd(k3,1),:);
    for ii=ijindex(testInd(k3,1))-x:ijindex(testInd(k3,1))+x
        for jj=ijindex(testInd(k3,1),2)-x:ijindex(testInd(k3,1),2)+x
            if ((ii<1) || (jj<1)||(ii>r)||(jj>c))
                distance=0;
            else
                distance=distanceImage(ii,jj);
            end
            temp=[reshape(pddedPaviaPCA(ii+x,jj+x,:),[1,numComp]) reshape(PaddedEMAPmatrix(ii+x,jj+x,:),[1,dim3]) distance];
            spatial=[spatial  temp];
        end
    end
    TestData(k3,:)=[spectral spatial];
    testlabels(k3,:)=labels(testInd(k3,1),1);
    ijTest(k3,1)=ijindex(testInd(k3,1),1);
    ijTest(k3,2)=ijindex(testInd(k3,1),2);
    
end
TestData=TestData';
tTest=zeros(numClasses,size(TestData,2));
tTest=zeros(numClasses,size(TestData,2));
for kkk=1:size(TestData,2)
    tTest(testlabels(kkk,1),kkk)=1;
end




