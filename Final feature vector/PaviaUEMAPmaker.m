function emapMatrix=PaviaUEMAPmaker(numComp,numThresh,numAttributes);
%PaviaUEMAPmaker() computes the EMAP feature matrix
%numAttributes:Number of attributes to use in the EMAP method
%numThresh:Number of thresholds to use for each attribute

load 'PaviaU';
load 'PaviaU_gt';
[r,c,lam]=size(paviaU);
ap1=[];
ap2=[];
%%
%Dimensionality reduction with PCA
paviaUReshaped=reshape(paviaU,[r*c,lam]);
[coeff,score,latent] = pca(paviaUReshaped,'NumComponents',numComp);
paviaPCA=reshape(score,[r,c,size(score,2)]);
paviaPCA=uint8(paviaPCA);
%%
%Building the EMAP matrix
for attCount=1:numAttributes
    switch attCount
        case 1
            for i=1:numComp
               A = attribute_profile(paviaPCA(:,:,i), 'a', [100 500 1000 5000]); 
                ap1 = cat(3, ap1, A);
            end
        case 2
            for i=1:numComp
                B=attribute_profile(paviaPCA(:,:,i), 'd', [10 25 50 100]);
                B(:,:,(numThresh+1))=[];
                ap2 = cat(3, ap2, B);
            end
    end
end
emapMatrix=cat(3, ap1, ap2);


