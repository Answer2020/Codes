
function [magImage]=GradientImage(paviaU,sigma)
%GradientImage() calculates the gradient image of the input HSI
[m,n,p]=size(paviaU);
for jj=1:p
    paviaUSmoothed(:,:,jj)= imgaussfilt(paviaU(:,:,jj), sigma,'Padding','replicate');
end
h45=[-2 -1 0;-1 0 1; 0 1 2];
h135=[0 1 2;-1 0 1;-2 -1 0];
h0 = fspecial('sobel');
h90=h0';

Jt0=zeros(m,n,p);
Jt90=zeros(m,n,p);
Jt45=zeros(m,n,p);
Jt135=zeros(m,n,p);

magImage=zeros(m,n);

for k=10:p
    J0 = imfilter(paviaUSmoothed(:,:,k),h0,'replicate');
    Jt0(:,:,k)=abs(J0);
    J90 = imfilter(paviaUSmoothed(:,:,k),h90,'replicate');
    Jt90(:,:,k)=abs(J90);
    J45 = imfilter(paviaUSmoothed(:,:,k),h45,'replicate');
    Jt45(:,:,k)=abs(J45);
    J135 = imfilter(paviaUSmoothed(:,:,k),h135,'replicate');
    Jt135(:,:,k)=abs(J135);    
end

Jt0accum=sum(Jt0,3);
Jt90accum=sum(Jt90,3);
Jt45accum=sum(Jt45,3);
Jt135accum=sum(Jt135,3);

magImage=(1/4)*(Jt0accum+Jt90accum+Jt45accum+Jt135accum);

mi=min(magImage(:));
ma=max(magImage(:)); 
magImage=(magImage-(mi*ones(m,n)))/(ma-mi);



  

