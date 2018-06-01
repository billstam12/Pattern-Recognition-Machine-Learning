% This is a supporting MATLAB file for the project

clear
format compact
close all

load Salinas_hyperspectral %Load the Salinas hypercube called "Salinas_Image"
[p,n,l]=size(Salinas_Image) % p,n define the spatial resolution of the image, while l is the number of bands (number of features for each pixel)
P=[1/2 1/2 1/2 1/2 1/2];

load classification_labels 
% This file contains three arrays of dimension 22500x1 each, called
% "Training_Set", "Test_Set" and "Operational_Set". In order to bring them
% in an 150x150 image format we use the command "reshape" as follows:
Training_Set_Image=reshape(Training_Set, p,n); % In our case p=n=150 (spatial dimensions of the Salinas image).
Test_Set_Image=reshape(Test_Set, p,n);
Operational_Set_Image=reshape(Operational_Set, p,n);

% 
% %Depicting the various bands of the Salinas image
% for i=1:l
%     figure(1), imagesc(Salinas_Image(:,:,i))
%     pause(0.05) % This command freezes figure(1) for 0.05sec. 
% end
% 
% % Depicting the training, test and operational sets of pixels (for the
% % pixels depicted with a dark blue color, the class label is not known.
% % Each one of the other colors in the following figures indicate a class).
% figure(2), imagesc(Training_Set_Image)
% figure(3), imagesc(Test_Set_Image)
% figure(4), imagesc(Operational_Set_Image)

% Constructing the 204xN array whose columns are the vectors corresponding to the
% N vectors (pixels) of the training set (similar codes cane be used for
% the test and the operational sets).
Train=zeros(p,n,l); % This is a 3-dim array, which will contain nonzero values only for the training pixels
for i=1:l
     %Multiply elementwise each band of the Salinas_Image with the mask 
     % "Training_Set_Image>0", which identifies only the training vectors.
    Train(:,:,i)=Salinas_Image(:,:,i).*(Training_Set_Image>0);
   % figure(5), imagesc(Train(:,:,i)) % Depict the training set per band
    pause(0.05)
end

Test=zeros(p,n,l); % This is a 3-dim array, which will contain nonzero values only for the training pixels
for i=1:l
     %Multiply elementwise each band of the Salinas_Image with the mask 
     % "Training_Set_Image>0", which identifies only the training vectors.
    Test(:,:,i)=Salinas_Image(:,:,i).*(Test_Set_Image>0);
   % figure(5), imagesc(Train(:,:,i)) % Depict the training set per band
    pause(0.05)
end

Train_array=[]; %This is the wanted 204xN array
Train_array_response=[]; % This vector keeps the label of each of the training pixels
Train_array_pos=[]; % This array keeps (in its rows) the position of the training pixels in the image.

Test_array=[]; %This is the wanted 204xN array
Test_array_response=[]; % This vector keeps the label of each of the training pixels
Test_array_pos=[]; % This array keeps (in its rows) the position of the training pixels in the image.
for i=1:p
    for j=1:n
        if(Training_Set_Image(i,j)>0) %Check if the (i,j) pixel is a training pixel
            Train_array=[Train_array squeeze(Train(i,j,:))];
            Train_array_response=[Train_array_response Training_Set_Image(i,j)];
            Train_array_pos=[Train_array_pos; i j];
        end
        
        if(Test_Set_Image(i,j)>0) %Check if the (i,j) pixel is a test pixel
            Test_array=[Test_array squeeze(Test(i,j,:))];
            Test_array_response=[Test_array_response Test_Set_Image(i,j)];
            Test_array_pos=[Test_array_pos; i j];
        end
    end
end

Train_array = Train_array';
Train_array_response = Train_array_response';
Test_array = Test_array';
Test_array_response = Test_array_response';
%--------------------------------------------- Naive Bayes Classification -------------------------------------------------------------
%MATLAB's Naive Bayes
%Mdl = fitcnb(Train_array, Train_array_response);
%predictions = predict(Mdl, Test_array);

% Calculate real a priori
for i = 1:5
    Probabilities(i) = sum(Train_array_response(i))/length(Train_array_response);
end

% Calculate mean and standard deviation

for i = 1:5
    mu(i,:) = mean(Train_array((Train_array_response==i),:),1);
    sigma(i,:) = std(Train_array((Train_array_response==i),:),1);
end

% Calculate a posteriori 
for i = 1:length(Test_array)
    %Normal pdf to calculate likelihood for all classes
    p = normpdf( ones(5,1)*Test_array( j, :), mu, sigma);
    P(i,:) = Probabilities.*prod(p,2)';
end 
