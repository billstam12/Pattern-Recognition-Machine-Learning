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

Op_array=[]; %This is the wanted 204xN array
Op_array_response=[]; % This vector keeps the label of each of the operation pixels
Op_array_pos=[]; % This array keeps (in its rows) the position of the operation pixels in the image.

for i=1:p
    for j=1:n
        if(Training_Set_Image(i,j)>0) %Check if the (i,j) pixel is a training pixel
            Train_array=[Train_array squeeze(Train(i,j,:))];
            Train_array_response=[Train_array_response Training_Set_Image(i,j)];
            Train_array_pos=[Train_array_pos; i j];
        end
        
        
    end
end
for i=1:p
    for j=1:n
        if(Test_Set_Image(i,j)>0) %Check if the (i,j) pixel is a test pixel
            Test_array=[Test_array squeeze(Test(i,j,:))];
            Test_array_response=[Test_array_response Test_Set_Image(i,j)];
            Test_array_pos=[Test_array_pos; i j];
        end
    end
end

for i=1:p
    for j=1:n
        if(Operational_Set_Image(i,j)>0) %Check if the (i,j) pixel is an operation pixel
            Op_array=[Op_array squeeze(Test(i,j,:))];
            Op_array_response=[Op_array_response Operational_Set_Image(i,j)];
            Op_array_pos=[Op_array_pos; i j];
        end
    end
end

%Transpose matrixes
Train_array = Train_array';
Train_array_response = Train_array_response';
Test_array = Test_array';
Test_array_response = Test_array_response';
Op_array = Op_array';
Op_array_response = Op_array_response';

%--------------------------------------------- Naive Bayes Classification -------------------------------------------------------------
%MATLAB's Naive Bayes

Mdl = fitcnb(Train_array, Train_array_response);
predictions = predict(Mdl, Test_array);

confusion_matrix = zeros(5, 5);
for i=1: length(predictions)
     confusion_matrix(Test_array_response(i), predictions(i) ) =  confusion_matrix(Test_array_response(i), predictions(i) ) + 1;
end
fprintf("Matlab Naive Bayes\n");
confusion_matrix
success_rate = trace(confusion_matrix)/  sum(sum(confusion_matrix))

% Calculate real a priori
for i = 1:5
    Probabilities(i) = sum(Train_array_response(i))/length(Train_array_response);
end

% Calculate mean and standard deviation

for i = 1:5
    mu(i,:) = mean(Train_array((Train_array_response==i),:),1);
    sigma(i,:) = std(Train_array((Train_array_response==i),:),1);
end

% Calculate likelihood and a posteriori
for i = 1:length(Train_array)

    p = normcdf(Train_array( i, :), mu, sigma);
    P(i,:) = Probabilities.*prod(p,2)';
end 

% get predicted output for train set
[pv0,id]=max(P,[],2);
for i=1:length(id)
    train_predictions(i,1) = id(i);
end



% Calculate likelihood and a posteriori
for i = 1:length(Test_array)
    p = normcdf(Test_array( i, :), mu, sigma);
    P(i,:) = Probabilities.*prod(p,2)';
end 

% get predicted output for test set
[pv0,id]=max(P,[],2);
for i=1:length(id)
    predictions(i,1) = id(i);
end

% Calculate a posteriori 
for i = 1:length(Op_array)
        % NA YPOLOGISW PINAKA SINDIASPORAS
%     for j = 1:5
%         cmp  = (1/(sqrt(2*pi)*sigma(j)))*exp(-(Test_array(i,:)-mu(j)).^2/(2*sigma(j)));
%         p(j) = prod(cmp,2)
%     end
    p = normcdf(Op_array( i, :), mu, sigma);
    P(i,:) = Probabilities.*prod(p,2)';
end 

% get predicted output for operations set
[pv0,id]=max(P,[],2);
for i=1:length(id)
    op_predictions(i,1) = id(i);
end

%Confusion matrix and Accuracy
confusion_matrix = zeros(5, 5);
for i=1: length(predictions)
     confusion_matrix(Test_array_response(i), predictions(i) ) =  confusion_matrix(Test_array_response(i), predictions(i) ) + 1;
end
fprintf("My Naive Bayes\n");
confusion_matrix
success_rate = trace(confusion_matrix)/  sum(sum(confusion_matrix))

op_confusion_matrix = zeros(5, 5);
for i=1: length(op_predictions)
     op_confusion_matrix(Op_array_response(i), op_predictions(i) ) =  confusion_matrix(Op_array_response(i), op_predictions(i) ) + 1;
end
op_confusion_matrix
success_rate = trace(op_confusion_matrix)/  sum(sum(op_confusion_matrix))

image_colours = zeros(150);
for i=1:length(Train_array)
    if (Train_array_response(i) == 1)
        image_colours(Train_array_pos(i,1), Train_array_pos(i, 2)) = 830;
    elseif (Train_array_response(i) == 2)
        image_colours(Train_array_pos(i,1), Train_array_pos(i, 2)) = 280;
    elseif (Train_array_response(i) == 3)
        image_colours(Train_array_pos(i,1), Train_array_pos(i, 2)) = 400;
    elseif (Train_array_response(i) == 4)
        image_colours(Train_array_pos(i,1), Train_array_pos(i, 2)) = 460;
    elseif (Train_array_response(i) == 5)
        image_colours(Train_array_pos(i,1), Train_array_pos(i, 2)) = 110;
    end
end
for i=1:length(Test_array)
    if (Test_array_response(i) == 1)
        image_colours(Test_array_pos(i,1), Test_array_pos(i, 2)) = 830;
    elseif (Test_array_response(i) == 2)
        image_colours(Test_array_pos(i,1), Test_array_pos(i, 2)) = 280;
    elseif (Test_array_response(i) == 3)
        image_colours(Test_array_pos(i,1), Test_array_pos(i, 2)) = 400;
    elseif (Test_array_response(i) == 4)
        image_colours(Test_array_pos(i,1), Test_array_pos(i, 2)) = 460;
    elseif (Test_array_response(i) == 5)
        image_colours(Test_array_pos(i,1), Test_array_pos(i, 2)) = 110;
    end
end
for i=1:length(Op_array)
    if (Op_array_response(i) == 1)
        image_colours(Op_array_pos(i,1), Op_array_pos(i, 2)) = 830;
    elseif (Op_array_response(i) == 2)
        image_colours(Op_array_pos(i,1), Op_array_pos(i, 2)) = 280;
    elseif (Op_array_response(i) == 3)
        image_colours(Op_array_pos(i,1), Op_array_pos(i, 2)) = 400;
    elseif (Op_array_response(i) == 4)
        image_colours(Op_array_pos(i,1), Op_array_pos(i, 2)) = 460;
    elseif (Op_array_response(i) == 5)
        image_colours(Op_array_pos(i,1), Op_array_pos(i, 2)) = 110;
    end
end

figure(1), imagesc(image_colours)
pause();

my_colours = zeros(150);
for i=1:length(Train_array)
    if (train_predictions(i) == 1)
        my_colours(Train_array_pos(i,1), Train_array_pos(i, 2)) = 830;
    elseif (train_predictions(i) == 2)
        my_colours(Train_array_pos(i,1), Train_array_pos(i, 2)) = 280;
    elseif (train_predictions(i) == 3)
        my_colours(Train_array_pos(i,1), Train_array_pos(i, 2)) = 400;
    elseif (train_predictions(i) == 4)
        my_colours(Train_array_pos(i,1), Train_array_pos(i, 2)) = 460;
    elseif (train_predictions(i) == 5)
        my_colours(Train_array_pos(i,1), Train_array_pos(i, 2)) = 110;
    end
end
for i=1:length(Test_array)
    if (predictions(i) == 1)
        my_colours(Test_array_pos(i,1), Test_array_pos(i, 2)) = 830;
    elseif (predictions(i) == 2)
        my_colours(Test_array_pos(i,1), Test_array_pos(i, 2)) = 280;
    elseif (predictions(i) == 3)
        my_colours(Test_array_pos(i,1), Test_array_pos(i, 2)) = 400;
    elseif (predictions(i) == 4)
        my_colours(Test_array_pos(i,1), Test_array_pos(i, 2)) = 460;
    elseif (predictions(i) == 5)
        my_colours(Test_array_pos(i,1), Test_array_pos(i, 2)) = 110;
    end
end
for i=1:length(Op_array)
    if (op_predictions(i) == 1)
        my_colours(Op_array_pos(i,1), Op_array_pos(i, 2)) = 830;
    elseif (op_predictions(i) == 2)
        my_colours(Op_array_pos(i,1), Op_array_pos(i, 2)) = 280;
    elseif (op_predictions(i) == 3)
        my_colours(Op_array_pos(i,1), Op_array_pos(i, 2)) = 400;
    elseif (op_predictions(i) == 4)
        my_colours(Op_array_pos(i,1), Op_array_pos(i, 2)) = 460;
    elseif (op_predictions(i) == 5)
        my_colours(Op_array_pos(i,1), Op_array_pos(i, 2)) = 110;
    end
end

figure(2), imagesc(my_colours)
pause();

%--------------------------------------------- Euclidean Distance Classifier  -------------------------------------------------------------
%Get mean of each class 
for i=1:5
    mu( i, :) = mean(Train_array((Train_array_response==i),:));
end

dist = [];

train_predictions = [];
for i = 1:length(Train_array)
  for j = 1:5
    dist(i, j) = [sqrt(sum((Train_array(i, :) - mu(j, :)).^ 2 ) )];
  end
end
[dist, train_predictions] = min(dist,[],2);

test_predictions = [];
for i = 1:length(Test_array)
  for j = 1:5
    dist(i, j) = [sqrt(sum((Test_array(i, :) - mu(j, :)).^ 2 ) )];
  end
end
[dist, test_predictions] = min(dist,[],2);


op_predictions = [];
for i = 1:length(Op_array)
  for j = 1:5
    dist(i, j) = [sqrt(sum((Op_array(i, :) - mu(j, :)).^ 2 ) )];
  end
end
[dist, op_predictions] = min(dist,[],2);

confusion_matrix = zeros(5, 5);
for i=1: length(test_predictions)
     confusion_matrix(Test_array_response(i), test_predictions(i) ) =  confusion_matrix(Test_array_response(i), test_predictions(i) ) + 1;
end
fprintf("Euclidean distance classifier\n");
confusion_matrix
success_rate = trace(confusion_matrix)/  sum(sum(confusion_matrix))

op_confusion_matrix = zeros(5, 5);
for i=1: length(op_predictions)
     op_confusion_matrix(Op_array_response(i), op_predictions(i) ) =  confusion_matrix(Op_array_response(i), op_predictions(i) ) + 1;
end
op_confusion_matrix
success_rate = trace(op_confusion_matrix)/  sum(sum(op_confusion_matrix))

my_colours = zeros(150);
for i=1:length(Train_array)
    if (train_predictions(i) == 1)
        my_colours(Train_array_pos(i,1), Train_array_pos(i, 2)) = 830;
    elseif (train_predictions(i) == 2)
        my_colours(Train_array_pos(i,1), Train_array_pos(i, 2)) = 280;
    elseif (train_predictions(i) == 3)
        my_colours(Train_array_pos(i,1), Train_array_pos(i, 2)) = 400;
    elseif (train_predictions(i) == 4)
        my_colours(Train_array_pos(i,1), Train_array_pos(i, 2)) = 460;
    elseif (train_predictions(i) == 5)
        my_colours(Train_array_pos(i,1), Train_array_pos(i, 2)) = 110;
    end
end
for i=1:length(Test_array)
    if (test_predictions(i) == 1)
        my_colours(Test_array_pos(i,1), Test_array_pos(i, 2)) = 830;
    elseif (test_predictions(i) == 2)
        my_colours(Test_array_pos(i,1), Test_array_pos(i, 2)) = 280;
    elseif (test_predictions(i) == 3)
        my_colours(Test_array_pos(i,1), Test_array_pos(i, 2)) = 400;
    elseif (test_predictions(i) == 4)
        my_colours(Test_array_pos(i,1), Test_array_pos(i, 2)) = 460;
    elseif (test_predictions(i) == 5)
        my_colours(Test_array_pos(i,1), Test_array_pos(i, 2)) = 110;
    end
end
for i=1:length(Op_array)
    if (op_predictions(i) == 1)
        my_colours(Op_array_pos(i,1), Op_array_pos(i, 2)) = 830;
    elseif (op_predictions(i) == 2)
        my_colours(Op_array_pos(i,1), Op_array_pos(i, 2)) = 280;
    elseif (op_predictions(i) == 3)
        my_colours(Op_array_pos(i,1), Op_array_pos(i, 2)) = 400;
    elseif (op_predictions(i) == 4)
        my_colours(Op_array_pos(i,1), Op_array_pos(i, 2)) = 460;
    elseif (op_predictions(i) == 5)
        my_colours(Op_array_pos(i,1), Op_array_pos(i, 2)) = 110;
    end
end

figure(3), imagesc(my_colours)
pause();


%--------------------------------------------- Cross-Validate-Nearest Neighbor -------------------------------------------------------------

cv = 5;
len = length(Train_array);
no_of_features = fix(len / cv);
best_predictor = 0 ;
max_accuracy = 0;

for k = [3 5 7 9 11 13 15 17]
 
    consistency = 0;
    for i = 0:(cv-1)
        start_line = 1 + (i*no_of_features);
        end_line = start_line + no_of_features-1;


        X_test = Train_array(start_line : end_line, :);
        y_test = Train_array_response(start_line:end_line);

        X_train = Train_array(1:start_line,:);
        X_train = [ X_train; Train_array(end_line:len,:)];
        y_train = Train_array_response(1:start_line);
        y_train = [ y_train; Train_array_response(end_line:len)];

        size(X_train);
        size(y_train);


        tree = KDTreeSearcher(X_train, 'Distance' , 'euclidean' , 'BucketSize',10);
        ids = knnsearch(tree, X_test, 'K',5);

        classes = ones(length(ids),5);
        for i = 1:length(classes)
            for j =1:5
                classes(i,j) = y_train(ids(i));
            end
        end
        
        predictions = zeros(length(classes),1);
        for i  = 1:length(classes)    
            predictions(i) = mode(classes(i,:));
        end
        
        consistency = consistency + sum(predictions == y_test)/length(y_test)

    end

    accuracy = consistency/cv;
    if (accuracy > max_accuracy)
        max_accuracy = accuracy;
        best_predictor = k;
    end
end   

best_predictor
max_accuracy

%--------------------------------------------- 3-Nearest Neighbor -------------------------------------------------------------

%MATLAB's KNN
mdl = fitcknn(Train_array, Train_array_response, 'NumNeighbors', 3) ;       
predictions = predict(mdl, Test_array);

confusion_matrix = zeros(5, 5);
for i=1: length(predictions)
     confusion_matrix(Test_array_response(i), predictions(i) ) =  confusion_matrix(Test_array_response(i), predictions(i) ) + 1;
end
fprintf("Matlab KNN\n");
confusion_matrix
success_rate = trace(confusion_matrix)/  sum(sum(confusion_matrix))

tree = KDTreeSearcher(Train_array, 'Distance' , 'euclidean' , 'BucketSize',10);
ids = knnsearch(tree, Test_array, 'K',3);

classes = ones(length(ids),5);
for i = 1:length(classes)
    for j =1:5
        classes(i,j) = Train_array_response(ids(i));
    end
end

for i  = 1:length(classes)    
    predictions(i) = mode(classes(i,:));
end

confusion_matrix = zeros(5, 5);
for i=1: length(predictions)
     confusion_matrix(Test_array_response(i), predictions(i) ) =  confusion_matrix(Test_array_response(i), predictions(i) ) + 1;
end
fprintf("My KNN\n");
confusion_matrix
success_rate = trace(confusion_matrix)/  sum(sum(confusion_matrix))

tree = KDTreeSearcher(Train_array, 'Distance' , 'euclidean' , 'BucketSize',10);
ids = knnsearch(tree, Op_array, 'K',3);

classes = ones(length(ids),5);
for i = 1:length(classes)
    for j =1:5
        classes(i,j) = Train_array_response(ids(i));
    end
end

for i  = 1:length(classes)    
    op_predictions(i) = mode(classes(i,:));
end

op_confusion_matrix = zeros(5, 5);
for i=1: length(op_predictions)
     op_confusion_matrix(Op_array_response(i), op_predictions(i) ) =  confusion_matrix(Op_array_response(i), op_predictions(i) ) + 1;
end
op_confusion_matrix
success_rate = trace(op_confusion_matrix)/  sum(sum(op_confusion_matrix))

my_colours = zeros(150);
for i=1:length(Train_array)
    if (Train_array_response(i) == 1)
        my_colours(Train_array_pos(i,1), Train_array_pos(i, 2)) = 830;
    elseif (Train_array_response(i) == 2)
        my_colours(Train_array_pos(i,1), Train_array_pos(i, 2)) = 280;
    elseif (Train_array_response(i) == 3)
        my_colours(Train_array_pos(i,1), Train_array_pos(i, 2)) = 400;
    elseif (Train_array_response(i) == 4)
        my_colours(Train_array_pos(i,1), Train_array_pos(i, 2)) = 460;
    elseif (Train_array_response(i) == 5)
        my_colours(Train_array_pos(i,1), Train_array_pos(i, 2)) = 110;
    end
end
for i=1:length(Test_array)
    if (predictions(i) == 1)
        my_colours(Test_array_pos(i,1), Test_array_pos(i, 2)) = 830;
    elseif (predictions(i) == 2)
        my_colours(Test_array_pos(i,1), Test_array_pos(i, 2)) = 280;
    elseif (predictions(i) == 3)
        my_colours(Test_array_pos(i,1), Test_array_pos(i, 2)) = 400;
    elseif (predictions(i) == 4)
        my_colours(Test_array_pos(i,1), Test_array_pos(i, 2)) = 460;
    elseif (predictions(i) == 5)
        my_colours(Test_array_pos(i,1), Test_array_pos(i, 2)) = 110;
    end
end
for i=1:length(Op_array)
    if (op_predictions(i) == 1)
        my_colours(Op_array_pos(i,1), Op_array_pos(i, 2)) = 830;
    elseif (op_predictions(i) == 2)
        my_colours(Op_array_pos(i,1), Op_array_pos(i, 2)) = 280;
    elseif (op_predictions(i) == 3)
        my_colours(Op_array_pos(i,1), Op_array_pos(i, 2)) = 400;
    elseif (op_predictions(i) == 4)
        my_colours(Op_array_pos(i,1), Op_array_pos(i, 2)) = 460;
    elseif (op_predictions(i) == 5)
        my_colours(Op_array_pos(i,1), Op_array_pos(i, 2)) = 110;
    end
end

figure(4), imagesc(my_colours)
pause();



    

