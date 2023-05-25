unzip carDataset.zip
data = load('carGroundTruth.mat');
carDataset = data.carDataset;

%display first rows and add path to data folder
carDataset(1:4,:)
carDataset.imageFilename = fullfile(pwd,carDataset.imageFilename);
rng(0);
indicies = randperm(height(carDataset));
index = floor(0.6 * length(indicies) );

%training data is split, 60% of data is allocated for training
%image and label data is loaded for training by creating datastores using
%imageDatastore and boxLabelDatastore
trainIndex = 1:index;
trainTable = carDataset(indicies(trainIndex),:);
trainDatastore = imageDatastore(trainTable{:,'imageFilename'});
trainLblDatastore = boxLabelDatastore(trainTable(:,'vehicle'));
trainData = combine(trainDatastore,trainLblDatastore);

%testing data is split, 30% of data is allocated for training
%image and label data is loaded for testing by creating datastores using
%imageDatastore and boxLabelDatastore
testIndex = valIndex(end)+1 : length(indicies);
testTable = carDataset(indicies(testIndex),:);
testDatastore = imageDatastore(testTable{:,'imageFilename'});
testLblDatastore = boxLabelDatastore(testTable(:,'vehicle'));
testData = combine(testDatastore,testLblDatastore);

%validation data is split, 10% of data is allocated for training
%image and label data is loaded for validation by creating datastores using
%imageDatastore and boxLabelDatastore
valIndex = index+1 : index + 1 + floor(0.1 * length(indicies) );
valTable = carDataset(indicies(valIndex),:);
valDatastore = imageDatastore(valTable{:,'imageFilename'});
valLblDatastore = boxLabelDatastore(valTable(:,'vehicle'));
valData = combine(valDatastore,valLblDatastore);

% display an car image with corisponding bounding box
data = read(trainData);
loadedData = data{1};
boundingBox= data{2};
labledCar = insertShape(loadedData,'filled-rectangle',boundingBox);
labledCar = imresize(labledCar,2);

%YOLOv2 Object detection network
%size of the training images 
inputSize = [224 224 3];
%number of classes to detect 
classNum = width(carDataset)-1;
%7 defined anchorboxes using the commputer vision toolbox
anchorNum= 7;
[anchorBoxes, meanIoU] = estimateAnchorBoxes(trainData, anchorNum)
%load pretrained ResNet50 model
extractionNetwork = resnet50;
%set feature extraction layer
featureLayer = 'activation_40_relu';
%storing the created object detection network in a variable
objDetector = yolov2Layers(inputSize,classNum,anchorBoxes,extractionNetwork,featureLayer);
% specify network settings 
options = trainingOptions('sgdm', ...
        'MiniBatchSize',16, ....
        'InitialLearnRate',1e-3, ...
        'MaxEpochs',20, ... 
        'CheckpointPath',tempdir, ...
        'ValidationData',valData)

%plot the log average miss rate to show the rate of miss detection errors compared to the ground truth 
[am,fppi,missRate] = evaluateDetectionMissRate(results,testData);
figure;
loglog(fppi, missRate);
grid on
title(sprintf('Log Average Miss Rate = %.1f', am))

%plot a precision/recall (PR) curve to show how precise the YOLOv2 detector
%is 
results = detect(detector, testData);
[ap,recall,precision] = evaluateDetectionPrecision(results, testData);
figure
plot(recall,precision)
xlabel('Recall')
ylabel('Precision')
grid on
title(sprintf('Average Precision = %.2f',ap))

% show the image with labled bounding boxes
figure
imshow(labledCar)



