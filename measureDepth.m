
function [testResult] = measureDepth(trainingData, numTrain, numTest)

data = load('masks.mat');
trainDists = zeros(1,numTrain);
trainAreas = zeros(1,numTrain);
testAreas = load('testAreas.mat').areas;
testResult = zeros(2, numTest);

for i = 3 : numTrain+2
    dist = extractBefore(trainingData(i).name, '.');
    mask = data.(strcat('mask', dist));
    area = sum(sum(mask));
    trainDists(i-2) = str2num(dist);
    trainAreas(i-2) = area;
end

fitData = fit(trainAreas.', trainDists.', 'poly4');
plot(fitData, transpose(trainAreas), transpose(trainDists));
hold on;

for i = 1:length(testAreas)
    distance = fitData(testAreas(2,i));
    testResult(1,i) = testAreas(1,i);
    testResult(2,i) = distance;
end

save('testDepthResult.mat', 'testResult');

end
