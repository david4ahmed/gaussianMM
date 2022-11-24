% main GMM file


% first we want to load in the images
training_data = dir('train_images');
testing_data = dir('test_images');
training_length = 23;
testing_length = 8;
num_gaussians = 5;
max_iterations = 50;
convergence_threshold = 100;
min_confidence_threshold = 0.3;

% if the variables don't exist in the worspace initialize them
% if they don't exist then training needs to occur
if ~exist('means','var')
    means = [];
end
if ~exist('covariances','var')
    covariances = [];
end
if ~exist('weights','var')
    weights = [];
end

if isempty(weights)
    [means,~,weights,covariances] = trainGMM.train(training_data, training_length, num_gaussians, max_iterations, convergence_threshold);
    disp("Trained! Run again to run on test images!");
else
    testGMM.test(testing_data, testing_length, min_confidence_threshold, means, weights, covariances);
    [depthResult] = measureDepth(training_data, training_length, testing_length);
    plotGMM.show_ellipsoid(means, covariances, 20);
end
