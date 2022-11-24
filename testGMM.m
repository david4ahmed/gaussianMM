classdef testGMM
    methods (Static)
        function [confidence, probability, mean_probability, average] = compute_posterior(means, weights, covariances, rgb_vector)
            % get the mean of the mean rgb value taking into account the weight for
            % each mean
            average = 0;
            for i=1:size(weights,1)
                average = average + double(floor((means(:,:,i) * weights(i))));
            end

            %     get the probability of the vector
            probability = 0;
            for i=1:size(weights,1)
                distribution = weights(i) * gaussian.calculate_probability(means(:,:,i), covariances(:,:,i), rgb_vector);
                probability = probability + distribution;
            end

            %     get the probability of the mean of means, we will measure the
            %     probability with respect to the mean
            mean_probability = 0;
            for i=1:size(weights,1)
                distribution = weights(i) * gaussian.calculate_probability(means(:,:,i), covariances(:,:,i), average);
                mean_probability = mean_probability + distribution;
            end

            %     transform the probability into a confidence
            confidence = probability/mean_probability;
        end 

        function test(testing_data, testing_length, min_confidence_threshold, means, weights, covariances)
            start = 3;
            areas = zeros(2, testing_length);
            index = 0;
            
            for i=start:start+testing_length-1
                %  read the image
                image = imread(strcat('test_images/', testing_data(i).name));
                colorMap = zeros(size(image,1), size(image,2));
                area = 0;
                index = index + 1;
                for j=1:size(image, 1)
                    for k=1:size(image, 2)
                        rgb_vector = [
                            double(image(j,k,1));
                            double(image(j,k,2));
                            double(image(j,k,3));
                        ];
                        [confidence, probability] = testGMM.compute_posterior(means, weights, covariances, rgb_vector);
                        if confidence >= min_confidence_threshold
%                             image(j,k,:) = [255;255;255];
                                 colorMap(j,k,:) = confidence;
                            area = area + 1;
                        else
%                             image(j,k,:) = [0;0;0];
                            colorMap(j,k,:) = confidence;
                        end
                    end
                end
                
                    clims =  [0, 1];
                    im = imagesc(colorMap, clims);
                    colorbar;

                
                
                imageNum = extractBefore(testing_data(i).name, '.');
                baseFileName = sprintf('%s.jpg', imageNum);
                fullFileName = fullfile('results', baseFileName);
                saveas((im), fullFileName);
% %                 
% imshow(image);

                
               
                
%                 imshow(im);
                
                %save the area
                areas(1,index) = str2num(imageNum);
                areas(2,index) = area;
                
                fprintf('%s was saved to %s\n', baseFileName, fullFileName);
            end
            
            save('testAreas.mat', 'areas');
        end
    end
end




 
% clear
% 
% training_data = dir('train_images');
% 
% % run this section to train
% [means, ~, weights, covariances] = trainGMM.train(training_data, 16, 5, 50, 100);
% 
% %% 
% 
% % the confidence is how many times better it is from the mean of means
% % a confidence of 3 signifies the pixel is 3 times more orange than the
% % mean of means
% [confidence] = compute_posterior(means, weights, covariances, [0;0;0]);
% 
% %% 
% 