% single gaussian
classdef gaussian
    methods (Static)
        function [mu, N] = calculate_mean(training_data, training_length)
            N = 0;
            totalRed = 0;
            totalBlue = totalRed;
            totalGreen = totalRed;


            for i=3:training_length+2
        %         read the image
                image = imread(strcat('train_images/', training_data(i).name));
        %         load the appropriate mask
                load('masks.mat', strcat('mask',extractBefore(training_data(i).name, '.')));
        %         get the binary representation of the mask
                mask = eval(strcat('mask',extractBefore(training_data(i).name, '.')));

                redChannel = image(:,:,1);
                greenChannel = image(:,:,2);
                blueChannel = image(:,:,3);

                redValues = redChannel(mask);
                greenValues = greenChannel(mask);
                blueValues = blueChannel(mask);

                totalRed = totalRed + sum(redValues);
                totalGreen = totalGreen + sum(greenValues);
                totalBlue = totalGreen + sum(blueValues);

                N = N + size(redValues, 1);
            end

            mu = [
                double(floor(totalRed/N));
                double(floor(totalGreen/N));
                double(floor(totalBlue/N));
            ];
        end

        function sigma = calculate_covariance(training_data, training_length, mu)
            sum = [];

            N = 0;

            for i=3:training_length+2
        %         read the image
                image = imread(strcat('train_images/', training_data(i).name));
        %         load the appropriate mask
                load('masks.mat', strcat('mask',extractBefore(training_data(i).name, '.')));
        %         get the binary representation of the mask
                mask = eval(strcat('mask',extractBefore(training_data(i).name, '.')));

                redChannel = image(:,:,1);
                greenChannel = image(:,:,2);
                blueChannel = image(:,:,3);

                redValues = redChannel(mask);
                greenValues = greenChannel(mask);
                blueValues = blueChannel(mask);

                for j=1:size(redValues, 1)
        %             rgb value
                    x_i = [double(redValues(i)); double(greenValues(i)); double(blueValues(i))];
        %             one instance of the covariance summation
                    cov_instance = (x_i - mu) * (x_i - mu)';
                    
                    if(isempty(sum))
                        sum = cov_instance;
                    else
                        sum = sum + cov_instance;
                    end
                    N = N + 1;
                end

            end 
            sigma = sum / N;
            
        end
     
        
        function [probability, likelyhood] = calculate_probability(mu, sigma, rgb_vector)       
            prior = 0.5;
            body = 1/(sqrt((2*pi)^3 * det(sigma)));
            expon = (-0.5 * ((rgb_vector - mu)'/sigma) * (rgb_vector - mu));
            
%             if rcond(sigma) < 0.5
%                 disp(sigma);
%                 disp(rcond(sigma));
%             end
            
            likelyhood = body * exp(expon);
            probability = likelyhood * prior;    
        end        
           
        function [confidence, is_orange] = calculate_confidence(training_data, training_length, rgb_vector, min_confidence)
            mu = gaussian.calculate_mean(training_data, training_length);
            sigma = gaussian.calculate_covariance(training_data, training_length, mu);
        
            max_probability = gaussian.calculate_probability(mu, sigma, mu);
            probability = gaussian.calculate_probability(mu, sigma, rgb_vector);
            
            confidence = probability/max_probability;
            if(confidence >= min_confidence)
                is_orange = true;
            else
                is_orange = false;
            end
            
        end
        
    end
end

