

% first we want to load in the images
% training_data = dir('train_images');
% [confidence, isOrange] = gaussian.calculate_confidence(training_data, 16,[215;117;119], 0.7);

% overlap the mask on the image
% maskedRgbImage = bsxfun(@times, img, cast(mask168, 'like', img));
% imshow(maskedRgbImage);
classdef trainGMM
    methods (Static)
        function [means, weights, covariances, red, green, blue] = initialize(training_data, training_length, num_gaussians)
            [mu, N] = gaussian.calculate_mean(training_data, training_length);
            sigma = gaussian.calculate_covariance(training_data, training_length, mu);
            
% %           prealocate the rgb arrays
            red = ones(N,1);
            green = ones(N,1);
            blue = ones(N,1);
            rgbIdx = 1;
            
%           create a num_gaussians x 1 dimensional vector to hold the weights for each gaussian  
            weights = ones(num_gaussians,1);
            means = mu;
            covariances = sigma;
            
            for i=2:num_gaussians
%               create a num_gaussians dimensional array to hold the mean
%               vector and the covariance matrix for each gaussian
                means(:,:,i) =  mu;
                covariances(:,:,i) = sigma;
                weights(i) = rand;
            end  
            
%             get all the masked pixels
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
                    red(rgbIdx) = redValues(j);
                    green(rgbIdx) = greenValues(j);
                    blue(rgbIdx) = blueValues(j);
                    rgbIdx = rgbIdx + 1;
                end
             end
        end       
        
        function alphas = calculate_cluster_weights(means, weights, covariances, red, green, blue, num_gaussians)
            alphas = ones(size(red,1),1);
            
            for i=2:num_gaussians
%               initialize the alphas array
                alphas(:,:,i) = ones(size(red,1),1);
            end  
            
            for i=1:size(weights,1)
                
%                 get the apporopriate info from dimension i
                weight = weights(i);
                mu = means(:,:,i);
                sigma = covariances(:,:,i);
                
                for j=1:size(alphas,1)
                    total = 0;
                    
                    rgb_vector = [
                        red(j);
                        green(j);
                        blue(j)
                    ];
                
%                     get the likelyhood of the pixel
                    [~, likelyhood] = gaussian.calculate_probability(mu, sigma, rgb_vector);
                   
%                     get the likelyhood of the point in all the gaussians
                    for k=1:size(weights,1)
                        [~, likelyhood2] = gaussian.calculate_probability(means(:,:,k), covariances(:,:,k), rgb_vector);
                        total = total + (weights(k) * likelyhood2);
                    end
                    
%                     set the cluster weight
                    alphas(j,1,i) = (weight * likelyhood)/total;
                end    
            end
        end
        
        function [means, previous_means, weights, covariances] = calculate_parameters(training_data, training_length, alphas, prev_means)
            
            [means, weights, covariances, red, green, blue] = trainGMM.initialize(training_data, training_length, size(alphas,3));
            
            previous_means = prev_means;
            
            for i=1:size(weights,1)
                mu = 0;
                sigma = 0;
                total = sum(alphas(:,1,i));
                N = size(alphas, 1);
                weight = total/N;
                
%                 calculate the sum of alpha_i,j*x_j
                for j=1:size(alphas,1)
                    rgb_vector = [
                        red(j);
                        green(j);
                        blue(j)
                    ];
                    
                    mu = mu + (alphas(j,1,i) * rgb_vector);
                end
                
%                  calculate mu
                 mu = double(floor((mu/total)));
                 
%                 calculate the sum of alpha_i,j * (x_j - mu_i) * (x_j - mu_i)T
                 for j=1:size(alphas,1)
                    rgb_vector = [
                        red(j);
                        green(j);
                        blue(j)
                    ];

                    sigma = sigma + (alphas(j,1,i) * (rgb_vector - mu) * (rgb_vector - mu)');
                 end
%                  calculate sigma
                 sigma = sigma/total;
               
                 weights(i) = weight;
                 means(:,:,i) = mu;
                 covariances(:,:,i) = sigma;
                
            end
        end
        
        function n_sum = norm_sum(means, previous_means)
            n_sum = 0;
            
            for i=1:size(means,3)
                vector = means(:,:,i) - previous_means(:,:,i);
                n_sum = n_sum + norm(vector);
            end
        end
        
        function [m, p, w, c, a] = train(training_data, training_length, num_gaussians, max_iterations, convergence_threshold)
            [means, weights, covariances, red, green, blue] = trainGMM.initialize(training_data, training_length, num_gaussians);
            m = means;
            p = means * 0;
            w = weights;
            c = covariances;
            
            i = 1;
            while i <= max_iterations && trainGMM.norm_sum(m,p) > convergence_threshold
                alphas = trainGMM.calculate_cluster_weights(m, w, c, red, green, blue, num_gaussians);
                a = alphas;
                
                [means, previous_means, weights, covariances] = trainGMM.calculate_parameters(training_data, training_length, a, p);
                m = means;
                p = previous_means;
                w = weights;
                c = covariances;
                
                i = i+1;
            end
        end
        
        
    end
    
    
end