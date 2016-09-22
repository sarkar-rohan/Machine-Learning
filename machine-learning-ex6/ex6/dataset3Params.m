function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;
steps = [0.01;0.03;0.1;0.3;1;3;10;30]
c_sigma = [];
errors =[];
for i =1:size(steps)
    for j =1:size(steps)
        c_sigma = [c_sigma;[steps(i) steps(j)]]
    end
end

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
for i = 1:size(c_sigma)
    model = svmTrain(X, y, c_sigma(i,1), @(x1, x2) gaussianKernel(x1, x2, c_sigma(i,2)));
    predictions = svmPredict(model, Xval);
    error = mean(double(predictions ~= yval));
    errors = [errors;error];
end 
[min_value min_index] = min(errors);
C = c_sigma(min_index,1)
sigma = c_sigma(min_index,2)




% =========================================================================

end
