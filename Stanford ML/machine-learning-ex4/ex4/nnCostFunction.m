function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%%%% Part 1

a1 = [ones(m, 1), X];  % add bias unit for X, now size(X) = 5000 * (400 + 1)
% Theta1 size : 25 * 401
% Theta2 size : 10 * 26
a2 = sigmoid(a1 * Theta1');  % size a2 = (5000*401) * (401 * 25)  ---> 5000 * 25
% add bias unit for a2
a2 = [ones(m, 1), a2];   % now size a2 = 5000 * 26
% compute a3
a3 = sigmoid(a2 * Theta2');  % a3 size : (5000 * 26) * (26 * 10) ---> 5000 * 10
Hx = a3; % Hx size 5000 * 10

for i=1:num_labels;  
    ynew(:,i) = y==i; %5000x10  
end  
% compute regularization
newTheta1 = Theta1(:, 2:end);  % newTheta1 size: 25 * 400
newTheta2 = Theta2(:, 2:end);  % newTheta2 size： 10 * 25
reg = lambda*(sum(sum(newTheta2 .^2)) + sum(sum(newTheta1 .^2)))/(2*m);
% compute J(theta)
J = sum(sum((-ynew.*log(Hx)) - (1-ynew).*log(1-Hx)))/m ;
J = J + reg;



%%%% Part 2

% compute delta
delta1 = 0;
delta2 = 0;

for i=1:m
%step 1
    a1 = [1, X(i,:)]; % 1 * 401
    z2 = a1 * Theta1'; % 1 * 25
    a2 = [1, sigmoid(z2)];  % 1 * 26
    z3 = a2 * Theta2';      % 1 * 10
    a3 = sigmoid(z3);       % 1 * 10
%step 2
    yt = ynew(i, :);        % 1 * 10
    deltaerror3 = a3 - yt;        % 1 * 10
%step 3        compute error term
    deltaerror2 =  (deltaerror3 * Theta2)(2:end) .* sigmoidGradient(z2);   % (1 * 10) * (10 * 25) .* (1 * 25)    = 1 * 25
%step 4
    delta1 = delta1 + deltaerror2' * a1;  % (25*1) * (1 * 401)  = 25 * 401;
    delta2 = delta2 + deltaerror3' * a2;  % (10*1) * (1 * 26)   = 10 * 26;
end
%step 5
Theta1_grad = delta1/m;             % 25 * 401   
Theta2_grad = delta2/m;             % 10 * 26



%%%% Part 3
m1 = size(newTheta1, 1);
m2 = size(newTheta2, 1);
Theta1_grad = Theta1_grad + (lambda/m) * [zeros(m1, 1), newTheta1];       % 25 * 401
Theta2_grad = Theta2_grad + (lambda/m) * [zeros(m2, 1), newTheta2];       % 10 * 26

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
