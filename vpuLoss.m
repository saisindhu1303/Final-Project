function loss = Vpuloss(inp, target, gamma, alpha)
    % Define constants for positive, negative, and unlabeled classes
    positive = 1;   % Value representing the positive class
    negative = -1;  % Value representing the negative class
    unlabeled = 0;  % Value representing the unlabeled class
    
    % Calculate the mean and standard deviation of inputs
    output_mean = mean(inp, 2); % Mean of inputs across the second dimension
    std = alpha * sqrt(sum((inp - output_mean).^2, 2) / size(inp, 2)); % Calculate standard deviation
    std = min(std, 1); % Clamp standard deviation to a maximum of 1
    std = std'; % Transpose to match dimensions

    % Create masks for positive and unlabeled samples
    positive_mask = target == positive; % Mask for positive samples (1 if positive, 0 otherwise)
    unlabeled_mask = ones(size(positive_mask)); % Mask for unlabeled samples (all ones)
    positive_mask = double(positive_mask); % Convert mask to double format
    unlabeled_mask = double(unlabeled_mask); % Convert mask to double format

    % Calculate the reciprocal counts for positive and unlabeled masks
    num_p_rec = 1 ./ sum(positive_mask, 1)'; % Count of positive examples (inverse)
    num_x_rec = 1 ./ sum(unlabeled_mask, 1)'; % Count of unlabeled examples (inverse)

    % Handle infinity cases by replacing with zeros
    num_p_rec(isinf(num_p_rec)) = 0; 
    num_x_rec(isinf(num_x_rec)) = 0; 

    % Calculate the sigmoid log probabilities
    output_all_log = log(sigmoid(inp ./ std) + 1e-10); % Log of sigmoid output for all inputs
    output_p_log = output_all_log .* positive_mask;      % Log probabilities for positive samples
    output_x_log = output_all_log .* unlabeled_mask;     % Log probabilities for unlabeled samples

    % Calculate sigmoid values for the inputs
    output_all_sig = sigmoid(inp ./ std); 

    % Compute the one-sided weights for unlabeled data
    pt1 = sum((1 - output_all_sig) .* unlabeled_mask, 1)' .* num_x_rec; % Weight calculation
    one_sided_w_n = (1 - pt1).^gamma; % Adjust weights by exponentiating

    % Calculate positive loss and unlabeled loss
    p_loss = sum(log(exp(output_p_log) + 1e-10) .* num_p_rec); % Positive log likelihood loss
    u_loss = sum(log(sum(exp(output_x_log), 1) .* num_x_rec + 1e-10) .* one_sided_w_n); % Unlabeled log likelihood loss

    % Calculate the overall variational loss
    var_loss = u_loss - p_loss; 

    % Return the final loss
    loss = var_loss; 
end

function loss = mixup(model, inputData, target, output, mix_alpha)
    % Split input data and target for mixing
    data_x1 = inputData(1:floor(size(inputData, 1)/2), :, :, :); % First half of the data
    data_x2 = inputData(ceil(size(inputData, 1)/2):end, :, :, :); % Second half of the data
    target_x1 = target(1:floor(size(target, 1)/2), :); % First half of the targets
    target_x2 = target(ceil(size(target, 1)/2):end, :); % Second half of the targets
    output_all_log = log(sigmoid(output) + 1e-10); % Calculate log probabilities of outputs
    output_x1_log = output_all_log(1:floor(size(output_all_log, 1)/2), :); % Log probabilities for first half
    output_x2_log = output_all_log(ceil(size(output_all_log, 1)/2):end, :); % Log probabilities for second half

    % Handle case where data_x2 is empty
    if size(data_x2, 1) == 0
        data_x2 = data_x1; % Copy over data from x1 to x2
        target_x2 = target_x1; % Copy over targets from x1 to x2
        output_x2_log = output_x1_log; % Copy over logs from x1 to x2
    end

    % Create masks for positive and unlabeled classes
    positive_x1 = target_x1 == 1; % Mask for positive samples in x1
    unlabeled_x1_1 = target_x1 == 0; % Mask for unlabeled samples (class 0) in x1
    unlabeled_x1_2 = target_x1 == -1; % Mask for unlabeled samples (class -1) in x1
    unlabeled_x1 = unlabeled_x1_1 + unlabeled_x1_2; % Combined unlabeled mask for x1
    positive_x1 = double(positive_x1); % Convert mask to double
    unlabeled_x1 = double(unlabeled_x1); % Convert mask to double
    output_x1 = positive_x1 + exp(output_x1_log) .* unlabeled_x1; % Final output for x1

    % Permutate the data and outputs
    rand_perm = randperm(size(positive_x1, 1)); % Generate a random permutation
    data_x1_perm = data_x1(rand_perm, :, :, :); % Permuted data for x1
    output_x1_perm = output_x1(rand_perm, :); % Permuted output for x1

    % Create masks for positive and unlabeled classes for x2
    positive_x2 = target_x2 == 1; % Mask for positive samples in x2
    unlabeled_x2_1 = target_x2 == 0; % Mask for unlabeled samples (class 0) in x2
    unlabeled_x2_2 = target_x2 == -1; % Mask for unlabeled samples (class -1) in x2
    unlabeled_x2 = unlabeled_x2_1 + unlabeled_x2_2; % Combined unlabeled mask for x2
    positive_x2 = double(positive_x2); % Convert mask to double
    unlabeled_x2 = double(unlabeled_x2); % Convert mask to double
    output_x2 = positive_x2 + exp(output_x2_log) .* unlabeled_x2; % Final output for x2

    % Generate a random mixing coefficient
    m = betarnd(mix_alpha, mix_alpha, 1, 1); % Draw a sample from the beta distribution
    lam = m; % Mixing coefficient
    output_x1_perm_lam = output_x1_perm .* lam; % Scale x1's outputs
    data_x1_perm_lam = data_x1_perm .* lam; % Scale x1's data
    output_x2_lam = output_x2 .* (1 - lam); % Scale x2's outputs
    data_x2_lam = data_x2 .* (1 - lam); % Scale x2's data

    % Mix the permuted and scaled datasets
    data = gpuArray((data_x1_perm_lam + data_x2_lam)); % Combined data on GPU
    output_mix = gpuArray((output_x1_perm_lam + output_x2_lam)); % Combined outputs on GPU

    % Forward pass through the model with the mixed data
    out_all = model(data); % Model prediction
    out_all = single(out_all); % Convert to single precision
    out_all_log = log(sigmoid(out_all) + 1e-10); % Calculate log probabilities of the mixed outputs

    % Calculate the regularization loss
    reg_loss = sum(sum((log(output_mix + 1e-10) - out_all_log).^2)) / size(data_x1, 1); % Mean squared error loss

    % Return the final loss
    loss = reg_loss; 
end
