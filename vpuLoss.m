% vpuLoss.m
classdef vpuLoss < handle
    properties
        gamma
        alpha
        positive = 1
        negative = -1
        unlabeled = 0
    end

    methods %calls the objectives of vpu loss class
        function obj = vpuLoss(gamma,alpha)
            obj.gamma = gamma;%assigns value
            obj.alpha = alpha;
        end

        function var_loss = forward(obj, inp, target)%calculates loss value
            % Calculate mean for each feature dimension
            output_mean = mean(inp, 1);

            % Calculate standard deviation to indicate that model fits data
            % well
            std = obj.alpha * sqrt(sum((inp - output_mean).^2, 1) ./ size(inp, 1));
            std = min(std, 1); % limit std to a maximum of 1
            std = gather(std); % Convert to CPU for calculation likely for gpu/cpu split

            % Create boolean masks
            positive = target == obj.positive;
            unlabeled = ones(size(positive));

            % Convert to double
            positive = double(positive);
            unlabeled = double(unlabeled);

            % Calculate normalization factors
            num_p_rec = 1 ./ sum(positive, 1);
            num_x_rec = 1 ./ sum(unlabeled, 1);

            % Handle potential division by zero
            num_p_rec(isinf(num_p_rec)) = 0;
            num_x_rec(isinf(num_x_rec)) = 0;

            % Calculate sigmoid_log
            output_all_log = log(sigmoid(inp .* (1 ./ std)) + 1e-10); % Element-wise division
            output_p_log = output_all_log .* positive;
            output_x_log = output_all_log .* unlabeled;

            output_all_sig = sigmoid(inp .* (1 ./ std));

         