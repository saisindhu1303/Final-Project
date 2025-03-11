function loss = Vpuloss(inp, target, gamma, alpha)
    positive = 1;
    negative = -1;
    unlabeled = 0;
    
    output_mean = mean(inp, 2);
    std = alpha * sqrt(sum((inp - output_mean).^2, 2) / size(inp, 2));
    std = min(std, 1);
    std = std';
    
    % calculate the variational loss
    positive_mask = target == positive;
    unlabeled_mask = ones(size(positive_mask));
    positive_mask = double(positive_mask);
    unlabeled_mask = double(unlabeled_mask);
    
    num_p_rec = 1 ./ sum(positive_mask, 1)';
    num_x_rec = 1 ./ sum(unlabeled_mask, 1)';
    num_p_rec(isinf(num_p_rec)) = 0;
    num_x_rec(isinf(num_x_rec)) = 0;
    
    % sigmoid_log
    output_all_log = log(sigmoid(inp ./ std) + 1e-10);
    output_p_log = output_all_log .* positive_mask;
    output_x_log = output_all_log .* unlabeled_mask;
    
    output_all_sig = sigmoid(inp ./ std);
    pt1 = sum((1 - output_all_sig) .* unlabeled_mask, 1)' .* num_x_rec;
    one_sided_w_n = (1 - pt1).^gamma;
    
    p_loss = sum(log(exp(output_p_log) + 1e-10) .* num_p_rec);
    u_loss = sum(log(sum(exp(output_x_log), 1) .* num_x_rec + 1e-10) .* one_sided_w_n);
    var_loss = u_loss - p_loss;
    
    loss = var_loss;
end
function loss = mixup(model, inputData, target, output, mix_alpha)
    data_x1 = inputData(1:floor(size(inputData, 1)/2), :, :, :);
    data_x2 = inputData(ceil(size(inputData, 1)/2):end, :, :, :);
    target_x1 = target(1:floor(size(target, 1)/2), :);
    target_x2 = target(ceil(size(target, 1)/2):end, :);
    output_all_log = log(sigmoid(output) + 1e-10);
    output_x1_log = output_all_log(1:floor(size(output_all_log, 1)/2), :);
    output_x2_log = output_all_log(ceil(size(output_all_log, 1)/2):end, :);

    if size(data_x2, 1) == 0
        data_x2 = data_x1;
        target_x2 = target_x1;
        output_x2_log = output_x1_log;
    end

    positive_x1 = target_x1 == 1;
    unlabeled_x1_1 = target_x1 == 0;
    unlabeled_x1_2 = target_x1 == -1;
    unlabeled_x1 = unlabeled_x1_1 + unlabeled_x1_2;
    positive_x1 = double(positive_x1);
    unlabeled_x1 = double(unlabeled_x1);
    output_x1 = positive_x1 + exp(output_x1_log) .* unlabeled_x1;

    rand_perm = randperm(size(positive_x1, 1));
    data_x1_perm = data_x1(rand_perm, :, :, :);
    output_x1_perm = output_x1(rand_perm, :);

    positive_x2 = target_x2 == 1;
    unlabeled_x2_1 = target_x2 == 0;
    unlabeled_x2_2 = target_x2 == -1;
    unlabeled_x2 = unlabeled_x2_1 + unlabeled_x2_2;
    positive_x2 = double(positive_x2);
    unlabeled_x2 = double(unlabeled_x2);
    output_x2 = positive_x2 + exp(output_x2_log) .* unlabeled_x2;

    m = betarnd(mix_alpha, mix_alpha, 1, 1);
    lam = m;
    output_x1_perm_lam = output_x1_perm .* lam;
    data_x1_perm_lam = data_x1_perm .* lam;
    output_x2_lam = output_x2 .* (1 - lam);
    data_x2_lam = data_x2 .* (1 - lam);

    data = gpuArray((data_x1_perm_lam + data_x2_lam));
    output_mix = gpuArray((output_x1_perm_lam + output_x2_lam));

    out_all = model(data);
    out_all = single(out_all);
    out_all_log = log(sigmoid(out_all) + 1e-10);

    reg_loss = sum(sum((log(output_mix + 1e-10) - out_all_log).^2)) / size(data_x1, 1);

    loss = reg_loss;
end
