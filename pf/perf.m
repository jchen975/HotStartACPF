function perf(str, mode, T, lambda)
    if mode == 'cold'
        load(['./results/', str, '_perf_cold.mat']);
    else 
        load(['./results/', str, '_perf_hot_', T, 'T_', lambda, 'lambda.mat']);
    end
    
    et_ac_per_sample = stat(et_ac);
    itr_per_sample = stat(itr_ac);
    if mode == 'cold'  % hot-start won't have dc fields
        et_dc_per_sample = stat(et_dc);
    end
    
    % average 1 iteration time for all N samples, compare this with forward
    % pass
    et_ac_per_itr = sum(et_ac ./ itr_ac);  
    
    fprintf('================ %s ================\n', str);
    if mode == 'cold'
        fprintf(' ACPF with cold start performance results:\n');
        fprintf(' >> iterations:              average = %.3f, std = %.3f, max = %i, min = %i\n', itr_per_sample(1), itr_per_sample(2), itr_per_sample(3), itr_per_sample(4));
        fprintf(' >> elapsed time per sample: average = %.7f, std = %.7f, max = %.7f, min = %.7f\n', et_ac_per_sample(1), et_ac_per_sample(2), et_ac_per_sample(3), et_ac_per_sample(4));
        fprintf(' >> elapsed time per itr:    average = %.7f (%i samples in total) \n', et_ac_per_itr, length(et_ac));
        fprintf(' \n DCPF performance results\n');
        fprintf(' >> elapsed time per sample: average = %.7f, std = %.7f, max = %.7f, min = %.7f\n', et_dc_per_sample(1), et_dc_per_sample(2), et_dc_per_sample(3), et_dc_per_sample(4));


    elseif mode == 'hot'
        fprintf(' ACPF with hot start performance results:\n');
        fprintf(' T = %s, lambda = %s\n', T, lambda);
        fprintf(' >> iterations:              average = %.3f, std = %.3f, max = %i, min = %i\n', itr_per_sample(1), itr_per_sample(2), itr_per_sample(3), itr_per_sample(4));
        fprintf(' >> elapsed time per sample: average = %.7f, std = %.7f, max = %.7f, min = %.7f\n', et_ac_per_sample(1), et_ac_per_sample(2), et_ac_sample(3), et_ac_sample(4));

    end
    fprintf('\n');
end

function ret = stat(arr)
    ret = [mean(arr), std(arr), max(arr), min(arr)];
end