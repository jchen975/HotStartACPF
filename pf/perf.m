function perf(case_str, mode, T)
    if strcmp(mode, 'cold')
        load(['./results/', case_str, '_perf_cold.mat']);
    else 
        load(['./results/', case_str, '_perf_hot_T=', T, '.mat']);
    end
    
    et_ac_per_sample = stat(et_ac);
    itr_per_sample = stat(itr_ac);
    if strcmp(mode, 'cold')  % hot-start won't have dc fields
        et_dc_per_sample = stat(et_dc);
    end
    
    fprintf('================ %s ================\n', case_str);
    if strcmp(mode, 'cold')
        fprintf(' ACPF with cold start performance results:\n');
        fprintf(' >> iterations:              avg = %.3f, std = %.3f, max = %i, min = %i\n', itr_per_sample(1), itr_per_sample(2), itr_per_sample(3), itr_per_sample(4));
        fprintf(' >> elapsed time per sample: avg = %.7f, std = %.7f, max = %.7f, min = %.7f\n', et_ac_per_sample(1), et_ac_per_sample(2), et_ac_per_sample(3), et_ac_per_sample(4));
        fprintf(' >> total elapsed time:            %.7f\n', sum(et_ac));
%         fprintf(' >> elapsed time per itr:    avg = %.7f (%i samples in total) \n', et_ac_per_itr, length(et_ac));
        fprintf(' \n DCPF performance results\n');
        fprintf(' >> elapsed time per sample: avg = %.7f, std = %.7f, max = %.7f, min = %.7f\n', et_dc_per_sample(1), et_dc_per_sample(2), et_dc_per_sample(3), et_dc_per_sample(4));
        fprintf(' >> total elapsed time: %.7f\n', sum(et_dc));
    elseif strcmp(mode, 'hot')
        fprintf(' ACPF with hot start performance results:\n');
        fprintf(' T = %s\n', T);
        fprintf(' >> iterations:                avg = %.3f, std = %.3f, max = %i, min = %i\n', itr_per_sample(1), itr_per_sample(2), itr_per_sample(3), itr_per_sample(4));
        fprintf(' >> elapsed time per sample:   avg = %.7f, std = %.7f, max = %.7f, min = %.7f\n', et_ac_per_sample(1), et_ac_per_sample(2), et_ac_per_sample(3), et_ac_per_sample(4));
        fprintf(' >> N\\T total elapsed time:          %.7f\n', sum(et_ac));
        fprintf(' >> extrapolated total elapsed time: %.7f\n', (sum(et_ac))/(1-str2num(T)));
    end
    fprintf('\n');
end

function ret = stat(arr)
    ret = [mean(arr), std(arr), max(arr), min(arr)];
end