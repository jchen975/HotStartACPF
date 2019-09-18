function perf(str, mode, T, lambda)
    if mode == 'cs'
        load(['./results/', str, '_perf_cs.mat']);
    else 
        load(['./results/', str, '_perf_hs_', T, 'T_', lambda, 'lambda.mat']);
    end
    et_ac_avg = mean(et_ac);
    et_ac_std = std(et_ac);
    
    itr_cs_avg = mean(itr_ac);
    itr_cs_std = std(itr_ac);
    
    et_ac_max = max(et_ac);
    et_ac_min = min(et_ac);
    
    itr_max = max(itr_ac);
    itr_min = min(itr_ac);
    
    if mode == 'cs'  % hot-start won't have dc fields
        et_dc_avg = mean(et_dc);
        et_dc_std = std(et_dc);
        et_dc_max = max(et_dc);
        et_dc_min = min(et_dc);
    end
    
    fprintf('================ %s ================\n', str);
    if mode == 'cs'
        fprintf(' ACPF with cold start performance results:\n');
        fprintf(' >> iterations:              average = %.3f, std = %.3f, max = %i, min = %i\n', itr_cs_avg, itr_cs_std, itr_max, itr_min);
        fprintf(' >> elapsed time per sample: average = %.7f, std = %.7f, max = %.7f, min = %.7f\n', et_ac_avg, et_ac_std, et_ac_max, et_ac_min);
        fprintf(' \n DCPF performance results\n');
        fprintf(' >> elapsed time per sample: average = %.7f, std = %.7f, max = %.7f, min = %.7f\n', et_dc_avg, et_dc_std, et_dc_max, et_dc_min);
        fprintf(' \n Norm of (AC - DC)\n');
        fprintf(' va: %.5f  vm: %.5f\n', norm_va, norm_vm);
    elseif mode == 'hs'
        fprintf(' ACPF with hot start performance results:\n');
        fprintf(' T = %s, lambda = %s\n', T, lambda);
        fprintf(' >> iterations:              average = %.3f, std = %.3f, max = %i, min = %i\n', itr_cs_avg, itr_cs_std, itr_max, itr_min);
        fprintf(' >> elapsed time per sample: average = %.7f, std = %.7f, max = %.7f, min = %.7f\n', et_ac_avg, et_ac_std, et_ac_max, et_ac_min);
        fprintf(' \n Norm of (AC - predicted (hot start))\n');
        fprintf(' va: %.5f  vm: %.5f\n', norm_va, norm_vm);
    end
    fprintf('\n');
end