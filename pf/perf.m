function perf(str, mode)
    load(['./results/', str, '_perf_', mode, '.mat']);
    et_ac_avg = mean(et_ac);
    itr_cs_avg = mean(itr_ac);
    et_accs_std = std(et_ac);
    itr_cs_std = std(itr_ac);
    if mode == 'cs'  % hot-start won't have dc fields
        et_dc_avg = mean(et_dc);
        et_dc_std = std(et_dc);
    end
    
    fprintf('\n================ %s ================\n', str);
    if mode == 'cs'
        fprintf(' ACPF with cold start performance results:\n');
        fprintf(' >> iterations:              average = %.3f, std = %.3f\n', itr_cs_avg, itr_cs_std);
        fprintf(' >> elapsed time per sample: average = %.7f, std = %.7f\n', et_ac_avg, et_accs_std);
        fprintf(' DCPF performance results\n');
        fprintf(' >> elapsed time per sample: average = %.7f, std = %.7f\n', et_dc_avg, et_dc_std);
    elseif mode == 'hs'
        fprintf(' ACPF with hot start performance results:\n');
        fprintf(' >> iterations:              average = %.3f, std = %.3f\n', itr_cs_avg, itr_cs_std);
        fprintf(' >> elapsed time per sample: average = %.7f, std = %.7f\n', et_ac_avg, et_accs_std);
    end
    fprintf('\n');
end