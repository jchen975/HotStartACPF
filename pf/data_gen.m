function data_gen()
%     case300: 8955/10000 failed
%     case1354pegase: 38/10000 failed
%     case13659pegase: 7550/10000 failed 
    cases = ["case118", "case2869pegase"]; 
%     cases = ["case300", "case13659pegase"];
%     cases = ["case300"];
    N = 20000;
    
    for c = cases
        %% PF based on PQ variations
        c = char(c);  % weird matlab string format
        [P, Q, Pp, Qp] = pq_var(c, N);
        [DCVM, DCVA, et_dc] = dc(c, P, Q);  % fail_dc should be []
        [ACVM, ACVA, et_ac, itr_ac, fail, mismatch_cold] = ac_cold(c, P, Q, DCVA, DCVM);

        %% Error check and remove all failed sample columns, if they exist
        % move failed P, Q (in per unit difference) and DC vm, va results
        % to 'fTest', which has the same (numBus, nChannel) dimension as 
        % training/val set, which contains **all** the successful samples
        n_fail = length(fail);
        if n_fail ~= 0
            fP = P(:, fail);
            P(:, fail) = [];
            fQ = Q(:, fail);
            Q(:, fail) = [];
            fPp = Pp(:, fail);
            Pp(:, fail) = [];
            fQp = Qp(:, fail);
            Qp(:, fail) = [];
            ACVM(:, fail) = [];
            ACVA(:, fail) = [];
            fDCVM = DCVM(:, fail);
            DCVM(:, fail) = [];
            fDCVA = DCVA(:, fail);
            DCVA(:, fail) = [];
            fmismatch = mismatch_cold(:, fail);
            mismatch_cold(:, fail) = [];
            itr_ac(fail) = [];
            et_ac(fail) = [];
            fprintf(' %i samples failed. Number of samples in dataset now: %i\n', n_fail, size(P, 2));
            fdata = zeros([size(fDCVM,1), size(fDCVM,2), 4], 'single');
            fdata(:, :, 1) = fDCVA;
            fdata(:, :, 2) = fDCVM;  % include this for PV bus vm
            fdata(:, :, 3) = fPp;
            fdata(:, :, 4) = fQp;
            save(['./results/', c, '_failed.mat'], 'fP', 'fQ', 'fdata', 'fmismatch', 'fail');
        end

        %% form final dataset for case c and save to files
        DCVA = deg2rad(DCVA);  % make DCVA, ACVA in radians for num stability
        ACVA = deg2rad(ACVA);
        
        % Shift AC, DC vm mean down by 1.0 to have mean (close to) 0.0
        DCVM = DCVM - 1.0;
        ACVM = ACVM - 1.0;
        
        data = zeros([size(DCVM,1), size(DCVM,2), 4], 'single');
        data(:, :, 1) = DCVA;
        data(:, :, 2) = DCVM;  % include this for PV bus vm
%         data(:, :, 1) = zeros(size(DCVM));
%         data(:, :, 2) = zeros(size(DCVA));
        data(:, :, 3) = Pp;
        data(:, :, 4) = Qp;
        target = zeros([size(ACVM,1), size(ACVM,2), 2], 'single');
        target(:, :, 1) = ACVA;
        target(:, :, 2) = ACVM;   

        save(['./results/', c, '_dataset.mat'], 'data', 'target', 'P', 'Q');
        save(['./results/', c, '_perf_cold.mat'], 'mismatch_cold', 'itr_ac', 'et_ac', 'et_dc', 'fail'); %, 'norm_va', 'norm_vm');
        
    end
        perf(c, 'cold', '');
    fprintf('Data generation finished.\n\n');
end