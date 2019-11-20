function data_gen()
%     case300: 8955/10000 failed
%     case1354pegase: 38/10000 failed
%     case13659pegase: 7550/10000 failed 
%     cases = ["case30", "case89pegase", "case118", "case145", "case2869pegase", "case3375wp"];
%     cases = ["case300", "case1354pegase", "case13659pegase"];
    cases = ["case13659pegase"];
    N = 10000;

    for c = cases
        %% PF based on PQ variations
        c = char(c);  % weird matlab string format
        [P, Q, Pp, Qp] = pq_var(c, N);
        [DCVM, DCVA, et_dc] = dc(c, P, Q);  % fail_dc should be []
        [ACVM, ACVA, et_ac, itr_ac, fail, mismatch_cold] = ac_cold(c, P, Q);

        %% Error check and remove all failed sample columns, if they exist
        n_fail = length(fail);
        if n_fail ~= 0
            fP = P(:, fail);
            P(:, fail) = [];
            fQ = Q(:, fail);
            Q(:, fail) = [];
            Pp(:, fail) = [];
            Qp(:, fail) = [];
            ACVM(:, fail) = [];
            ACVA(:, fail) = [];
            DCVM(:, fail) = [];
            DCVA(:, fail) = [];
            fmismatch = mismatch_cold(:, fail);
            mismatch_cold(:, fail) = [];
            itr_ac(fail) = [];
            et_ac(fail) = [];
            fprintf(' %i samples failed. Number of samples in dataset now: %i\n', n_fail, size(P, 2));
            save(['./results/', c, '_failed.mat'], 'fP', 'fQ', 'fmismatch', 'fail');
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
        data(:, :, 3) = Pp;
        data(:, :, 4) = Qp;
        target = zeros([size(ACVM,1), size(ACVM,2), 2], 'single');
        target(:, :, 1) = ACVA;
        target(:, :, 2) = ACVM;   

        save(['./results/', c, '_dataset.mat'], 'data', 'target', 'P', 'Q');
        save(['./results/', c, '_perf_cold.mat'], 'mismatch_cold', 'itr_ac', 'et_ac', 'et_dc', 'fail'); %, 'norm_va', 'norm_vm');
        
        perf(c, 'cold', '');
    end
    fprintf('Data generation finished.\n\n');
end