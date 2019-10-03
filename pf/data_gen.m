function data_gen()
%     case300: 8955/10000 failed
%     case1354pegase: 38/10000 failed
%     case13659pegase: 7550/10000 failed 
    cases = ["case30", "case89pegase", "case118", "case145", "case2869pegase"];
    N = 10000;

    for c = cases
        %% PF based on PQ variations
        c = char(c);  % weird matlab string format
        [P, Q, Qp] = pq_var(c, N);
        [DCVM, DCVA, et_dc] = dc(c, P, Q);  % fail_dc should be []
        [ACVM, ACVA, et_ac, itr_ac, fail, mismatch] = ac_cs(c, P, Q);

        %% Error check and remove all failed sample columns, if they exist
        n_fail = length(fail);
        if n_fail ~= 0
            P(:, fail) = [];
            Q(:, fail) = [];
            ACVM(:, fail) = [];
            ACVA(:, fail) = [];
            DCVM(:, fail) = [];
            DCVA(:, fail) = [];
            mismatch(:, fail) = [];
            itr_ac(fail) = [];
            et_ac(fail) = [];
            fprintf(' %i samples failed. Number of samples in dataset now: %i\n', n_fail, size(P, 2));
        end

        %% form final dataset for case c and save to files
        DCVA = deg2rad(DCVA);  % make DCVA, ACVA in radians for num stability
        ACVA = deg2rad(ACVA);
        
        fprintf("Avg one iteration time: %.5f\n", mean(et_ac ./ itr_ac) * N);
        
        % Shift AC, DC vm mean down by 1.0 to have (near) mean 0.0
        DCVM = DCVM - 1.0;
        ACVM = ACVM - 1.0;
        
        % calculate L2 norm between AC and DC voltage phasors
        norm_va = mean(vecnorm(ACVA - DCVA));
        norm_vm = mean(vecnorm(ACVM - DCVM)); 
        
        data = [DCVA; DCVM, Qp];
        target = [ACVA; ACVM];
        
        assert(size(data, 1) == 2*size(P, 1) && size(data, 2) == size(P, 2));
        assert(size(target, 1) == 2*size(P, 1) && size(target, 2) == size(P, 2));

%         save(['./results/', c, '_dataset.mat'], 'data', 'target', 'diff', 'P', 'Q');
%         save(['./results/', c, '_perf_cs.mat'], 'mismatch', 'itr_ac', 'et_ac', 'et_dc', 'fail', 'norm_va', 'norm_vm');
        
%         perf(c, 'cs', '', '');
    end
    fprintf('Data generation finished.\n\n');
end