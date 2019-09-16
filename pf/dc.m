function [V_M, V_A, itr_et, fail] = dc(str, P, Q)
    define_constants;
    mpc = loadcase(str);
    
    V_M = ones(size(P), 'single');  % equivalent of Julia Float32
    V_A = zeros(size(P), 'single');
    
    numSample = size(P, 2);
    itr_et = zeros(numSample, 1);  % iteration elapsed time
    
    % arr for i-th failed dcpf
    fail = []; 
    
    % run acpf numSample times
    mpopt = mpoption('out.all', 0, 'verbose', 0);
    for i = 1:numSample
        if mod(i, 1000) == 0
            fprintf(' >> dcpf iteration %d\n', i);
        end
        mpc.bus(:, PD) = P(:, i);
        mpc.bus(:, QD) = Q(:, i);
        ret = rundcpf(mpc, mpopt);
        if ret.success == 1
            V_A(:, i) = ret.bus(:, VA);
            itr_et(i) = ret.et;
        else
            fail = [fail, i];
        end
    end
    assert(isempty(fail) == 1);
end