function init_sol(str)
    mpopt = mpoption('out.all', 0, 'verbose', 0);
    mpc = runpf(str, mpopt);
    if mpc.iterations ~= 1
        savecase(['./data/', str], mpc);
        mpc2 = loadcase(str);
        ret = runpf(mpc2, mpopt);
        assert(ret.iterations == 1);
    end
end