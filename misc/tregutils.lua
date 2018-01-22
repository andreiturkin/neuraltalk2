--
--
grad = require 'autograd'
local tregutils = {}
tregutils.add_to_grad_params = nil

--    LSTM weights:
--    W is a weight for the input,
--    U is a weight for the recurrent connection.
--    lambdaval is the regularization parameter
function tregutils.treg(LSTMweights)

    --    rho_x = sqrt(2)*gamma_x + (2/u)(gamma*gamma_x*|W_oh| + |W_ox|)
    --    rho_h = sqrt(2)*gamma_h + (sqrt(2)/u)*gamma*gamma_h*norm_W_oh
    --    u = exp(1 - gamma*|W_oh|),
    --    invu = exp(gamma*|W_oh| - 1),
    --    gamma_f <= 1/4,
    --    gamma <= 17/16,
    --    gamma_i_ci <= 17/16,
    --    gamma_x = gamma_f*|W_fx| + gamma_i_ci*(|W_ix| + |W_cix|),
    --    gamma_h = gamma_f*|W_fh| + gamma_i_ci*(|W_ih| + |W_cih|).

--    local zero = torch.Tensor({0.0}):cuda()
--    local one = torch.Tensor({1.0}):cuda()
--    local two = torch.Tensor({2.0}):cuda()
--    local sqrt2 = torch.sqrt(two):cuda()
--    local gamma_f = torch.Tensor({0.25}):cuda()
--    local gamma_i_ci = torch.Tensor({17.0/16.0}):cuda()
--    local gamma =  torch.Tensor({17.0/16.0}):cuda()
--    local lambdaval = torch.Tensor({0.0005}):cuda()

    local zero = torch.Tensor({0.0}):cuda()
    local one = torch.Tensor({1.0}):cuda()
    local two = torch.Tensor({2.0}):cuda()
    local sqrt2 = torch.sqrt(two):cuda()
    local gamma_f = torch.Tensor({0.25}):cuda()
    local gamma_i_ci = torch.Tensor({17.0/16.0}):cuda()
    local gamma =  torch.Tensor({17.0/16.0}):cuda()
    local lambdaval = torch.Tensor({0.05}):cuda()

    local norm_W_ix = torch.sqrt(torch.sum(torch.pow(LSTMweights['W_i'], 2),1))
    local norm_W_ih = torch.sqrt(torch.sum(torch.pow(LSTMweights['U_i'], 2),1))

    local norm_W_fx = torch.sqrt(torch.sum(torch.pow(LSTMweights['W_f'], 2),1))
    local norm_W_fh = torch.sqrt(torch.sum(torch.pow(LSTMweights['U_f'], 2),1))

    local norm_W_cix = torch.sqrt(torch.sum(torch.pow(LSTMweights['W_c'], 2),1))
    local norm_W_cih = torch.sqrt(torch.sum(torch.pow(LSTMweights['U_c'], 2),1))

    local norm_W_ox = torch.sqrt(torch.sum(torch.pow(LSTMweights['W_o'], 2),1))
    local norm_W_oh = torch.sqrt(torch.sum(torch.pow(LSTMweights['U_o'], 2),1))

    local gamma_x = torch.cmul(gamma_f,norm_W_fx) + torch.cmul(gamma_i_ci,(norm_W_ix + norm_W_cix))
    local gamma_h = torch.cmul(gamma_f,norm_W_fh) + torch.cmul(gamma_i_ci,(norm_W_ih + norm_W_cih))

    local invu = torch.exp(torch.cmul(gamma,norm_W_oh) - one)

    local rho_x_2 = torch.pow(torch.cmul(sqrt2,gamma_x) +
            torch.cmul(torch.cmul(two,invu), torch.cmul(torch.cmul(gamma,gamma_x),norm_W_oh) + norm_W_ox), 2)
    local rho_h_2 = torch.pow(torch.cmul(sqrt2,gamma_h) +
            torch.cmul(torch.cmul(torch.cmul(torch.cmul(sqrt2,invu),gamma),gamma_h),norm_W_oh), 2)

    if lambdaval == 0.0 then
        return 0.0
    else
        return torch.sum(torch.cmul(lambdaval,torch.cdiv(rho_x_2, (one - rho_h_2))))
--        return torch.sum(torch.cmul(lambdaval,torch.cdiv(rho_x_2, (one - rho_h_2))) +
--                0.0005*torch.cmax(zero, rho_h_2 - one) +
--                0.0005*torch.cmax(zero, gamma*norm_W_oh - one))
    end
end

function tregutils.getparameters(params, opt)
    local LSTMparam = {}
    -- LSTM W
    local LSTMparam1 = params:narrow(1, 1, 4*opt.input_encoding_size*opt.rnn_size)
    local W_i = LSTMparam1:narrow(1, 1, opt.input_encoding_size*opt.rnn_size)
    local W_f = LSTMparam1:narrow(1, opt.input_encoding_size*opt.rnn_size + 1, opt.input_encoding_size*opt.rnn_size)
    local W_o = LSTMparam1:narrow(1, 2*opt.input_encoding_size*opt.rnn_size + 1, opt.input_encoding_size*opt.rnn_size)
    local W_c = LSTMparam1:narrow(1, 3*opt.input_encoding_size*opt.rnn_size + 1, opt.input_encoding_size*opt.rnn_size)
    LSTMparam['W_i'] = W_i
    LSTMparam['W_f'] = W_f
    LSTMparam['W_o'] = W_o
    LSTMparam['W_c'] = W_c

    -- LSTM W bias (we don't need it)
    --LSTMparam2 = params:narrow(1, 2048*512+1, 2048)

    -- LSTM U
    local LSTMparam3 = params:narrow(1, 4*opt.input_encoding_size*opt.rnn_size + 4*opt.rnn_size + 1, 4*opt.rnn_size*opt.rnn_size)
    local U_i = LSTMparam3:narrow(1, 1, opt.rnn_size*opt.rnn_size)
    local U_f = LSTMparam3:narrow(1, opt.rnn_size*opt.rnn_size + 1, opt.rnn_size*opt.rnn_size)
    local U_o = LSTMparam3:narrow(1, 2*opt.rnn_size*opt.rnn_size + 1, opt.rnn_size*opt.rnn_size)
    local U_c = LSTMparam3:narrow(1, 3*opt.rnn_size*opt.rnn_size + 1, opt.rnn_size*opt.rnn_size)
    LSTMparam['U_i'] = U_i
    LSTMparam['U_f'] = U_f
    LSTMparam['U_o'] = U_o
    LSTMparam['U_c'] = U_c
    -- LSTM U bias (we don't need it)
    --LSTMparam4 = params:narrow(1, 2048*512+2048+2048*512+1, 2048)
    --
    --LSTM output + bias (we don't need it)
    --LSTMparam5 = params:narrow(1, 2048*512+2048+2048*512+2048+1, 9568*512)
    --LSTMparam6 = params:narrow(1, 2048*512+2048+2048*512+2048+9568*512+1, 9568)

    --print('LSTM parameters from params:')
    --print(LSTMparam)
    return LSTMparam
end

function tregutils.checking(thin_lm, LSTMparam, opt)
    -- Getting Language Model Parameters
    local p1,_ = thin_lm.core:parameters()
    --W_size = (ipt_size) x (4 * rnn_size)
    --U_size = (rnn_size) x (4 * rnn_size)
    local LSTMparam_layer = {}
    for i, v in pairs(p1) do
        if i == 1 then
            local W_i = v:sub(1,opt.rnn_size)
            LSTMparam_layer['W_i'] = W_i
            local W_f = v:sub(opt.rnn_size+1,2*opt.rnn_size)
            LSTMparam_layer['W_f'] = W_f
            local W_o = v:sub(2*opt.rnn_size+1,3*opt.rnn_size)
            LSTMparam_layer['W_o'] = W_o
            local W_c = v:sub(3*opt.rnn_size+1,4*opt.rnn_size)
            LSTMparam_layer['W_c'] = W_c
        end
        if i == 3 then
            local U_i = v:sub(1,opt.rnn_size)
            LSTMparam_layer['U_i'] = U_i
            local U_f = v:sub(opt.rnn_size+1,2*opt.rnn_size)
            LSTMparam_layer['U_f'] = U_f
            local U_o = v:sub(2*opt.rnn_size+1,3*opt.rnn_size)
            LSTMparam_layer['U_o'] = U_o
            local U_c = v:sub(3*opt.rnn_size+1,4*opt.rnn_size)
            LSTMparam_layer['U_c'] = U_c
        end
    end

    --print('LSTM parameters from the layer:')
    --print(LSTMparam_layer)

    ---- W
    --print('Check Ws')
    --print(torch.all(LSTMparam_layer['W_i']:eq(LSTMparam['W_i']))and
    --torch.all(LSTMparam_layer['W_f']:eq(LSTMparam['W_f']))and
    --torch.all(LSTMparam_layer['W_o']:eq(LSTMparam['W_o']))and
    --torch.all(LSTMparam_layer['W_c']:eq(LSTMparam['W_c'])))
    assert(torch.all(LSTMparam_layer['W_i']:eq(LSTMparam['W_i']))and
            torch.all(LSTMparam_layer['W_f']:eq(LSTMparam['W_f']))and
            torch.all(LSTMparam_layer['W_o']:eq(LSTMparam['W_o']))and
            torch.all(LSTMparam_layer['W_c']:eq(LSTMparam['W_c'])))
    ---- U
    --print ('Check Us')
    --print(torch.all(LSTMparam_layer['U_i']:eq(LSTMparam['U_i']))and
    --torch.all(LSTMparam_layer['U_f']:eq(LSTMparam['U_f']))and
    --torch.all(LSTMparam_layer['U_o']:eq(LSTMparam['U_o']))and
    --torch.all(LSTMparam_layer['U_c']:eq(LSTMparam['U_c'])))
    assert(torch.all(LSTMparam_layer['U_i']:eq(LSTMparam['U_i']))and
            torch.all(LSTMparam_layer['U_f']:eq(LSTMparam['U_f']))and
            torch.all(LSTMparam_layer['U_o']:eq(LSTMparam['U_o']))and
            torch.all(LSTMparam_layer['U_c']:eq(LSTMparam['U_c'])))

end

function tregutils.getgradparams(params, grad_LSTMparams, opt)
    if tregutils.add_to_grad_params == nil then
        tregutils.add_to_grad_params = torch.Tensor(params:size()):fill(0)
    else
        tregutils.add_to_grad_params:fill(0)
    end

    -- LSTM W
    local LSTMparam1 = tregutils.add_to_grad_params:narrow(1, 1, 4*opt.input_encoding_size*opt.rnn_size)
    local W_i = LSTMparam1:narrow(1, 1, opt.input_encoding_size*opt.rnn_size)
    local W_f = LSTMparam1:narrow(1, opt.input_encoding_size*opt.rnn_size + 1, opt.input_encoding_size*opt.rnn_size)
    local W_o = LSTMparam1:narrow(1, 2*opt.input_encoding_size*opt.rnn_size + 1, opt.input_encoding_size*opt.rnn_size)
    local W_c = LSTMparam1:narrow(1, 3*opt.input_encoding_size*opt.rnn_size + 1, opt.input_encoding_size*opt.rnn_size)
    W_i:copy(grad_LSTMparams['W_i'])
    W_f:copy(grad_LSTMparams['W_f'])
    W_o:copy(grad_LSTMparams['W_o'])
    W_c:copy(grad_LSTMparams['W_c'])

--    assert(torch.all(W_i:eq(grad_LSTMparams['W_i']))and
--            torch.all(W_f:eq(grad_LSTMparams['W_f']))and
--            torch.all(W_o:eq(grad_LSTMparams['W_o']))and
--            torch.all(W_c:eq(grad_LSTMparams['W_c'])))

    assert(torch.all(W_i:cuda():eq(grad_LSTMparams['W_i']))and
            torch.all(W_f:cuda():eq(grad_LSTMparams['W_f']))and
            torch.all(W_o:cuda():eq(grad_LSTMparams['W_o']))and
            torch.all(W_c:cuda():eq(grad_LSTMparams['W_c'])))
    -- LSTM W bias (we don't need it, but need to be sure the grad elements are zeros)
    local LSTMparam2 = tregutils.add_to_grad_params:narrow(1, 4*opt.input_encoding_size*opt.rnn_size + 1, 4*opt.rnn_size)
    assert(torch.sum(LSTMparam2)==0)

    -- LSTM U
    local LSTMparam3 = tregutils.add_to_grad_params:narrow(1, 4*opt.input_encoding_size*opt.rnn_size + 4*opt.rnn_size + 1, 4*opt.rnn_size*opt.rnn_size)
    local U_i = LSTMparam3:narrow(1, 1, opt.rnn_size*opt.rnn_size)
    local U_f = LSTMparam3:narrow(1, opt.rnn_size*opt.rnn_size + 1, opt.rnn_size*opt.rnn_size)
    local U_o = LSTMparam3:narrow(1, 2*opt.rnn_size*opt.rnn_size + 1, opt.rnn_size*opt.rnn_size)
    local U_c = LSTMparam3:narrow(1, 3*opt.rnn_size*opt.rnn_size + 1, opt.rnn_size*opt.rnn_size)
    U_i:copy(grad_LSTMparams['U_i'])
    U_f:copy(grad_LSTMparams['U_f'])
    U_o:copy(grad_LSTMparams['U_o'])
    U_c:copy(grad_LSTMparams['U_c'])

--    assert(torch.all(U_i:eq(grad_LSTMparams['U_i']))and
--            torch.all(U_f:eq(grad_LSTMparams['U_f']))and
--            torch.all(U_o:eq(grad_LSTMparams['U_o']))and
--            torch.all(U_c:eq(grad_LSTMparams['U_c'])))

    assert(torch.all(U_i:cuda():eq(grad_LSTMparams['U_i']))and
            torch.all(U_f:cuda():eq(grad_LSTMparams['U_f']))and
            torch.all(U_o:cuda():eq(grad_LSTMparams['U_o']))and
            torch.all(U_c:cuda():eq(grad_LSTMparams['U_c'])))

    -- LSTM U bias (we don't need it, but need to be sure the grad elements are zeros)
    local LSTMparam4 = tregutils.add_to_grad_params:narrow(1, 4*opt.input_encoding_size*opt.rnn_size + 4*opt.rnn_size + 4*opt.rnn_size*opt.rnn_size + 1, 4*opt.rnn_size)
    assert(torch.sum(LSTMparam4)==0)

    --LSTM output + bias (we don't need it, but need to be sure the grad elements are zeros)
    local vocab_size = 9568
    local LSTMparam5 = tregutils.add_to_grad_params:narrow(1, 4*opt.input_encoding_size*opt.rnn_size + 4*opt.rnn_size + 4*opt.rnn_size*opt.rnn_size + 4*opt.rnn_size + 1, opt.rnn_size*vocab_size)
    assert(torch.sum(LSTMparam5)==0)
    local LSTMparam6 = tregutils.add_to_grad_params:narrow(1, 4*opt.input_encoding_size*opt.rnn_size + 4*opt.rnn_size + 4*opt.rnn_size*opt.rnn_size + 4*opt.rnn_size + 1, opt.rnn_size*vocab_size + 1, vocab_size)
    assert(torch.sum(LSTMparam6)==0)
end

tregutils.consts = {
    sqrt2 = math.sqrt(2.0),
    alpha = 0.25,
    beta = 17.0/16.0,
    lambdaval = 0.05,
}

function tregutils.getvars(LSTMweights)
    local vartbl = {}

    vartbl.norm_W_ix = math.sqrt(torch.sum(torch.pow(LSTMweights['W_i'], 2)))
    vartbl.norm_W_ih = math.sqrt(torch.sum(torch.pow(LSTMweights['U_i'], 2)))

    vartbl.norm_W_fx = math.sqrt(torch.sum(torch.pow(LSTMweights['W_f'], 2)))
    vartbl.norm_W_fh = math.sqrt(torch.sum(torch.pow(LSTMweights['U_f'], 2)))

    vartbl.norm_W_cix = math.sqrt(torch.sum(torch.pow(LSTMweights['W_c'], 2)))
    vartbl.norm_W_cih = math.sqrt(torch.sum(torch.pow(LSTMweights['U_c'], 2)))

    vartbl.norm_W_ox = math.sqrt(torch.sum(torch.pow(LSTMweights['W_o'], 2)))
    vartbl.norm_W_oh = math.sqrt(torch.sum(torch.pow(LSTMweights['U_o'], 2)))

    vartbl.gamma_x = tregutils.consts.alpha*vartbl.norm_W_fx +
                        tregutils.consts.beta*(vartbl.norm_W_ix + vartbl.norm_W_cix)
    vartbl.gamma_h = tregutils.consts.alpha*vartbl.norm_W_fh +
                        tregutils.consts.beta*(vartbl.norm_W_ih + vartbl.norm_W_cih)

    vartbl.invu = math.exp(tregutils.consts.beta*vartbl.norm_W_oh - 1.0)
    vartbl.rho_x = tregutils.consts.sqrt2*vartbl.gamma_x +
                   2.0*vartbl.invu*(tregutils.consts.beta*vartbl.gamma_x*vartbl.norm_W_oh + vartbl.norm_W_ox)
    vartbl.rho_h = tregutils.consts.sqrt2*vartbl.gamma_h +
                   tregutils.consts.sqrt2*vartbl.invu*tregutils.consts.beta*vartbl.gamma_h*vartbl.norm_W_oh
    vartbl.cexp_Woh = math.exp(1.0-tregutils.consts.beta*vartbl.norm_W_oh)
    vartbl.not_rho_h_sq = 1.0 - math.pow(vartbl.rho_h,2)

    return vartbl
end

function tregutils.gradchecker_Wfh(dtreg, LSTMweights, vartbl)

    local rho_x_prime_Wfh = (tregutils.consts.sqrt2*tregutils.consts.alpha/vartbl.norm_W_fh)*
                            (1.0 + (tregutils.consts.beta*vartbl.norm_W_oh)/vartbl.cexp_Woh)

    local R_wrt_Wfh = 2.0*tregutils.consts.lambdaval*math.pow(vartbl.rho_x,2)*vartbl.rho_h*rho_x_prime_Wfh/math.pow(vartbl.not_rho_h_sq,2)

    local dR_dWfh = LSTMweights['U_f']:clone()
    assert(LSTMweights['U_f']:nElement()==dR_dWfh:nElement())

    local s = dR_dWfh:storage()
    for i=1,s:size() do
        s[i] = s[i]*R_wrt_Wfh
    end
    return dR_dWfh
end

function tregutils.gradchecker_Wcih(dtreg, LSTMweights, vartbl)

    local rho_x_prime_Wcih = (tregutils.consts.sqrt2*tregutils.consts.beta/vartbl.norm_W_cih)*
                             (1.0 + (tregutils.consts.beta*vartbl.norm_W_oh)/vartbl.cexp_Woh)

    local R_wrt_Wcih = 2.0*tregutils.consts.lambdaval*math.pow(vartbl.rho_x,2)*vartbl.rho_h*rho_x_prime_Wcih/math.pow(vartbl.not_rho_h_sq,2)

    local dR_dWcih = LSTMweights['U_c']:clone()
    assert(LSTMweights['U_c']:nElement()==dR_dWcih:nElement())

    local s = dR_dWcih:storage()
    for i=1,s:size() do
        s[i] = s[i]*R_wrt_Wcih
    end
    return dR_dWcih
end

function tregutils.gradchecker_Wih(dtreg, LSTMweights, vartbl)

    local rho_x_prime_Wih = (tregutils.consts.sqrt2*tregutils.consts.beta/vartbl.norm_W_ih)*
                            (1.0 + (tregutils.consts.beta*vartbl.norm_W_oh)/vartbl.cexp_Woh)

    local R_wrt_Wih = 2.0*tregutils.consts.lambdaval*math.pow(vartbl.rho_x,2)*vartbl.rho_h*rho_x_prime_Wih/math.pow(vartbl.not_rho_h_sq,2)

    local dR_dWih = LSTMweights['U_i']:clone()
    assert(LSTMweights['U_i']:nElement()==dR_dWih:nElement())

    local s = dR_dWih:storage()
    for i=1,s:size() do
        s[i] = s[i]*R_wrt_Wih
    end
    return dR_dWih
end

function tregutils.gradchecker_Wfx(dtreg, LSTMweights, vartbl)

    local rho_x_prime_Wfx = (tregutils.consts.sqrt2*tregutils.consts.alpha/vartbl.norm_W_fx)*
                            (1.0 + (tregutils.consts.sqrt2*tregutils.consts.beta*vartbl.norm_W_oh)/vartbl.cexp_Woh)

    local R_wrt_Wfx = 2.0*tregutils.consts.lambdaval*vartbl.rho_x*rho_x_prime_Wfx/vartbl.not_rho_h_sq

    local dR_dWfx = LSTMweights['W_f']:clone()
    assert(LSTMweights['W_f']:nElement()==dR_dWfx:nElement())

    local s = dR_dWfx:storage()
    for i=1,s:size() do
        s[i] = s[i]*R_wrt_Wfx
    end
    return dR_dWfx
end

function tregutils.gradchecker_Wcix(dtreg, LSTMweights, vartbl)

    local rho_x_prime_Wcix = (tregutils.consts.sqrt2*tregutils.consts.beta/vartbl.norm_W_cix)*
                             (1.0 + (tregutils.consts.sqrt2*tregutils.consts.beta*vartbl.norm_W_oh)/vartbl.cexp_Woh)

    local R_wrt_Wcix = 2.0*tregutils.consts.lambdaval*vartbl.rho_x*rho_x_prime_Wcix/vartbl.not_rho_h_sq

    local dR_dWcix = LSTMweights['W_c']:clone()
    assert(LSTMweights['W_c']:nElement()==dR_dWcix:nElement())

    local s = dR_dWcix:storage()
    for i=1,s:size() do
        s[i] = s[i]*R_wrt_Wcix
    end
    return dR_dWcix
end

function tregutils.gradchecker_Wix(dtreg, LSTMweights, vartbl)

    local rho_x_prime_Wix = (tregutils.consts.sqrt2*tregutils.consts.beta/vartbl.norm_W_ix)*
                            (1.0 + (tregutils.consts.sqrt2*tregutils.consts.beta*vartbl.norm_W_oh)/vartbl.cexp_Woh)

    local R_wrt_Wix = 2.0*tregutils.consts.lambdaval*vartbl.rho_x*rho_x_prime_Wix/vartbl.not_rho_h_sq

    local dR_dWix = LSTMweights['W_i']:clone()
    assert(LSTMweights['W_i']:nElement()==dR_dWix:nElement())

    local s = dR_dWix:storage()
    for i=1,s:size() do
        s[i] = s[i]*R_wrt_Wix
    end
    return dR_dWix
end

function tregutils.gradchecker_Wox(dtreg, LSTMweights, vartbl)

    local rho_x_prime_Wox = 2.0/(vartbl.norm_W_oh*vartbl.cexp_Woh)

    local R_wrt_Wox = 2.0*tregutils.consts.lambdaval*vartbl.rho_x*rho_x_prime_Wox/vartbl.not_rho_h_sq

    local dR_dWox = LSTMweights['W_o']:clone()
    assert(LSTMweights['W_o']:nElement()==dR_dWox:nElement())
    
    local s = dR_dWox:storage()
    for i=1,s:size() do
        s[i] = s[i]*R_wrt_Wox
    end
    return dR_dWox
end

function tregutils.gradchecker_Woh(dtreg, LSTMweights, vartbl)

    local rho_h_prime_Woh = (tregutils.consts.sqrt2*tregutils.consts.beta*vartbl.gamma_h/vartbl.cexp_Woh)*
                            (tregutils.consts.beta + 1.0/vartbl.norm_W_oh)
    local rho_x_prime_Woh = tregutils.consts.sqrt2*rho_h_prime_Woh +
                            2.0*tregutils.consts.beta*(vartbl.norm_W_ox/(vartbl.norm_W_oh*vartbl.cexp_Woh))
    
    local num = vartbl.rho_x*vartbl.not_rho_h_sq*rho_x_prime_Woh +
                math.pow(vartbl.rho_x,2)*vartbl.rho_h*rho_h_prime_Woh
    local R_wrt_Woh = 2.0*tregutils.consts.lambdaval*num/math.pow(vartbl.not_rho_h_sq,2)
    
    local dR_dWoh = LSTMweights['U_o']:clone()
    assert(LSTMweights['U_o']:nElement()==dR_dWoh:nElement())
    local s = dR_dWoh:storage()
    for i=1,s:size() do -- fill up the Storage
        s[i] = s[i]*R_wrt_Woh
    end
    return dR_dWoh
end

tregutils.treggrad = grad(tregutils.treg)

function tregutils.gettreggrads(params, thin_lm, opt)
    local LSTMparams = tregutils.getparameters(params, opt)
    tregutils.checking(thin_lm, LSTMparams, opt)
    
    --local dtreg_by_dLSTMparams = tregutils.treggrad(LSTMparams)
    local vartbl = tregutils.getvars(LSTMparams)
    dR_dW = {}
    dR_dW['W_i'] = tregutils.gradchecker_Wix(dtreg_by_dLSTMparams, LSTMparams, vartbl)
    dR_dW['U_i'] = tregutils.gradchecker_Wih(dtreg_by_dLSTMparams, LSTMparams, vartbl)
    dR_dW['W_f'] = tregutils.gradchecker_Wfx(dtreg_by_dLSTMparams, LSTMparams, vartbl)
    dR_dW['U_f'] = tregutils.gradchecker_Wfh(dtreg_by_dLSTMparams, LSTMparams, vartbl)
    dR_dW['W_c'] = tregutils.gradchecker_Wcix(dtreg_by_dLSTMparams, LSTMparams, vartbl)
    dR_dW['U_c'] = tregutils.gradchecker_Wcih(dtreg_by_dLSTMparams, LSTMparams, vartbl)
    dR_dW['W_o'] = tregutils.gradchecker_Wox(dtreg_by_dLSTMparams, LSTMparams, vartbl)
    dR_dW['U_o'] = tregutils.gradchecker_Woh(dtreg_by_dLSTMparams, LSTMparams, vartbl)
    tregutils.getgradparams(params, dR_dW, opt)
    assert(tregutils.add_to_grad_params:nElement() == params:nElement())
end

return tregutils
