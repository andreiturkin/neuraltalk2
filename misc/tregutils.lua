--    LSTM weights:
--    W is a weight for the input,
--    U is a weight for the recurrent connection.
--    lambdaval is the regularization parameter
--
--
local tregutils = {}
tregutils.add_to_grad_params = nil

tregutils.consts = {
    sqrt2 = math.sqrt(2.0),
    alpha = 0.25,
    beta = 17.0/16.0,
    lambda1 = 0.05,
    lambda2 = 0.01,
    lambdaval = 0.05,
}

function tregutils.getvars(W_i, U_i, W_f, U_f, W_c, U_c, W_o, U_o)
    local vartbl = {}

    vartbl.norm_W_ix = math.sqrt(torch.sum(torch.pow(W_i, 2)))
    vartbl.norm_W_ih = math.sqrt(torch.sum(torch.pow(U_i, 2)))

    vartbl.norm_W_fx = math.sqrt(torch.sum(torch.pow(W_f, 2)))
    vartbl.norm_W_fh = math.sqrt(torch.sum(torch.pow(U_f, 2)))

    vartbl.norm_W_cix = math.sqrt(torch.sum(torch.pow(W_c, 2)))
    vartbl.norm_W_cih = math.sqrt(torch.sum(torch.pow(U_c, 2)))

    vartbl.norm_W_ox = math.sqrt(torch.sum(torch.pow(W_o, 2)))
    vartbl.norm_W_oh = math.sqrt(torch.sum(torch.pow(U_o, 2)))

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

function tregutils.getparameters(params, thin_lm, opt)
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
    -- Checking whether or not we get what we wanted to
    tregutils.getparameters_checking(thin_lm, LSTMparam, opt)

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
    assert(torch.all(LSTMparam_layer['W_i']:eq(LSTMparam['W_i']))and
            torch.all(LSTMparam_layer['W_f']:eq(LSTMparam['W_f']))and
            torch.all(LSTMparam_layer['W_o']:eq(LSTMparam['W_o']))and
            torch.all(LSTMparam_layer['W_c']:eq(LSTMparam['W_c'])))
    ---- U
    assert(torch.all(LSTMparam_layer['U_i']:eq(LSTMparam['U_i']))and
            torch.all(LSTMparam_layer['U_f']:eq(LSTMparam['U_f']))and
            torch.all(LSTMparam_layer['U_o']:eq(LSTMparam['U_o']))and
            torch.all(LSTMparam_layer['U_c']:eq(LSTMparam['U_c'])))
end

--
function tregutils.getgradparams(opt)
    -- LSTM W
    local LSTMparam1 = tregutils.add_to_grad_params:narrow(1, 1, 4*opt.input_encoding_size*opt.rnn_size)

    local W_i = LSTMparam1:narrow(1, 1, opt.input_encoding_size*opt.rnn_size)
    local W_f = LSTMparam1:narrow(1, opt.input_encoding_size*opt.rnn_size + 1, opt.input_encoding_size*opt.rnn_size)
    local W_o = LSTMparam1:narrow(1, 2*opt.input_encoding_size*opt.rnn_size + 1, opt.input_encoding_size*opt.rnn_size)
    local W_c = LSTMparam1:narrow(1, 3*opt.input_encoding_size*opt.rnn_size + 1, opt.input_encoding_size*opt.rnn_size)

    -- LSTM W bias (we don't need it, but need to be sure the grad elements are zeros)
    local LSTMparam2 = tregutils.add_to_grad_params:narrow(1, 4*opt.input_encoding_size*opt.rnn_size + 1, 4*opt.rnn_size)
    LSTMparam2:zero()

    -- LSTM U
    local LSTMparam3 = tregutils.add_to_grad_params:narrow(1, 4*opt.input_encoding_size*opt.rnn_size + 4*opt.rnn_size + 1, 4*opt.rnn_size*opt.rnn_size)
    local U_i = LSTMparam3:narrow(1, 1, opt.rnn_size*opt.rnn_size)
    local U_f = LSTMparam3:narrow(1, opt.rnn_size*opt.rnn_size + 1, opt.rnn_size*opt.rnn_size)
    local U_o = LSTMparam3:narrow(1, 2*opt.rnn_size*opt.rnn_size + 1, opt.rnn_size*opt.rnn_size)
    local U_c = LSTMparam3:narrow(1, 3*opt.rnn_size*opt.rnn_size + 1, opt.rnn_size*opt.rnn_size)

    local vartbl = tregutils.getvars(W_i, U_i, W_f, U_f, W_c, U_c, W_o, U_o)

    -- Derivative calculation
    -- They will replace the parameters that are in tregutils.add_to_grad_params
    tregutils.gradchecker_Wix(W_i, vartbl)
    tregutils.gradchecker_Wfx(W_f, vartbl)
    tregutils.gradchecker_Wox(W_o, vartbl)
    tregutils.gradchecker_Wcix(W_c, vartbl)

    tregutils.gradchecker_Wih(U_i, vartbl)
    tregutils.gradchecker_Wfh(U_f, vartbl)
    tregutils.gradchecker_Woh(U_o, vartbl)
    tregutils.gradchecker_Wcih(U_c, vartbl)

    -- LSTM U bias (we don't need it, but need to be sure the grad elements are zeros)
    local LSTMparam4 = tregutils.add_to_grad_params:narrow(1, 4*opt.input_encoding_size*opt.rnn_size + 4*opt.rnn_size + 4*opt.rnn_size*opt.rnn_size + 1, 4*opt.rnn_size)
    LSTMparam4:zero()

    --LSTM output + bias (we don't need it, but need to be sure the grad elements are zeros)
    local vocab_size = 9568
    local LSTMparam5 = tregutils.add_to_grad_params:narrow(1, 4*opt.input_encoding_size*opt.rnn_size + 4*opt.rnn_size + 4*opt.rnn_size*opt.rnn_size + 4*opt.rnn_size + 1, opt.rnn_size*vocab_size)
    LSTMparam5:zero()

    local LSTMparam6 = tregutils.add_to_grad_params:narrow(1, 4*opt.input_encoding_size*opt.rnn_size + 4*opt.rnn_size + 4*opt.rnn_size*opt.rnn_size + 4*opt.rnn_size + 1, opt.rnn_size*vocab_size + 1, vocab_size)
    LSTMparam6:zero()
end

function tregutils.gradchecker_Wfh(dR_dWfh, vartbl)

    local rho_h_prime_Wfh = (tregutils.consts.sqrt2*tregutils.consts.alpha/vartbl.norm_W_fh)*
                            (1.0 + (tregutils.consts.beta*vartbl.norm_W_oh)/vartbl.cexp_Woh)

    local R_wrt_Wfh = 2.0*tregutils.consts.lambdaval*math.pow(vartbl.rho_x,2)*vartbl.rho_h*rho_h_prime_Wfh/math.pow(vartbl.not_rho_h_sq,2)
    if (math.pow(vartbl.rho_h,2) >= 1.0) then
        dR_dWfh:apply(function(x) return x*R_wrt_Wfh + tregutils.consts.lambda1*2.0*vartbl.rho_h*rho_h_prime_Wfh*x end)
    else
        dR_dWfh:apply(function(x) return x*R_wrt_Wfh end)
    end
end

function tregutils.gradchecker_Wcih(dR_dWcih, vartbl)

    local rho_h_prime_Wcih = (tregutils.consts.sqrt2*tregutils.consts.beta/vartbl.norm_W_cih)*
                             (1.0 + (tregutils.consts.beta*vartbl.norm_W_oh)/vartbl.cexp_Woh)

    local R_wrt_Wcih = 2.0*tregutils.consts.lambdaval*math.pow(vartbl.rho_x,2)*vartbl.rho_h*rho_h_prime_Wcih/math.pow(vartbl.not_rho_h_sq,2)
    if (math.pow(vartbl.rho_h,2) >= 1.0) then
        dR_dWcih:apply(function(x) return x*R_wrt_Wcih + tregutils.consts.lambda1*2.0*vartbl.rho_h*rho_h_prime_Wcih*x end)
    else
        dR_dWcih:apply(function(x) return x*R_wrt_Wcih end)
    end
end

function tregutils.gradchecker_Wih(dR_dWih, vartbl)

    local rho_h_prime_Wih = (tregutils.consts.sqrt2*tregutils.consts.beta/vartbl.norm_W_ih)*
                            (1.0 + (tregutils.consts.beta*vartbl.norm_W_oh)/vartbl.cexp_Woh)
    local R_wrt_Wih = 2.0*tregutils.consts.lambdaval*math.pow(vartbl.rho_x,2)*vartbl.rho_h*rho_h_prime_Wih/math.pow(vartbl.not_rho_h_sq,2)
    if (math.pow(vartbl.rho_h,2) >= 1.0) then
        dR_dWih:apply(function(x) return x*R_wrt_Wih + tregutils.consts.lambda1*2.0*vartbl.rho_h*rho_h_prime_Wih*x end)
    else
        dR_dWih:apply(function(x) return x*R_wrt_Wih end)
    end
end

function tregutils.gradchecker_Wfx(dR_dWfx, vartbl)

    local rho_x_prime_Wfx = (tregutils.consts.sqrt2*tregutils.consts.alpha/vartbl.norm_W_fx)*
                            (1.0 + (tregutils.consts.sqrt2*tregutils.consts.beta*vartbl.norm_W_oh)/vartbl.cexp_Woh)
    local R_wrt_Wfx = 2.0*tregutils.consts.lambdaval*vartbl.rho_x*rho_x_prime_Wfx/vartbl.not_rho_h_sq

    dR_dWfx:apply(function(x) return x*R_wrt_Wfx end)
end

function tregutils.gradchecker_Wcix(dR_dWcix, vartbl)

    local rho_x_prime_Wcix = (tregutils.consts.sqrt2*tregutils.consts.beta/vartbl.norm_W_cix)*
                             (1.0 + (tregutils.consts.sqrt2*tregutils.consts.beta*vartbl.norm_W_oh)/vartbl.cexp_Woh)
    local R_wrt_Wcix = 2.0*tregutils.consts.lambdaval*vartbl.rho_x*rho_x_prime_Wcix/vartbl.not_rho_h_sq

    dR_dWcix:apply(function(x) return x*R_wrt_Wcix end)
end

function tregutils.gradchecker_Wix(dR_dWix, vartbl)

    local rho_x_prime_Wix = (tregutils.consts.sqrt2*tregutils.consts.beta/vartbl.norm_W_ix)*
                            (1.0 + (tregutils.consts.sqrt2*tregutils.consts.beta*vartbl.norm_W_oh)/vartbl.cexp_Woh)
    local R_wrt_Wix = 2.0*tregutils.consts.lambdaval*vartbl.rho_x*rho_x_prime_Wix/vartbl.not_rho_h_sq

    dR_dWix:apply(function(x) return x*R_wrt_Wix end)
end

function tregutils.gradchecker_Wox(dR_dWox, vartbl)

    local rho_x_prime_Wox = 2.0/(vartbl.norm_W_oh*vartbl.cexp_Woh)
    local R_wrt_Wox = 2.0*tregutils.consts.lambdaval*vartbl.rho_x*rho_x_prime_Wox/vartbl.not_rho_h_sq

    dR_dWox:apply(function(x) return x*R_wrt_Wox end)
end

function tregutils.gradchecker_Woh(dR_dWoh, vartbl)

    local rho_h_prime_Woh = (tregutils.consts.sqrt2*tregutils.consts.beta*vartbl.gamma_h/vartbl.cexp_Woh)*
                            (tregutils.consts.beta + 1.0/vartbl.norm_W_oh)
    local rho_x_prime_Woh = tregutils.consts.sqrt2*rho_h_prime_Woh +
                            2.0*tregutils.consts.beta*(vartbl.norm_W_ox/(vartbl.norm_W_oh*vartbl.cexp_Woh))
    
    local num = vartbl.rho_x*vartbl.not_rho_h_sq*rho_x_prime_Woh +
                math.pow(vartbl.rho_x,2)*vartbl.rho_h*rho_h_prime_Woh
    local R_wrt_Woh = 2.0*tregutils.consts.lambdaval*num/math.pow(vartbl.not_rho_h_sq,2)

    if (math.pow(vartbl.rho_h,2) >= 1.0) and (tregutils.consts.beta*vartbl.norm_W_oh >= 1.0) then
        dR_dWoh:apply(function(x) return x*R_wrt_Woh + tregutils.consts.lambda1*2.0*vartbl.rho_h*rho_x_prime_Woh*x +
                                         tregutils.consts.lambda2*(tregutils.consts.beta/vartbl.norm_W_oh)*x end)
    end
    if (math.pow(vartbl.rho_h,2) >= 1.0) then
        dR_dWoh:apply(function(x) return x*R_wrt_Woh + tregutils.consts.lambda1*2.0*vartbl.rho_h*rho_h_prime_Woh*x end)
    end
    if (tregutils.consts.beta*vartbl.norm_W_oh >= 1.0) then
        dR_dWoh:apply(function(x) return x*R_wrt_Woh + tregutils.consts.lambda2*(tregutils.consts.beta/vartbl.norm_W_oh)*x end)
    end
    if (tregutils.consts.beta*vartbl.norm_W_oh < 1.0) and (math.pow(vartbl.rho_h,2) < 1.0) then
        dR_dWoh:apply(function(x) return x*R_wrt_Woh end)
    end
end

function tregutils.gettreggrads(params, thin_lm, opt)
    tregutils.add_to_grad_params = params:clone()
    tregutils.getgradparams(opt)
    assert(tregutils.add_to_grad_params:nElement() == params:nElement())
end

return tregutils
