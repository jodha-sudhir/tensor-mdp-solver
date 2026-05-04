function [V, policy, cpu_time] = mdpsolver_jodha(P_s, R_sys, discount, N_horizon)
% MDPSOLVER_JODHA Tensor-Based Value Iteration for Factored MDPs
%
% This function solves a finite-horizon Markov Decision Process (MDP)
% It mitigates the curse of dimensionality by exploiting Kronecker-factored 
% transition dynamics and tensor algebra, reducing computational complexity 
% from exponential to linear in the number of components. 
%
% Reference:
% Jodha, S. P., & Papakonstantinou, K. G. (2026). "Probabilistic Hazard 
% Analysis Framework with Stochastic Optimal Control for Deteriorating Civil 
% Infrastructure Systems." Structural Safety.
%
% Copyright (c) 2026 Sudhir P. Jodha
% Licensed under the MIT License.

    % ---- 1. System Dimensions & Initialization ----
    Nc = numel(P_s); % Number of system components
    Avec = zeros(1, Nc);
    Svec = zeros(1, Nc);
    
    % Extract state and action space sizes per component
    for k = 1:Nc
        Avec(k) = numel(P_s{k});
        Svec(k) = size(P_s{k}{1}, 1);
    end
    S_tot = prod(Svec); % Total system state space size

    % ---- 2. Precomputations for Reward Broadcasting ----
    % Precompute row sums of transition matrices to avoid recalculating 
    % them during the backward induction loop.
    rowSums = cell(1, Nc);
    for k = 1:Nc
        rowSums{k} = cell(1, Avec(k));
        for a = 1:Avec(k)
            rowSums{k}{a} = sum(P_s{k}{a}, 2);  % S_k x 1 sparse vector
        end
    end

    % Mixed-radix stride to map joint-action vectors to a 1D linear index
    stride = ones(1, Nc);
    for k = 2:Nc
        stride(k) = stride(k-1) * Avec(k-1); 
    end
    linA = @(a) uint32(1 + sum((a-1) .* stride));

    % Allocate value and policy arrays
    V      = zeros(S_tot, N_horizon + 1);
    policy = zeros(S_tot, N_horizon, 'uint32');
    
    % Pre-allocate dimension map for efficient tensor broadcasting
    dims_template = ones(1, Nc);

    t0 = tic;

    % ---- 3. Backward Induction (Bellman Optimality) ----
    for n = 0:N_horizon-1
        V_prev  = V(:, N_horizon - n + 1);
        bestVal = -inf(S_tot, 1);
        bestAct = zeros(S_tot, 1, 'uint32');

        % Lexicographic joint-action loop
        a = ones(1, Nc);
        done = false;
        
        while ~done
            % Step A: Expected Future Value via Tensor MatVec
            % Computes (P_1{a1} ⊗ ... ⊗ P_Nc{aNc}) * V_prev exactly, 
            % without ever forming the massive system transition matrix.
            v_next = kron_matvec_nd_opt(P_s, a, V_prev, Svec); 

            % Step B: Immediate Expected Reward via Array Broadcasting
            aidx = linA(a);
            Qa_tensor = reshape(full(R_sys{aidx}), Svec); 

            for k = 1:Nc
                R_k_vec = rowSums{k}{a(k)}; 
                m = Nc - k + 1; % Align dimension indexing
                
                % Reshape component row-sums into singleton dimensions
                dims = dims_template;
                dims(m) = Svec(m);
                R_k_bcast = reshape(full(R_k_vec), dims);
                
                % Element-wise broadcast multiply (highly memory efficient)
                Qa_tensor = Qa_tensor .* R_k_bcast;
            end
            
            % Step C: Compute Q-value and update optimal policy
            Qa = Qa_tensor(:) + discount * v_next;
            
            % Vectorized running maximum 
            take = Qa > bestVal;
            bestVal(take) = Qa(take);
            bestAct(take) = aidx;

            % Increment joint-action vector
            for k = 1:Nc
                a(k) = a(k) + 1;
                if a(k) <= Avec(k)
                    break
                else
                    a(k) = 1;
                    if k == Nc
                        done = true; 
                    end
                end
            end
        end
        
        disp(['Decision Epoch ' num2str(N_horizon-n) ' evaluated.']);
        V(:, N_horizon - n)      = bestVal;
        policy(:, N_horizon - n) = bestAct;
    end

    cpu_time = toc(t0);
end

% =========================================================================
% Helper Function: Tensor-Based Kronecker Matrix-Vector Product
% =========================================================================
function v_out = kron_matvec_nd_opt(P_s, a, v_in, Svec)
% Computes v_out = (P_1 ⊗ P_2 ⊗ ... ⊗ P_N) * v_in by leveraging the mixed
% product property. It reshapes the input vector into an N-D tensor and 
% applies sequential mode-k multiplications.

    X = reshape(v_in, Svec);
    Nc = numel(Svec);
    
    for lev = 1:Nc
        k = Nc - lev + 1;  % Component index
        X = mode_mul_nd_opt(P_s{k}{a(k)}, X, Svec, lev); 
    end
    v_out = X(:);
end

% =========================================================================
% Helper Function: Mode-m Tensor Multiplication
% =========================================================================
function Xout = mode_mul_nd_opt(P, X, Svec, m)
% Multiplies matrix P along mode m of tensor X. Flattening the tensor to 2D
% allows MATLAB to utilize highly optimized sparse BLAS routines.

    N = numel(Svec);

    % Handle single-component boundary case
    if N == 1
        Xout = P * X;
        return;
    end

    % Permute tensor to bring the target mode m to the first dimension
    ord = [m, 1:m-1, m+1:N];      
    Xp  = permute(X, ord);
    
    % Flatten remaining dimensions
    other_dims = Svec(ord(2:end));
    if isempty(other_dims)
        L = 1; 
    else
        L = prod(other_dims);
    end
    X2d = reshape(Xp, Svec(m), L); 

    % Execute fast 2D sparse-dense matrix multiplication
    X2d = P * X2d;                 
    
    % Reconstruct tensor dimensions and invert permutation
    if isempty(other_dims)
        Xp = X2d; 
    else
        Xp = reshape(X2d, [Svec(m), other_dims]);
    end
    Xout = ipermute(Xp, ord);
end