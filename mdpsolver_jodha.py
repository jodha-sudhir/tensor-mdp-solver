import numpy as np
import scipy.sparse as sp
import time
import itertools

def mdpsolver_jodha(P_s, R_sys, discount, N_horizon):
    """
    MDPSOLVER_JODHA Tensor-Based Value Iteration for Factored MDPs
    
    This function solves a finite-horizon Markov Decision Process (MDP).
    It mitigates the curse of dimensionality by exploiting Kronecker-factored 
    transition dynamics and tensor algebra, reducing computational complexity 
    from exponential to linear in the number of components.
    
    Reference:
    Jodha, S. P., & Papakonstantinou, K. G. (2026). "Probabilistic Hazard 
    Analysis Framework with Stochastic Optimal Control for Deteriorating Civil 
    Infrastructure Systems." Engineering Structures.
    
    Copyright (c) 2026 Sudhir P. Jodha
    Licensed under the MIT License.
    
    Parameters:
    -----------
    P_s : list of lists of scipy.sparse matrices
        Component transition matrices. P_s[k][a] is the sparse transition 
        matrix for component k under action a.
    R_sys : list of numpy arrays or scipy.sparse vectors
        System reward arrays, mapped to the joint-action index.
    discount : float
        Discount factor (gamma)
    N_horizon : int
        Finite planning horizon (T)
        
    Returns:
    --------
    V : numpy.ndarray
        Optimal value function matrix of shape (S_tot, N_horizon + 1)
    policy : numpy.ndarray
        Optimal policy matrix of shape (S_tot, N_horizon)
    cpu_time : float
        Execution time in seconds
    """
    
    # ---- 1. System Dimensions & Initialization ----
    Nc = len(P_s) # Number of system components
    Avec = [len(comp) for comp in P_s]
    Svec = [comp[0].shape[0] for comp in P_s]
    S_tot = int(np.prod(Svec)) # Total system state space size
    
    # ---- 2. Precomputations for Reward Broadcasting ----
    # Precompute row sums of transition matrices to avoid recalculating 
    # them during the backward induction loop.
    rowSums = []
    for k in range(Nc):
        comp_sums = []
        for a in range(Avec[k]):
            # Sum across rows (axis=1) and flatten to 1D array
            r_sum = np.asarray(P_s[k][a].sum(axis=1)).flatten()
            comp_sums.append(r_sum)
        rowSums.append(comp_sums)
        
    # Allocate value and policy arrays
    V = np.zeros((S_tot, N_horizon + 1), dtype=np.float64)
    policy = np.zeros((S_tot, N_horizon), dtype=np.uint32)
    
    t0 = time.time()
    
    # Generate lexicographic joint-action space natively in C-order
    # (i.e., the last component's action changes fastest)
    action_ranges = [range(A) for A in Avec]
    
    # ---- 3. Backward Induction (Bellman Optimality) ----
    for n in range(N_horizon):
        V_prev = V[:, N_horizon - n]
        bestVal = np.full(S_tot, -np.inf, dtype=np.float64)
        bestAct = np.zeros(S_tot, dtype=np.uint32)
        
        # Iterate over all possible joint-action combinations
        for aidx, a in enumerate(itertools.product(*action_ranges)):
            
            # Step A: Expected Future Value via Tensor MatVec
            # Computes (P_0{a0} ⊗ ... ⊗ P_Nc-1{aNc-1}) * V_prev exactly, 
            # without ever forming the massive system transition matrix.
            v_next = kron_matvec_nd_opt(P_s, a, V_prev, Svec)
            
            # Step B: Immediate Expected Reward via Array Broadcasting
            # Extract reward (handle both dense arrays and sparse vectors)
            R_curr = R_sys[aidx]
            if sp.issparse(R_curr):
                R_curr = R_curr.todense()
            Qa_tensor = np.asarray(R_curr, dtype=np.float64).reshape(Svec).copy()
            
            for k in range(Nc):
                R_k_vec = rowSums[k][a[k]]
                
                # Reshape component row-sums into singleton dimensions
                dims = [1] * Nc
                dims[k] = Svec[k]
                R_k_bcast = R_k_vec.reshape(dims)
                
                # Element-wise broadcast multiply (highly memory efficient)
                Qa_tensor *= R_k_bcast
                
            # Step C: Compute Q-value and update optimal policy
            Qa = Qa_tensor.flatten() + discount * v_next
            
            # Vectorized running maximum
            take = Qa > bestVal
            bestVal[take] = Qa[take]
            bestAct[take] = aidx
            
        print(f"Decision Epoch {N_horizon - n} evaluated.")
        V[:, N_horizon - n - 1] = bestVal
        policy[:, N_horizon - n - 1] = bestAct
        
    cpu_time = time.time() - t0
    return V, policy, cpu_time

# =========================================================================
# Helper Function: Tensor-Based Kronecker Matrix-Vector Product
# =========================================================================
def kron_matvec_nd_opt(P_s, a, v_in, Svec):
    """
    Computes v_out = (P_0 ⊗ P_1 ⊗ ... ⊗ P_N) * v_in by leveraging the mixed
    product property. Reshapes input vector into an N-D tensor and applies
    sequential mode-k multiplications.
    """
    # Reshape input into N-D tensor (assumes Python native C-order memory)
    X = v_in.reshape(Svec)
    Nc = len(Svec)
    
    for k in range(Nc):
        # Multiply P_k along mode/axis k
        X = mode_mul_nd_opt(P_s[k][a[k]], X, Svec, k)
        
    return X.flatten()

# =========================================================================
# Helper Function: Mode-m Tensor Multiplication
# =========================================================================
def mode_mul_nd_opt(P, X, Svec, m):
    """
    Multiplies sparse matrix P along mode m of tensor X. Flattening the 
    tensor to 2D allows SciPy to utilize highly optimized BLAS routines.
    """
    N = len(Svec)
    
    # Handle single-component boundary case
    if N == 1:
        return P.dot(X)
    
    # Permute tensor to bring the target mode m to the first dimension (axis 0)
    Xp = np.moveaxis(X, m, 0)
    shape_Xp = Xp.shape
    
    # Flatten remaining dimensions
    X2d = Xp.reshape(Svec[m], -1)
    
    # Execute fast 2D sparse-dense matrix multiplication
    X2d_out = P.dot(X2d)
    
    # Reconstruct tensor dimensions and invert permutation
    Xp_out = X2d_out.reshape(shape_Xp)
    Xout = np.moveaxis(Xp_out, 0, m)
    
    return Xout
