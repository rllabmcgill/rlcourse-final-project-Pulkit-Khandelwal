# -*- coding: utf-8 -*-
"""
I follow code snippets from Sergey Levine's 2011 paper's MATLAB implementation
and M. Alger's code as a framework within which I implement my versions of the study as described in the report.
"""

import sys
if "../" not in sys.path:
  sys.path.append("../") 
  
from itertools import product

import numpy as np
import numpy.random as rn
import theano as th
import theano.tensor as T
import math

from . import maxent

FLOAT = th.config.floatX

def find_svf(n_states, trajectories):
    

    svf = np.zeros(n_states)

    for trajectory in trajectories:
        for state, _, _ in trajectory:
            svf[state] += 1

    svf /= trajectories.shape[0]

    return th.shared(svf, "svf", borrow=True)

def optimal_value(n_states, n_actions, transition_probabilities, reward,
                  discount, threshold=1e-2):
    
    v = T.zeros(n_states, dtype=FLOAT)

    def update(s, prev_diff, v, reward, tps):
        max_v = float("-inf")
        v_template = T.zeros_like(v)
        for a in range(n_actions):
            tp = tps[s, a, :]
            max_v = T.largest(max_v, T.dot(tp, reward + discount*v))
        new_diff = abs(v[s] - max_v)
        if T.lt(prev_diff, new_diff):
            diff = new_diff
        else:
            diff = prev_diff
        return (diff, T.set_subtensor(v_template[s], max_v)), {}

    def until_converged(diff, v):
        (diff, vs), _ = th.scan(
                fn=update,
                outputs_info=[{"initial": diff, "taps": [-1]},
                              None],
                sequences=[T.arange(n_states)],
                non_sequences=[v, reward, transition_probabilities])
        return ((diff[-1], vs.sum(axis=0)), {},
                th.scan_module.until(diff[-1] < threshold))

    (_, vs), _ = th.scan(fn = until_converged,
                         outputs_info=[
                            # Need to force an inf into the right Theano
                            # data type and this seems to be the only way that
                            # works.
                            {"initial": getattr(np, FLOAT)(float("inf")),
                             "taps": [-1]},
                            {"initial": v,
                             "taps": [-1]}],
                         n_steps=1000)

    return vs[-1]

def find_policy(n_states, n_actions, transition_probabilities, reward, discount,
                threshold=1e-2, v=None):
    
    if v is None:
        v = optimal_value(n_states, n_actions, transition_probabilities, reward,
                          discount, threshold)

    # Get Q using equation 9.2 from Ziebart's thesis.
    Q = T.zeros((n_states, n_actions))
    def make_Q(i, j, tps, Q, reward, v):
        Q_template = T.zeros_like(Q)
        tp = transition_probabilities[i, j, :]
        return T.set_subtensor(Q_template[i, j], tp.dot(reward + discount*v)),{}

    prod = np.array(list(product(range(n_states), range(n_actions))))
    state_range = th.shared(prod[:, 0])
    action_range = th.shared(prod[:, 1])
    Qs, _ = th.scan(fn=make_Q,
                    outputs_info=None,
                    sequences=[state_range, action_range],
                    non_sequences=[transition_probabilities, Q, reward, v])
    Q = Qs.sum(axis=0)
    Q -= Q.max(axis=1).reshape((n_states, 1))  # For numerical stability.
    Q = T.exp(Q)/T.exp(Q).sum(axis=1).reshape((n_states, 1))
    return Q

def find_expected_svf(n_states, r, n_actions, discount,
                      transition_probability, trajectories):
    

    n_trajectories = trajectories.shape[0]
    trajectory_length = trajectories.shape[1]

    policy = find_policy(n_states, n_actions,
                         transition_probability, r, discount)

    start_state_count = T.extra_ops.bincount(trajectories[:, 0, 0],
                                             minlength=n_states)
    p_start_state = start_state_count.astype(FLOAT)/n_trajectories

    def state_visitation_step(i, j, prev_svf, policy, tps):
        """
        The sum of the outputs of a scan over this will be a row of the svf.
        """

        svf = prev_svf[i] * policy[i, j] * tps[i, j, :]
        return svf, {}

    prod = np.array(list(product(range(n_states), range(n_actions))))
    state_range = th.shared(prod[:, 0])
    action_range = th.shared(prod[:, 1])
    def state_visitation_row(prev_svf, policy, tps, state_range, action_range):
        svf_t, _ = th.scan(fn=state_visitation_step,
                           sequences=[state_range, action_range],
                           non_sequences=[prev_svf, policy, tps])
        svf_t = svf_t.sum(axis=0)
        return svf_t, {}

    svf, _ = th.scan(fn=state_visitation_row,
                     outputs_info=[{"initial": p_start_state, "taps": [-1]}],
                     n_steps=trajectories.shape[1]-1,
                     non_sequences=[policy, transition_probability, state_range,
                                 action_range])

    return svf.sum(axis=0) + p_start_state

def irl(structure, feature_matrix, n_actions, discount, transition_probability,
        trajectories, epochs, learning_rate, initialisation="normal", l1=0.1,
        l2=0.1):


    n_states, d_states = feature_matrix.shape
    transition_probability = th.shared(transition_probability, borrow=True)
    trajectories = th.shared(trajectories, borrow=True)

    # Initialise W matrices; b biases.
    n_layers = len(structure)-1
    weights = []
    hist_w_grads = []  # For AdaGrad.
    biases = []
    hist_b_grads = []  # For AdaGrad.
    for i in range(n_layers):
        # W
        shape = (structure[i+1], structure[i])
        if initialisation == "normal":
            matrix = th.shared(rn.normal(size=shape), name="W", borrow=True)
        else:
            matrix = th.shared(rn.uniform(size=shape), name="W", borrow=True)
        weights.append(matrix)
        hist_w_grads.append(th.shared(np.zeros(shape), name="hdW", borrow=True))

        # b
        shape = (structure[i+1], 1)
        if initialisation == "normal":
            matrix = th.shared(rn.normal(size=shape), name="b", borrow=True)
        else:
            matrix = th.shared(rn.uniform(size=shape), name="b", borrow=True)
        biases.append(matrix)
        hist_b_grads.append(th.shared(np.zeros(shape), name="hdb", borrow=True))

    # Initialise α weight, β bias.
    if initialisation == "normal":
        alpha = th.shared(rn.normal(size=(1, structure[-1])), name="alpha",
                      borrow=True)
    else:
        alpha = th.shared(rn.uniform(size=(1, structure[-1])), name="alpha",
                      borrow=True)
    hist_alpha_grad = T.zeros(alpha.shape)  # For AdaGrad.

    adagrad_epsilon = 1e-6  # AdaGrad numerical stability.

    #### Theano symbolic setup. ####

    # Symbolic input.
    s_feature_matrix = T.matrix("x")
    # Feature matrices.
    # All dimensions of the form (d_layer, n_states).
    psis = [s_feature_matrix.T]
    # Forward propagation.
    for W, b in zip(weights, biases):
        psi = T.nnet.sigmoid(th.compile.ops.Rebroadcast((0, False), (1, True))(b)
                           + W.dot(psis[-1]))
        psis.append(psi)
        # φs[1] = φ1 etc.
    # Reward.
    r = alpha.dot(psis[-1]).reshape((n_states,))
    # Engineering hack: z-score the reward.
    r = (r - r.mean())/r.std()
    # Associated feature expectations.
    expected_svf = find_expected_svf(n_states, r,
                                     n_actions, discount,
                                     transition_probability,
                                     trajectories)
    svf = maxent.find_svf(n_states, trajectories.get_value())
    # Derivatives (backward propagation).
    updates = []
    alpha_grad = psis[-1].dot(svf - expected_svf).T
    hist_alpha_grad += alpha_grad**2
    '''
    rho_1 = 0.9
    rho_2 = 0.999
    s= 0
    rr = 0
    s = rho_1 * s + (1-rho_1) * alpha_grad
    rr = rho_2 * rr + (1-rho_2) * alpha_grad**2
    s_cap = (s)/ (1-rho_1)
    rr_cap = (rr)/ (1-rho_2)
    '''
    #adj_alpha_grad =  (-1 * s_cap)/(adagrad_epsilon + math.sqrt(rr_cap)) #Adam

    adj_alpha_grad = alpha_grad/(adagrad_epsilon + T.sqrt(hist_alpha_grad)) #Adagrad
    updates.append((alpha, alpha + adj_alpha_grad*learning_rate))

    def grad_for_state(s, theta, svf_diff, r):
        """
        Calculate the gradient with respect to theta for one state.
        """

        regularisation = abs(theta).sum()*l1 + (theta**2).sum()*l2
        return svf_diff[s] * T.grad(r[s], theta) - regularisation, {}

    for i, W in enumerate(weights):
        w_grads, _ = th.scan(fn=grad_for_state,
                             sequences=[T.arange(n_states)],
                             non_sequences=[W, svf - expected_svf, r])
        w_grad = w_grads.sum(axis=0)
        hist_w_grads[i] += w_grad**2
        '''
        rho_1 = 0.9
        rho_2 = 0.999
        s= 0
        rr = 0
        s = rho_1 * s + (1-rho_1) * alpha_grad
        rr = rho_2 * rr + (1-rho_2) * alpha_grad**2
        s_cap = (s)/ (1-rho_1)
        rr_cap = (rr)/ (1-rho_2)
        adj_w_grad = (-1 * s_cap)/(adagrad_epsilon + math.sqrt(rr_cap)) #Adam
        '''
        adj_w_grad = w_grad/(adagrad_epsilon + T.sqrt(hist_w_grads[i])) #Adagrad
        updates.append((W, W + adj_w_grad*learning_rate))
    for i, b in enumerate(biases):
        b_grads, _ = th.scan(fn=grad_for_state,
                             sequences=[T.arange(n_states)],
                             non_sequences=[b, svf - expected_svf, r])
        b_grad = b_grads.sum(axis=0)
        hist_b_grads[i] += b_grad**2
        '''
        s = 0
        rr = 0       
        rho_1 = 0.9
        rho_2 = 0.999
        s = rho_1 * s + (1-rho_1) * alpha_grad
        rr = rho_2 * rr + (1-rho_2) * alpha_grad**2
        s_cap = (s)/ (1-rho_1)
        rr_cap = (rr)/ (1-rho_2)
        #adj_b_grad = (-1 * s_cap)/(adagrad_epsilon + math.sqrt(rr_cap)) #Adam
        '''
        adj_b_grad = b_grad/(adagrad_epsilon + T.sqrt(hist_b_grads[i])) #Adagrad
        updates.append((b, b + adj_b_grad*learning_rate))

    train = th.function([s_feature_matrix], updates=updates, outputs=r)
    run = th.function([s_feature_matrix], outputs=r)

    for e in range(epochs):
        reward = train(feature_matrix)

    return reward.reshape((n_states,))
