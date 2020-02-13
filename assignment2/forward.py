import numpy as np
from numpy.linalg import multi_dot


def normalize(ar): return ar/sum(ar)


def forward(E, t, prior):
    """
    Filtering function: recursively estimate probability, P(X_t+1)
    given evidence, e_1:t+1. That is: P(X_t+1 | e_1:t+1)

    Based on equation (15.12) from Russell & Norvig 2010 Chapter 15 p. 579

    Parameters
    ----------
         E : binary numpy array
             observable variables
         t : int
             previous time-slice t
     prior : numpy array
             probability vector of prior beliefs, P(X_0).
    Returns
    -------
         a : numpy array
             normalized probabilities of rain and not rain on time-slice (day) t+1. [P(rain),P(not rain)].
       """

    T = np.array([[0.7, 0.3], [0.3, 0.7]])  # Transition model matrix
    O = np.array([[[0.1, 0], [0, 0.8]], [[0.9, 0], [0, 0.2]]])  # Sensor model matrix
    if t == 0:
        return normalize(multi_dot([O[E[t]], T.T, prior.T]))
    f = multi_dot([O[E[t]], T.T, forward(E, t-1, prior)])   # equation (15.12)
    return normalize(f)


def backward(E, k, t):
    """
    Smoothing function: recursively compute the distribution of over past
    states given evidence to present, P(X_k | e_1:t) for
    0 <= k < t.

    Based on equation (15.13) from Russell & Norvig 2010 Chapter 15 p. 579

    Parameters
    ----------
        E : binary numpy array
            observable variables
        k : int
            how far back to time-slice, k, the recursion rolls from time-slice t
        t : int
            time-slice t
    Returns
    -------
        a : numpy array
            ratio of true false rain on day k. [P(rain),P(not rain)].
    """

    T = np.array([[0.7, 0.3], [0.3, 0.7]])  # Transition model matrix
    O = np.array([[[0.1, 0], [0, 0.8]], [[0.9, 0], [0, 0.2]]])  # Sensor model matrix
    """Inital vector with 1s because P(e_t+1:t | X_t) = 1"""
    I = np.array([1, 1])
    if k == t:
        return multi_dot([T, O[E[k]], I.T])
    return  multi_dot([T, O[E[k]], backward(E, k+1, t)])    # equation (15.13)


def forward_backward(ev, init, pri):
    """
    The forward–backward algorithm for finding the smoothed estimates: computes the posterior probabilities
    of a sequence og state variables given a sequence of observed variable, evidence variables.
    That is: P(X_t | e:1:T) where T > t.

    It makes a forward prediction, filtering, using the forward() function and simultaneously
    uses the backward() function to compute the backward probabilities and then combines these
    to get the final smoothed value estimates.

    Based on Forward-Backward algorithm, figure 15.4, from Russell & Norvig 2010 Chapter 15 p. 576

    Parameters
    ----------
          ev : binary numpy array
               observable variables
        init : numpy array
               initial probability vector with 1s because P(e_t+1:t | X_t) = 1
         pri : numpy array
               probability vector of prior beliefs, P(X_0).

    Returns
    -------
        a : numpy array
            normalized probabilities of rain and not rain on time-slice(day) t. [P(rain),P(not rain)].
    """
    j = 0
    """Initialize arrays"""
    bv_array = np.array([backward(ev, j, len(ev) - 1)])
    fv_array = pri
    print("Part C: all backwarded messages")
    for i in range(len(ev)):
        """Handle vector element order"""
        if i == 0:
            fv = forward(ev, i, pri)
            fv = np.array([np.append(fv[0], fv[1])])
            fv_array = np.concatenate((fv_array, fv), axis=0)
            continue

        fv = forward(ev, i, pri)
        fv = np.array([np.append(fv[0], fv[1])])
        fv_array = np.concatenate((fv_array, fv), axis=0)
        bv = np.array([backward(ev, i, len(ev)-1)])
        bv_array = np.concatenate((bv_array, bv), axis=0)

    bv_array = np.concatenate((bv_array, init), axis=0)
    res = fv_array*bv_array
    """normalize res array"""
    for i in range(len(res)):
        res[i] = normalize(res[i])
    return res


if __name__ == "__main__":
    """Constants:"""
    prior = np.array([[0.5, 0.5]])
    ini = np.array([[1, 1]])
    evidence = np.array([1, 1, 0, 1, 1])

    """Functions:"""
    #a = forward(evidence, 0, prior)
    #b = backward(evidence, 0, len(evidence)-1)
    c = forward_backward(evidence, ini, prior)

    print(c)




