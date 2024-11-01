import tensorflow as tf

@tf.function
def negative_loglikelihood(y, class_probs):
    """Calculates negative log-likelihood.
        
    Args:
        class_probs : A 'Tensor' of class probabilities, 
            whose shape is [batch size, num_class].
        y : A 'Tensor' of output, whose shape is 
            [batch size, num_output_units].

    Returns:
        nll: A 'Tensor' representing negative log-likelihood.

    Note:
        To prevent divergence, we add epsilon=10e-12 into the log function as
        log(x) => log(x+epsilon).

    """
    # Converts y (+1 or -1) into binary representation (+1 or 0)
    y_binary = 0.5 * (y+1.0)

    # Extracts probability for target class
    target_prob = tf.reduce_sum(class_probs * y_binary, axis=1)
        
    # Returns negative log-likelihood
    epsilon = 10e-12
    nll = tf.reduce_mean(-tf.math.log(target_prob+epsilon), axis=0)
    return nll

