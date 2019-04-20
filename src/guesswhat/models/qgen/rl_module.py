import tensorflow as tf


class PolicyGradient(object):

    def __init__(self, stop_gradient, weight_td=1, weight_entropy=0):
        self.stop_gradient = stop_gradient

        self.weight_td = weight_td
        self.weight_entropy = weight_entropy

    def __call__(self, cum_rewards, value_function, policy_state, actions, action_mask):

        with tf.variable_scope('td_loss'):
            td_loss = tf.square(cum_rewards - value_function)
            td_loss *= action_mask
            td_loss = tf.reduce_sum(td_loss)

        with tf.variable_scope('policy_gradient_loss'):
            log_of_policy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=policy_state, labels=actions)

            policy_loss = tf.multiply(log_of_policy, cum_rewards - value_function)  # score function
            policy_loss *= action_mask

            policy_loss = tf.reduce_sum(policy_loss, axis=1)  # sum over the dialogue trajectory
            policy_loss = tf.reduce_mean(policy_loss, axis=0)  # reduce over batch dimension

        with tf.variable_scope('entropy_loss'):
            entropy_loss = 0
            if self.weight_entropy > 0:
                policy = tf.nn.softmax(policy_state)

                entropy_loss = tf.reduce_sum(policy * tf.log(policy), axis=-1)
                entropy_loss *= action_mask

                entropy_loss = tf.reduce_sum(entropy_loss, axis=1)
                entropy_loss = tf.reduce_mean(entropy_loss, axis=0)

        loss = policy_loss + self.weight_td*td_loss - self.weight_entropy*entropy_loss

        return loss


