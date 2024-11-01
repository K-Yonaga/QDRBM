import tensorflow as tf
from tensorflow.python.keras.engine import data_adapter

class DRBM(tf.keras.Sequential):
    """Class for Discriminative Restricted Boltzmann Machine

    DRBM provides training and prediction methods. 
    The class inherits from tf.keras.Sequential.

    Parameters:
        _num_classes (int) : The number of target classes.
        
        _weight_hidden (float) : Weight for rewaiting the learning rate of
            the hidden layer (e.g. lr_hidden = _weight_hidden * lr).
        
        _epsilon (float) : Parameter to prevent divergence of the log function
            (default to 10e-12).

        _onehot (tf.float32) : Tensor of one-hot bits (+1, or 0), whose
            shape is [_num_class, _num_class].

        _onehot_spin (tf.float32) : Tensor of one-hot spins (+1, or -1), whose
            shape is [_num_class, _num_class].
            
    Raises:
        ValueError:
            If the number of total layers is more than two (QDRBM supports 
            only single hidden layer).

        ValueError:
            If the hidden layer is not layers.RBMLayer or 
            layers.QRBMLayer.

        ValueError:
            If the output layer is not layers.RBMLayer.
    """

    def __init__(
            self, 
            layers=None, 
            name=None,
            num_classes=None,
            weight_hidden=10.0):
        super().__init__(layers=layers, name=name)
        self._num_classes = num_classes
        self._weight_hidden = weight_hidden
    
        # Create onehot spins.
        self._onehot = tf.one_hot(
            [i for i in range(self._num_classes)], self._num_classes)
        self._onehot_spin = self._to_spin(self._onehot)

    #########################################
    # Properties and utilities
    #########################################

    @property
    def onehot(self):
        return self._onehot
    
    @property
    def onehot_spin(self):
        return self._onehot_spin

    def _to_spin(self, bin):
        return 2*bin - 1
 
    def _reweight_grads(self, grads):
        for i in range(len(grads)-2):
            grads[i] = self._weight_hidden * grads[i]
        return grads
    
    #########################################
    # Methods for loss function
    #########################################
    
    @tf.function
    def _effective_energy(self, x):
        """Calculates effective energy.
        
        Args:
            x: A 'Tensor' of input, whose shape is 
                [batch size, num_input_units].

        Returns:
            E_eff: A 'Tensor' representing effective energy, 
                whose shape is [batch size, num_classes].
        """
        # Energy for output layer.
        E_output = -tf.tensordot(
            self.layers[-1].bias, self.onehot_spin, axes=[0, 1])
        
        # Efffective magnetic field:
        #     hz_input_hidden : [batch_size, num_hidden_units],
        #     hz_output_hidden : [num_classes, num_hidden_units],
        #     hz : [batch_size, num_classes, num_hidden_units].
        hz_input_hidden = tf.matmul(x, self.layers[0].weight)
        hz_output_hidden = tf.matmul(
            self.onehot_spin, tf.transpose(self.layers[-1].weight))
        hz = (
            self.layers[0].bias 
            + tf.expand_dims(hz_input_hidden, axis=1) 
            + tf.expand_dims(hz_output_hidden, axis=0))

        # Energy for hidden layer:
        #     E_hidden : [batch_size, num_classes, num_hidden_units].
        hx = self.layers[0].quantum_bias
        E_hidden = tf.sqrt(tf.square(hz)+tf.square(hx))

        # log(2cosh(E)) => abs(E) + log(1+exp(-2*abs(E)))
        log_2cosh = tf.abs(E_hidden) + tf.math.log(1+tf.exp(-2*tf.abs(E_hidden)))

        # Effective energy
        #     E_eff : [batch_size, num_classes]
        E_eff = tf.reduce_sum(log_2cosh, axis=-1)
        return E_output + E_eff

    @tf.function
    def _free_energy_xclamped(self, E):
        """Calculates free energy with x clmaped.
        
        Args:
            E: A 'Tensor' of energy, whose shape is [batch size, num_classes].

        Returns:
            F_xclamped: A 'Tensor' representing free energy, 
                whose shape is [batch size].
        """
        # Maximum energy, which is for avoinding overflow
        E_max = tf.math.reduce_max(E, axis=1)

        # Free energy
        sum_exp = tf.reduce_sum(
            tf.exp(E-tf.expand_dims(E_max, axis=1)), axis=1)
        F_xclamped = -E_max - tf.math.log(sum_exp)
        return F_xclamped

    @tf.function
    def _free_energy_xyclamped(self, E):
        """Calculates free energy with x and y clmaped.
        
        Args:
            E: A 'Tensor' of energy, whose shape is [batch size, num_classes].

        Returns:
            F_xyclamped: A 'Tensor' representing free energy, 
                whose shape is [batch size, num_class].
        """
        # Free energy
        # F = -log(exp(E_class)) = -E_class
        F_xyclamped = -E
        return F_xyclamped
    
    @tf.function
    def _class_probability(self, x):
        """Calculates class probability.
        
        Args:
            x: A 'Tensor' of input, whose shape is 
                [batch size, num_input_units].

        Returns:
            class_probability: A 'Tensor' representing class probability, 
                whose shape is [batch size, num_class].
        """
        # Effective energy for all classes
        E_eff = self._effective_energy(x)

        # Free energy
        F_xclamped = self._free_energy_xclamped(E_eff)
        F_xyclamped = self._free_energy_xyclamped(E_eff)

        dF = F_xyclamped - tf.expand_dims(F_xclamped, axis=1)
        class_probability = tf.exp(-dF)
        return class_probability
    
    #########################################
    # Tensroflow methods
    #########################################
    
    def compile(
        self,
        optimizer="rmsprop",
        loss=None,
        metrics=None,
        loss_weights=None,
        weighted_metrics=None,
        run_eagerly=None,
        steps_per_execution=None,
        jit_compile=None,
        **kwargs):
        """Configures the model for training.
        
        The method inherits from tf.keras.Sequential.compile. After calling 
        the super-class method, The method confirms the network configuration.

        Raises:
            ValueError:
                If the number of total layers is more than two (QDRBM supports
                only single hidden layer).

            ValueError:
                If the hidden layer is not layers.RBMLayer or 
                layers.QRBMLayer.

            ValueError:
                If the output layer is not layers.RBMLayer.
        """
        # Call method from super class.
        super().compile(
            optimizer,
            loss,
            metrics,
            loss_weights,
            weighted_metrics,
            run_eagerly,
            steps_per_execution,
            jit_compile,
            **kwargs)

        # Check laerys
        if len(self.layers) != 2:
            raise ValueError('DRBM supports single hidden layer.')
        if (self.layers[0].__class__.__name__ != 'RBMLayer' and
            self.layers[0].__class__.__name__ != 'QRBMLayer'):
            raise TypeError(
                'DRBM supports RBMLayer or QRBMLayer in the hidden layer.')
        if self.layers[-1].__class__.__name__ != 'RBMLayer':
            raise TypeError('DRBM supports RBMLayer in the output layer.')

    def train_step(self, data):
        """Method for one training step.
        Args:
            data: A nested structure of Tensors.

        Returns: 
            results : A 'dict' containing loss and accuracy.
        """
        # Unpacs data.
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        # Calculate loss and gradients.
        with tf.GradientTape() as tape:
            class_probs = self(x)
            #loss = self._negative_loglikelihood(class_probs, y)
            loss = self.compute_loss(x, y, class_probs, sample_weight)
        grads = tape.gradient(loss, self.trainable_variables)

        # Reweight gradients in hidden layer
        grads = self._reweight_grads(grads)
        
        # Update trainable variables
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, class_probs)

        # Return results
        return {m.name: m.result() for m in self.metrics}
    
    def call(self, inputs, training=None, mask=None):
        class_probs = self._class_probability(inputs)
        return class_probs