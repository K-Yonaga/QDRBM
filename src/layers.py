import math
import tensorflow as tf

class QRBMLayer(tf.keras.layers.Layer):
    """Class for Quantum Restricted Boltzmann Machine Layer (QRBMLayer).

    QRBMLayer is a Layer object for quantum restricted Boltzmann
    machine. This class inherits from tf.keras.layers.Layer.

    Parameters:
        _num_units (int) : The number of units

        _dtype (tf.float32) : The dtype of the layer's computations
            and variables.

        _weight_initializer : Initializer instance of tensorflow.

        _bias_initializer : : Initializer instance of tensorflow.

        _quantum_bias_initializer : : Initializer instance of tensorflow.

        _weight_trainable (bool) : If True, the weight variables will be part
            of the layer's "trainable_variables".

        _bias_trainable (bool) : If True, the bias variables will be part
            of the layer's "trainable_variables".

        _quantum_bias_trainable (bool) : If True, the quantum-bias variables
            will be part of the layer's "trainable_variables".

        _use_quantum_bias (bool) : If False, the quantum-bias variables
            will not be part of the layer's variables.

    """

    def __init__(self,
                 num_units=None,
                 dtype=tf.float32,
                 weight_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 quantum_bias_initializer='glorot_uniform',
                 weight_trainable=True,
                 bias_trainable=True,
                 quantum_bias_trainable=True,
                 use_quantum_bias=True):
        # Builds layer.
        self._num_units = num_units
        self._dtype = dtype
        self._use_quantum_bias = use_quantum_bias
        super().__init__()
        
        # Parameters for classical Boltzmann machine.
        self._weight_initializer = weight_initializer
        self._bias_initializer = bias_initializer
        self._weight_trainable = weight_trainable
        self._bias_trainable = bias_trainable

        # Parameters for quantum Boltzmann machine.
        self._quantum_bias_initializer = quantum_bias_initializer
        self._quantum_bias_trainable = quantum_bias_trainable

        # Initializes weight and bias.
        self._weight = None
        self._bias = None
        self._quantum_bias = None

    #########################################
    # Properties and utilities
    #########################################
    
    @property
    def num_unit(self):
        return self._num_units

    @property
    def dtype(self):
        return self._dtype
    
    @property
    def weight(self):
        return self._weight
    
    @property
    def bias(self):
        return self._bias

    @property
    def quantum_bias(self):
        return self._quantum_bias 
    
    #########################################
    # Tensroflow methods
    #########################################

    def build(self, input_shape):
        """Creates layer's variables.
        
        The method inherits from tf.keras.layers.Layer.build. This method
        creates variables for weight, bias, and quantum bias.

        """
        # Add classical weight
        self._weight = super().add_weight(
            name='weight',
            shape=[int(input_shape[-1]), self._num_units], 
            dtype=self._dtype, 
            initializer=self._weight_initializer, 
            trainable=self._weight_trainable)
        
        # Add classical bias
        self._bias = super().add_weight(
            name='bias',
            shape=[self._num_units], 
            dtype=self._dtype, 
            initializer=self._bias_initializer, 
            trainable=self._bias_trainable)

        # Add quantum bias
        if self._use_quantum_bias:
            self._quantum_bias = super().add_weight(
                name='quantum_bias',
                shape=[self._num_units], 
                dtype=self._dtype, 
                initializer=self._quantum_bias_initializer, 
                trainable=self._quantum_bias_trainable)
        
    def call(self, input):
        return tf.matmul(input, self._weight)
        
class RBMLayer(QRBMLayer):
    """Class for Restricted Boltzmann Machine Layer (QRBMLayer).

    RBMLayer is a Layer object for the restricted Boltzmann
    machine. This class inherits from QRBM. 

    Note:
        RBMLayer call QRBMLayer with:
            quantum_bias_initializer = None,
            qautntum_bias_trainable = False.
            use_quantum_bias = False

    One can access "quantum_bias", but this class always returns "tf.zeros".

    """

    def __init__(self,
                 num_units=None,
                 dtype=tf.float32,
                 weight_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 weight_trainable=True,
                 bias_trainable=True,):
        super().__init__(
            num_units=num_units,
            dtype=dtype,
            weight_initializer=weight_initializer,
            bias_initializer=bias_initializer,
            quantum_bias_initializer='zeros',
            weight_trainable=weight_trainable,
            bias_trainable=bias_trainable,
            quantum_bias_trainable=False,
            use_quantum_bias=False)
    
    #########################################
    # Properties and utilities
    #########################################
    
    @property
    def quantum_bias(self):
        return tf.zeros(self._num_units, dtype=self._dtype)
        
class InputLayer(tf.keras.layers.InputLayer):
    """Class for input layer (InputLayer).

    InputLayer is a Layer object for input layer in discriminative 
    restricted Boltzmann machine (DRBM). 
    This class inherits from tf.keras.layers.InputLayer.

    """

    def __init__(
        self, 
        num_input_units=None, 
        dtype=None,
        **kwargs):
        super().__init__(input_shape=(num_input_units,), dtype=dtype, **kwargs)
