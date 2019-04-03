# Multivariate Gaussian distributions for Tensorflow

This repository contains parts of the implementation code for the projects 'Structured Uncertainty Prediction Networks' (CVPR 2018) and 'Training VAEs Under Structured Residuals' (arxiv 2018). 

<br/>

## Papers
[Structured Uncertainty Prediction Networks](https://arxiv.org/pdf/1802.07079.pdf) <br/>
[Garoe Dorta](http://people.bath.ac.uk/gdp24/)<sup> 1,2</sup>, Sara Vicente<sup> 2</sup>, [Lourdes Agapito](http://www0.cs.ucl.ac.uk/staff/L.Agapito/)<sup> 3</sup>, [Neill D.F. Campbell](http://cs.bath.ac.uk/~nc537/)<sup> 1</sup> and Ivor Simpson<sup> 2</sup><br/>
<sup>1 </sup>University of Bath, <sup>2 </sup>Anthropics Technology Ltd., <sup>3 </sup>University College London  <br/>
IEEE Conference on Computer Vision and Pattern Recognition ([CVPR](http://cvpr2018.thecvf.com/)), 2018

[Training VAEs Under Structured Residuals](https://arxiv.org/pdf/1804.01050.pdf) <br/>
[Garoe Dorta](http://people.bath.ac.uk/gdp24/)<sup> 1,2</sup>, Sara Vicente<sup> 2</sup>, [Lourdes Agapito](http://www0.cs.ucl.ac.uk/staff/L.Agapito/)<sup> 3</sup>, [Neill D.F. Campbell](http://cs.bath.ac.uk/~nc537/)<sup> 1</sup> and Ivor Simpson<sup> 2</sup><br/>
<sup>1 </sup>University of Bath, <sup>2 </sup>Anthropics Technology Ltd., <sup>3 </sup>University College London  <br/>
arXiv e-prints, 2018

<br/>

## Dependencies
* [Python 3.5](https://www.python.org/)
* [Tensorflow 1.13.1](https://www.tensorflow.org/)
* [TensorFlow Probability 0.6](https://github.com/tensorflow/probability)

<br/>

## Detailed description

This code provides a collection of Multivariate Gaussian distributions parametrized with log diagonals. This parametrization leads to more stable computations of log probabilities. The distributions are subclasses of tensorflow_distributions and can directly replace any Multivariate Gaussian distribution class.

In practice this means changing the activation of the layer that predicts the covariance matrix from softplus to no activation.
For example in a dense layer setting
```python
import tensorflow_probability as tfp
tf_dist = tensorflow_probability.distributions
import mvg_distributions as mvg_dist

n = # ... Data dimensionality
loc = # ... The predicted means
h = # ... Tensor of a hidden layer in the network

# Tensorflow probability approach
diag_covariance = keras.layers.Dense(n, activation=tf.nn.softplus)(h)
softplus_mvg = tfp.distributions.MultivariateNormalDiag(loc=loc, scale_diag=tf.sqrt(diag_covariance))

# mvg_distributions approach
log_diag_covariance = keras.layers.Dense(n, activation=None)(h)
log_mvg = mvg_dist.MultivariateNormalDiag(loc=loc, log_diag_covariance=log_diag_covariance)
```

The provided distributions are
* **MultivariateNormalDiag**: for diagonal covariance matrices
* **MultivariateNormalChol**: for Cholesky covariance matrices
* **MultivariateNormalPrecCholFilters**: for sparse Cholesky precision matrices
* **MultivariateNormalPrecCholFiltersDilation**: for sparse Cholesky precision matrices with dilated sparsity pattern
* **IsotropicMultivariateNormal**: N(0,I) distribution, useful for numerically stable KL divergence with MultivariateNormalDiag
* **CholeskyWishart**: a Cholesky-Whistart distribution, i.e. a distribution over Cholesky matrices
* **Gamma**: a Gamma distribution that evaluates probabilities on log_values and samples log_values, useful for setting a prior on the *log_diag_precision* argument of a *MultivariateNormalDiag* distribution 

### Examples

`examples/autoencoder_mvg_chol_filters.py` shows how the use *MultivariateNormalPrecCholFilters* in an autoencoder setting, which is a demonstration of the work in *Structured Uncertainty Prediction Networks*

`examples/autoencoder_mvg_diag.py` shows how the use *MultivariateNormalDiag* in the same setting as the previous example.

 `kl_diag_isotropic.py` shows how to use *IsotropicMultivariateNormal* and *MultivariateNormalDiag* to compute\
 `KL(N(mu, sigma I) || N(0, I))`, which is common in VAE networks.
 
 `kl_chol_diag.py` and `log_prob_chol_filters.py` contain additional simple examples of KL divergences and log prob evaluations.

<br/>

**If this work is useful for your research, please cite our papers.**
