class GaussianMixtureDensity(measures.GaussianMixtureMeasure):
    def __init__(
        self, components: Iterable["GaussianDensity"], weights: jnp.ndarray = None
    ):
        """ Class of mixture of Gaussian measures
        
            u(x) = sum_i w_i * u_i(x)
            
            where w_i are weights and u_i the component measures.
            
        :param components: list
            List of Gaussian densities.
        :param weights: jnp.ndarray [num_components] or None
            Weights of the components, that must be positive. If None they are assumed to be 1/num_components. 
            (Default=None)
        """
        super().__init__(components, weights)
        self.normalize()

    def normalize(self):
        """ Normalizes the mixture (assuming, that its components are already normalized).
        """
        self.weights /= jnp.sum(self.weights)

    def sample(self, num_samples: int) -> jnp.ndarray:
        """ Generates samples from the Gaussian mixture density.
        
        :param num_samples: int
            Number of samples that are generated.
        
        :return: jnp.ndarray [num_samples, R, D]
            The samples.
        """
        cum_weights = jnp.cumsum(self.weights)
        rand_nums = jnp.random.rand(num_samples)
        comp_samples = jnp.searchsorted(cum_weights, rand_nums)
        samples = jnp.empty((num_samples, self.R, self.D))
        for icomp in range(self.num_components):
            comp_idx = jnp.where(comp_samples == icomp)[0]
            samples[comp_idx] = self.components[icomp].sample(len(comp_idx))
        return samples

    def slice(self, indices: jnp.ndarray) -> "GaussianMixtureDensity":
        """ Returns an object with only the specified entries.
        
        :param indices: list
            The entries that should be contained in the returned object.
            
        :return: GaussianMixtureMeasure
            The resulting Gaussian mixture measure.
        """
        components_new = []
        for icomp in range(self.num_components):
            comp_sliced = self.components[icomp].slice(indices)
            components_new.append(comp_sliced)

        return GaussianMixtureDensity(components_new, self.weights)

