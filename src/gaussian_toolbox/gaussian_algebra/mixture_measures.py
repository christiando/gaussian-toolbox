class GaussianMixtureMeasure:
    def __init__(
        self, components: Iterable["GaussianMeasure"], weights: jnp.ndarray = None
    ):
        """ Class of mixture of Gaussian measures

            u(x) = sum_i w_i * u_i(x)

            where w_i are weights and u_i the component measures.

        :param components: list
            List of Gaussian measures.
        :param weights: jnp.ndarray [num_components] or None
            Weights of the components. If None they are assumed to be 1. (Default=None)
        """
        self.num_components = len(components)
        if weights is None:
            self.weights = jnp.ones(self.num_components)
        else:
            self.weights = weights
        self.components = components
        self.R, self.D = self.components[0].R, self.components[0].D

    def slice(self, indices: list) -> "GaussianMixtureMeasure":
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

        return GaussianMixtureMeasure(components_new, self.weights)

    def evaluate_ln(self, x: jnp.ndarray) -> jnp.ndarray:
        """ Evaluates the log-exponential term at x.

        :param x: jnp.ndarray [N, D]
            Points where the factor should be evaluated.

        :return: jnp.ndarray [N, R]
            Log exponential term.
        """
        ln_comps = jnp.empty((self.num_components, self.R, x.shape[0]))

        for icomp in range(self.num_components):
            ln_comps[icomp] = self.components[icomp].evaluate_ln(x)
        ln_u, signs = logsumexp(
            ln_comps, b=self.weights[:, None, None], axis=0, return_sign=True
        )
        return ln_u, signs

    def evaluate(self, x: jnp.ndarray) -> jnp.ndarray:
        """ Evaluates the exponential term at x.

        :param x: jnp.ndarray [N, D]
            Points where the factor should be evaluated.

        :return: jnp.ndarray [N, R]
            Exponential term.
        """
        ln_u, signs = self.evaluate_ln(x)
        return signs * jnp.exp(ln_u)

    def multiply(
        self, factor: factors.ConjugateFactor, update_full: bool = False
    ) -> "GaussianMeasure":
        """ Computes the product between the measure u and a conjugate factor f

            f(x) * u(x)

            and returns the resulting Gaussian measure.

        :param factor: ConjugateFactor
            The conjugate factor the measure is multiplied with.
        :param update_full: bool
            Whether also the covariance and the log determinants of the new Gaussian measure should be computed. 
            (Default=True)

        :return: GaussianMixtureMeasure
            Returns the resulting GaussianMixtureMeasure.
        """
        components_new = []
        for icomp in range(self.num_components):
            comp_new = factor.multiply_with_measure(
                self.components[icomp], update_full=update_full
            )
            components_new.append(comp_new)
        return GaussianMixtureMeasure(components_new, weights=self.weights)

    def hadamard(
        self, factor: factors.ConjugateFactor, update_full: bool = False
    ) -> "GaussianMeasure":
        """ Computes the hadamard (componentwise) product between the measure u and a conjugate factor f

            f(x) * u(x)

            and returns the resulting Gaussian measure.

        :param factor: ConjugateFactor
            The conjugate factor the measure is multiplied with.
        :param update_full: bool
            Whether also the covariance and the log determinants of the new Gaussian measure should be computed. 
            (Default=True)

        :return: GaussianMixtureMeasure
            Returns the resulting GaussianMixtureMeasure.
        """
        components_new = []
        for icomp in range(self.num_components):
            comp_new = factor.hadamard_with_measure(
                self.components[icomp], update_full=update_full
            )
            components_new.append(comp_new)
        return GaussianMixtureMeasure(components_new, weights=self.weights)

    def integrate(self, expr: str = "1", **kwargs) -> jnp.ndarray:
        """ Integrates the indicated expression with respect to the Gaussian mixture measure.

        :param expr: str
            Indicates the expression that should be integrated. Check measure's integration dict. Default='1'.
        :kwargs:
            All parameters, that are required to evaluate the expression.
        """
        integration_res = self.weights[0] * self.components[0].integration_dict[expr](
            **kwargs
        )
        for icomp in range(1, self.num_components):
            integration_res += self.weights[icomp] * self.components[
                icomp
            ].integration_dict[expr](**kwargs)
        return integration_res
