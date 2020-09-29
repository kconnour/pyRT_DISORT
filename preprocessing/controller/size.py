class Size:
    def __init__(self, n_layers, n_moments, n_streams, n_umu, n_phi, n_user_levels):
        self.n_layers = n_layers
        self.n_moments = n_moments
        self.n_streams = n_streams     # number of computational polar angles; NSTR = MAXCMU
        self.n_umu = n_umu
        self.n_phi = n_phi
        self.n_user_levels = n_user_levels

        self.check_inputs()

    def check_inputs(self):
        """ Check that the inputs are good enough

        Returns
        -------
        None
        """
        if self.n_moments < self.n_streams:
            raise SystemExit('The number of phase function moments must be >= number of streams')

        if self.n_streams < 2:
            raise SystemExit('The number of streams must be >= 2')

        if self.n_streams % 2 != 0:
            raise SystemExit('The number of streams must be even')
