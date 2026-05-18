We currently does not have implemented a point source of light, light profile, which would be a delta function
implemneted in a single pixel in the image. This should be easy to addd, and would be added in the light/profiles
module of autogalaxy.

First it would be added as a standard light profile, and then variants for linear would be added.

For point sources of light, 2d convolution is tricky, as it really requires 2D subsampling of the PSF and convoluiton,
which are features that will be added relatively soon. Thikn about if there are simple approaches we can use to
add this now, but its fine to defer until we have full support for over sampled PSF convolution. In that case,
add some sort of a warning when this light profile is used for modeling.