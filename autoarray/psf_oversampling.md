A point spreadh function is used to blur images via 2d convolution.

This blurring occurs predominantly in the package @PyAutoArray/autoarray/operators/convolver.py.

The source code currently requires PSF blurring to occur at the same resolution (pixel scale) as the
image, meaning the PSF is always the same resolution as the image.

However, for modeling, convolution can be performed at a higher resolution than the image, which allows for more accurate 
blurring and modeling of the image. This requires us to have an oversampled PSF, which is a PSF that has a higher 
resolution than the image. 

For modeling, where images are generated PSF blurring happens in @PyAutoGalaxy/autogalaxy/operate/image.py.

Modeling can always evaluate images using a hgiher resolition grid, blurring them with the PSF at high
resolution and then downsample to the observed image resolution. Oversampling is implemented in
@PyAutoArray/autoarray/operators/over_sampling.

Note that over sampling often uses an adaptive sub-szie, which means that 2D covnolution with a PSF is not
well defined. for now, we will assume adaptive over sampling is not used.

I want us to be able to append the Convolver class with a convolve_over_sample_size integer, which specifies the over sample size of the PSF. 
This will allow us to perform convolution at a higher resolution than the image, which will improve the accuracy of the blurring and modeling of the image.
For example, if convolve_over_sample_size is 2, then the PSF will be oversampled by a factor of 2, meaning it will have a resolution that is 2 times higher than the image.

This, in turn, means out imaging object @PyAutoArray/autoarray/dataset/imaging/dataset.py will need to be 
extended to include the convolve_over_sample_size_lp and convolve_over_sample_size_pixelization attributes, which will 
specify the over sample size of the PSF for the lensing and pixelization operations, respectively.

class Imaging(AbstractDataset):
    def __init__(
        self,
        data: Array2D,  
        noise_map: Optional[Array2D] = None,
        psf: Optional[Convolver] = None,
        psf_setup_state: bool = False,
        noise_covariance_matrix: Optional[np.ndarray] = None,
        over_sample_size_lp: Union[int, Array2D] = 4,
        over_sample_size_pixelization: Union[int, Array2D] = 4,
        use_normalized_psf: Optional[bool] = True,
        check_noise_map: bool = True,
        sparse_operator: Optional[ImagingSparseOperator] = None,
    ):


Also,read through the @PyAutoArray/autoarray/inversion/inversion/imaging package, and parents, to see
how PSF convolution enters this. I think we can get it to work in PyAutoArray/autoarray/inversion/inversion/imaging/mapping.py,
and will leave work in PyAutoArray/autoarray/inversion/inversion/imaging/sparse.py to future work. 

This is a complex task, therefore I think we should extend @autolens_workspace_test/scripts/imaging/convolution.py
with a numerical test. 

We should then build on this test in a separate test file using a simple over sampled PSF, to get a numerical
result we can test the source code against.

@autolens_workspace/scripts/imaging/simulator.py is a good example we can build on to show how to use
over sampled PSFs in a real simulation. We can extend this script to show how to use over sampled PSFs in a real simulation.

Come up with a plan to implement over sampled PSFs.