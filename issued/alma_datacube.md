My acollaborators Aris and Hannah want to be able to do interferometer analysis of ALMA data cubes, which are
basically Interferometer objects but lists of them across channels.

Heres hannah's initial issue:

https://github.com/PyAutoLabs/PyAutoPrompt/issues/18

Heres our SLACK conversion:

Jam  [8:43 AM]
If you can open an issue on here describing what you think the code should do using natural language and example python snippets, I can sort that for you: https://github.com/PyAutoLabs/PyAutoPrompt
PyAutoLabs/PyAutoPromptStarting point of the PyAuto workflow: prompt registry and prompt-coupled Claude Code skills.LanguageShellLast updated11 hours agoAdded by GitHubHannah Stacey  [1:34 PM]
Hey @Aris I made an issue in PyAutoPrompt do you have some code as an example for how you are doing the per channel modelling at the moment?
[1:35 PM]https://github.com/PyAutoLabs/PyAutoPrompt/issues/18
Jam  [1:44 PM]
Ok, put anything in the issue you can, at 7pm I get my next wave of Claude tokens, I'll put it to work on the issue, so prepare to have your mind blown. Any information / context you can put in the issue that you think might help please do, especially an existing python snipper or example :slightly_smiling_face:
Hannah Stacey  [1:49 PM]
I had a go at creating an optical 3D modelling script using Cursor but it didn't really work :sweat_smile:
Aris  [2:00 PM]
I have an idea of how we can make it a bit lot more efficient.

For a single inteferometer object you will input the visibilities, uv_wavelengths and noise_map (i.e. visibility errors). For a cube you will have a list of interferometer objects each with their own set of visibilities, uv_wavelengths and noise_map.

The most expensive calculation in the code is, L^T * W_tilde * L, where L is the lensing operator (fixed for a set of lens parameters) and what we call the w_tilde matrix, D^T C^{-1} D, which only depends on uv_wavelengths and noise_map.

However, for channel-to-channel the uv_wavelengths change very little which makes the fourier transform almost identical. The noise_map does change from channel to channel but unless the emission line is at the edge of a spectral window it shouldnt change that much, so we can also assume that it is the same for all channels.

So to make things more efficient we can have a list of interferometer objects but only the visibilities will be different for each object and they will all share the same uv_Wavelengths and noise_map.

So when we compute the likelihood for the cube, all objects have the same L^T * W_tilde * L (which we compute once) and then we solve the linear system for each intereferometer object to compute the source-pixel values. So we end up with a list of reconstructions (this is so much cheaper compared to calculating, L^T * W_tilde * L).

Finally, the total likelihood is the sum of all likelihoods for each interferometer object.

Note that the dirty_image of each interferometer is different but that depends on visibilities (as well as uv_wavelengths, noise_map which we assume are the same) (edited) 
Hannah Stacey  [2:01 PM]
is there a reason why to have a list of interferometer objects instead of having like a boolean parameter to decide whether to use the channel information or collapse it?
[2:02 PM]or even, this would simplify things more, to just have an input uvfits that contains all of this information in a single file?
Aris  [2:03 PM]
a lot of things depend on the structure of the dataset object, e.g. all calculations of the likelihood. If we were to create a new type of dataset, i.e. a 3D cube, then we will have to do a big restructure
Hannah Stacey  [2:03 PM]
would you really? or can it just be interpreted as multi-band?
Aris  [2:04 PM]
I am not that familiar with the details of the multi-dataset modelling in autolens, but I assume it creates different dataset object (in this case imaging datasets).
Hannah Stacey  [2:04 PM]
wouldn't it make your life easier to have a single uvfits so you don't need to do that extra data preparation?
Aris  [2:05 PM]
the amount of wotk for this extra data prep step is nothing compared to having to re-structure the code to expect 3D arrays instead of 2D
Hannah Stacey  [2:05 PM]
ok
[2:06 PM]but will you then still have to manually input 50 lots of interferometer objects
Aris  [2:06 PM]
I had to do it for the PyLensKin package, but it was simpler because its only parametric source mdoels
[2:08 PM]having said that the the suggestion I gave above where all objects share the same w_tilde matrix might not play well with a single L^T * W_tilde * L operation, cause this does not occur at the dataset level
Hannah Stacey  [2:10 PM]
i see what you're saying about assuming sigma and lambda are the same but i'm not sure i like it.. it seems like it would be better to get it correct from the start. I guess it depends what Claude is capable of. We could try the heavy option and see what Claude does? I guess the whole idea is to reduce the amount of manual labour of restructuring the whole code
[2:10 PM]what do you think? is it worth trying?
[2:11 PM]i'm happy to test it on my side to see if it works
Aris  [2:11 PM]
This operation, L^T * W_tilde * L, can take more than 1min for the highest res dataset. Multiple this by the number of channels you are fitting to get an estimate of a single likelihood evaluaiton
[2:12 PM]on CPUs
[2:13 PM]so if W_tilde doesnt change to a level that affects things then we should defo do what I suggested.
[2:13 PM]if you want your runs to end before the end of this year
Hannah Stacey  [2:17 PM]
hmm well obviously i want things to finish, but at MPA we could do 3D modelling of something like SPT-0418 in a day, and I'm sure that autolens could do just as well. Another argument in favour of the ''long way" is that if you go to lower frequencies the fractional bandwidth is larger and you have more per-channel flagging due to RFI, so the assumption might become a problem. If you ever wanted to do VLBI with autolens it would be necessary (edited) 
[2:18 PM]James what is your opinion
Aris  [2:18 PM]
I think the Dataset3D class should be a list of of datasets still but we can add helper fuctions like from_fits_3D where it will recongnise that it should load arrays of shape (n_channels, n_visibilities, 2). That will solve your issue of data prep
Hannah Stacey  [2:18 PM]
we could also try both ways?
Aris  [2:18 PM]
The problem will be later on in the code
[2:19 PM]where we perform the reconstructions
Jam  [2:19 PM]
I will have the first pass at claude do it via Aris's simplification, because its good to get to a point wheere something runs end-to-end, and it sounds like doing it the "proper" way isnt much more than making it so that the list of interferometer objects each have their own uv_wavelengrhs and call their NUFFT indepedently, which will be an easy Claude follow up issue.

The GPU code is 50x faster than the CPU code Aris is used to, I think, so I am not worried about run times. but its good to break the problem down into smaller steps and get ech one running with Claude rather than do everything at once
Hannah Stacey  [2:21 PM]
ok maybe we try that then - we can suggest it a from_fits_3d helper that Aris suggested (maybe also a from_uvfits_3d for visibilities?) then use the simpler approach, see if that works
Aris  [2:22 PM]
ok, so who will make the .md with all that?
Jam  [2:23 PM]
The hardest part is gonna be going from where we are now (lots of context / descriptions but no Python code) to a working example. So I'll instruct claude to get us to the simpler case Aris describes. Once we're there I think the more advanced stuff Claude will be able to do it without much guidance.
Hannah Stacey  [2:23 PM]
either you have to tell me what to do or someone else does it :sweat_smile:
Jam  [2:23 PM]
I will just copy and paste this SLACK chat into Claude and I think we'll get there. If you have any Python code or snippets I think that's the last bit of context we need.
[2:24 PM]doesnt need to be end-to-end Python but a few instrutive snippets on the interface at these key points
Aris  [2:38 PM]
class Interferometer3D:

    def __init__(
        self,
        data: list,
        noise_map: VisibilitiesNoiseMap,
        uv_wavelengths: np.ndarray,
        real_space_mask: Mask2D,
        transformer_class=TransformerNUFFT,
        sparse_operator: Optional[InterferometerSparseOperator] = None,
        raise_error_dft_visibilities_limit: bool = True,
    ):
        pass

        list_of_interferometers = []
        for i in range(uv_wavelengths.shape[0]):
            list_of_interferometers.append(
                Interferometer(
                    data: list,
                    noise_map: list,
                    uv_wavelengths: list,
                    real_space_mask: Mask2D,
                    transformer_class=TransformerNUFFT,
                )
            )

        self.transformer = transformer_class(
            uv_wavelengths=uv_wavelengths,
            real_space_mask=real_space_mask,
        )

    def from_fits_3D(
        data_path,
        noise_map_path,
        uv_wavelengths_path,
        real_space_mask,
        visibilities_hdu=0,
        noise_map_hdu=0,
        uv_wavelengths_hdu=0,
        transformer_class=TransformerNUFFT,
    ):


        visibilities = ndarray_via_fits_from(
            file_path=data_path,
            hdu=visibilities_hdu
        )
        if is_3D(visibilities):
            list_of_visibilities = [
                Visibilities(visibilities[i, :, :])
                for i in range(visibilities.shape[0])
            ]
        else:
            raise NotImplementedError()

        noise_map = ndarray_via_fits_from(
            file_path=noise_map_path,
            hdu=noise_map_hdu
        )
        if is_3D(noise_map):
            noise_map_mean = np.mean(noise_map, axis=0) # NOTE: NOT SURE THIS IS THE BEST WAY


        uv_wavelengths = ndarray_via_fits_from(
            file_path=uv_wavelengths_path,
            hdu=uv_wavelengths_hdu
        )
        if is_3D(uv_wavelengths):
            uv_wavelengths_mean = np.mean(uv_wavelengths, axis=0)

        Interferometer3D(
            visibilities=list_of_visibilities,
            noise_map=VisibilitiesNoiseMap(noise_map)
            uv_wavelengths=uv_wavelengths_mean,
        )

    def is_3D(array):
        if len(array.shape) == 3:
            return True
        else:
            return FalseInterferomter3D could be something like this
Jam  [2:39 PM]
Cool, I think I've got enough. Final question, is there any reason we need Interferometer3D rather than just a list of Interferometer objects? It doesnt feel to me like we need it to be its own Python class
[2:40 PM]I guess its because you want the Inversion to reuse the NUFFT once across all channels, and the list dataset API wouldnt do that naturally. Ok
Hannah Stacey  [2:41 PM]
Maybe this is more a general computing question, but is there any memory advantage to a list of nxm arrays as opposed to an nxmxl array 
Jam  [2:42 PM]
Ok, I'll prob actually get the slower implementation working which doesnt reuse NUFFT and we can build from there. Thats a question for Claude at this point
Aris  [2:43 PM]
that would be the simpler thing to implement by far. You will jsut reuse existing functionality for everything
Aris  [2:44 PM]
i guess start testing from there, have a feeling for run times, and then we move to the sligtly more complex implementation
Hannah Stacey  [2:45 PM]
Maybe you could eventually have a boolean switch to choose which version to use

Here are a few threads:

Hannah Stacey  [2:41 PM]
Maybe this is more a general computing question, but is there any memory advantage to a list of nxm arrays as opposed to an nxmxl array 
Jam  [2:47 PM]
No, the primary motivation of using lists is that many of the Python objects which do the computation (e.g. AnalysisInterferometer, FitInterferometer can naturally be iterated over a lists, and so we can set this up reusing a lot of code.

I think most Astronomers would of started by defining a datacube as a 3d ndarray (x, y, channel) but this would mean most the downstream code needs specific functionality to h andle the third dimension.

In terms of memory, JAX will convert all this to special array types before modeling begins so it doesnt matter what Python objects we use

Aris, Hannah Stacey Aris and youAris  [2:44 PM]
i guess start testing from there, have a feeling for run times, and then we move to the sligtly more complex implementation
Jam  [2:45 PM]
Yeah exactly, it wont be hard to implmenet your speed up but better to have Claude do it from a point where things are stable and working

Aris, Hannah Stacey Aris, Hannah Stacey, and youAris  [2:18 PM]
I think the Dataset3D class should be a list of of datasets still but we can add helper fuctions like from_fits_3D where it will recongnise that it should load arrays of shape (n_channels, n_visibilities, 2). That will solve your issue of data prep
Hannah Stacey  [2:19 PM]
yeah that could work
Jam  [2:20 PM]
Yeah I think we'll end up with a Dataset3D object like this, which reuses all the FitInterferomter / AnalysisInterferometer objects internally so avoid code restructuring / redevelopment
Aris  [2:21 PM]
perhaps we can have like a preload fitting class where if the operation, L^T * W_tilde * L, is performed once then it is reused for all channels.

This will require the least development and potential to breakt he code

Also work a read through of @autolens_workspace/scritps/interferometer/features/pixelization/likelihood_function.py for a step-by-step guide about what we're doing.


My take is the following:

1) We will end up with lists of Interferometer objects, one for each cube, prioritizing the computationally-expensive but implmenetation-simple approach of doing a NUFFT for each channel initially.
2) We will do all initial work on autolens_workspace_developer/datacube, and work through to integration workspaces from there.
3) We first want a good, representative autolens_workspace_developer/datacube, followed by the kind of step-by-step JAX likelihood fucntion we have in scripts like autolens_workspace_developer/jax_profiling/interferometer.
4) We will then make some autolens_workspace scripts, primnarily a simulator.py and modeling.py.

Obviously this is a pretty huge feature and thus do deep research before you give me a plan. Think critically about design, but work towards getting the simplest (but could be slower) implementation going first. Make absolutely certain resuing all code ande existing API makes sense over making bespoke source code (e.g. Datacube3D object,AnalysisDataCube3D class, etc.)