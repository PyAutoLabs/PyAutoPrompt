The project @z_projects/ic50_workspace is our IC50  use case which we are now aiming to scale up the EP framework
to the IC50 use case.

Can we have this set up so I can do runs on the HPC, noting that all hpc interface information that forms
the link is given in @autolens_assistant/hpc. Thus, this folder should be read carefully and it then
worked out how it links to the HPC on whichw e do all runs.

Note that the HPC folder there describes a PyAutoLens lensing project whereas the use case we will test here
is Ic50 datasets, but the HPC link itself is the same.

We will ultimately want to do runs on CPU and GPU so make sure both are supported.