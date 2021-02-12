# BilevelExperiments

Set of experiments reported in the paper "Optimality Conditions and a Trust Region Algorithm for Bilevel Parameter Learning in Total Variation Image Denoising".

## Installation
This module depends on Tuomo's Valkonen AlgTools and ImageTools packages as well as VariationalImaging and TestDatasets developed by David Villacís.

```sh
$ hg clone https://tuomov.iki.fi/repos/AlgTools/
$ hg clone https://tuomov.iki.fi/repos/ImageTools/
```

Once cloned the repositories, we need to upload those to the julia package manager

```julia
pkg> develop AlgTools
pkg> develop ImageTools
pkg> add https://github.com/dvillacis/VariationalImaging.git
pkg> add TestDatasets
```

To reproduce the experiments it is necessary to clone the code repository and initialize the julia modules

```sh
$ git clone https://github.com/dvillacis/BPLDenoising.git
$ julia --project=BilevelExperiments
```

Once in the julia REPL just import the module and the experiment functions

```julia
julia> using BilevelExperiments.ScalarParameterLearning
julia> run_bilevel_learn("dataset_name")
```

## Datasets

