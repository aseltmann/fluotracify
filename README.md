# fluotracify - doctoral research project done in a reproducible way

This work is a collection of experiments about Fluorescence Correlation
Spectroscopy (FCS) time-series artifacts. In multiple experiments, we simulated
such artifacts, trained U-Net machine learning models to predict them, and found
methods to optimize FCS time-series or Time-Correlated Single Photon Counting
(TCSPC) data. This repository contains all code, most of the data, and
interactive `org-mode` files describing the work I've done during my time in the
[Eggeling lab](https://www.biophysical-imaging.com) from January 2020 til Summer
2023 (after working on the same project since ~ April 2019). 

## Connected paper and licenses
The following paper is currently under review:
Seltmann, A.; Carravilla, P.; Reglinski, K.; Eggeling, E.; Waithe, D. Neural
Network Informed Photon Filtering Reduces Artifacts in Fluorescence Correlation
Spectroscopy Data. 2023 (currently under review)

The nanosimpy module found under `src/nanosimpy` is a fork of Dominic Waithe's
original nanosimpy library - here the original licensing applies.

All software components (code) of this repository, e.g. the rest of the `src`
directory, the `data/mlruns` directory, or python functions defined in the
`LabBook` files, are licensed under the Apache-2.0 License - see the
[LICENSE](LICENSE) file for details.

All non-software components of this repository, e.g. plots, figures,
correlated FCS files, reports on correlation fits, and the non-software
components of the LabBooks, are licensed under a 
[Creative Commons Attribution 4.0 International License][cc-by] - see the
[LICENSE-CC-BY](LICENSE-CC-BY) file for details.

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png

## Quickly viewing experiments

Quickly view a rendered version of the full LabBook
[here](https://aseltmann.github.io/fluotracify/data/LabBook-all.html). 

## Getting started

### Viewing the electronic LabBook on Github

The `.org` files can be viewed with markdown-like rendering here on Github.
Github is not rendering the full information of these files yet, this is why
displaying the raw file is recommended.

Note that the file looks different depending on the branch you are on. The
`main` branch does only include a barebones file without experiments, the `data`
branch includes a record of all experiments (saved under
`data/LabBook-all.org`). 

### Viewving the electronic LabBook on your computer

You can open `.org` files in any text editor. The file is
written in [org-mode](https://orgmode.org/), so leverage org-mode's
possibilities of easy viewing, navigating and editing the file, you need a text
editor which understands org-mode ([Emacs](https://www.gnu.org/software/emacs/)
[and](https://medium.com/urbint-engineering/emacs-doom-for-newbies-1f8038604e3b)
[Co](http://kitchingroup.cheme.cmu.edu/scimax),
[Non-Emacs
alternatives](https://opensource.com/article/19/1/productivity-tool-org-mode)).

Check out [this talk](https://www.youtube.com/watch?v=SzA2YODtgK4) by
[Harry R. Schwartz](https://github.com/hrs) to get an overview of what org-mode
can do. It inspired me to use it.

### Using the source code

This electronic notebook contains a lot of code, and even though the
**src/fluotracify** folder is prepared to be a python package, possible useful
modules will be released in a separate project if needed. Currently, the
internal use of this package is done with a dirty hack:

``` python
import sys
sys.path.append('/path/to/fluotracify/src/')
```

Then you can import the fluotracify modules e.g. like this:

``` python
from fluotracify.training import build_model as bm
```

## Organization of this repository - A reproducible workflow

This repository follows a worklfow which is aimed to assure reproducibility
based on [this paper by Stanisic et
al.](https://hal.inria.fr/hal-01112795/document). There is a [free online
course](https://www.fun-mooc.fr/courses/course-v1:inria+41016+self-paced/5d99aa3742e34d6f87eed84b71fdde74/)
on reproducible research organized in part by the authors of the paper, which 
might be the way to go, if you want to learn more about this approach. The
following gives a short overview on the intended meaning of the most important
branches of this `git` repository:

### `main` branch:

- contains all the source code in folder `src/` which is used for experiments.
- contains the `LabBook.org` template
- contains setup- and metadata files such as `MLproject` or `conda.yaml`
- the log contains only lasting alterations on the folders and files mentioned
  above, which are e.g. used for conducting experiments or which introduce new
  features. 

### `exp-#` branches:

- if one starts an experiment, the code and templates will be branched out from
  `main` in an `exp-#` branch, # substituting some meaningful descriptor.
- all data generated during the experiment (e.g. .csv files, plots, images,
  etc), is stored in a folder with the name `data/exp-#`, except machine
  learning-specific data and metadata from `mlflow` runs, which are saved
  under `data/mlruns` (this allows easily comparing machine learning runs
  with different experimental settings)
- The `LabBook.org` file is essential
  - If possible, all code is executed from inside this file (meaning analysis
    scripts or calling the code from the `scr/` directory).
  - All other steps taken during an experiment are noted down, as well as
    conclusions or my thought process while conducting the experiment
  - Provenance data, such as metadata about the environment the code was
    executed in, or the command line output of the code, are saved.

### `data` branch:

- contains a full cronicle of the whole research process
- all `exp-#` branches are merged here. A *Git tag* shows the merge
  commit to make accessing single experiments easy. The `LabBook.org` file with
  all experimental processes is moved to `data/exp-#/LabBook-exp-#.org` and a
  copy to `data/LabBook-all.org` for archiving.
