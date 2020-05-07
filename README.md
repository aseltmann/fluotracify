# fluotracify - doctoral research project done in a reproducible way

This repository contains all code, most of the data, and a LabBook describing
the work I've done during my time in the Eggeling lab from January 2020 (after
working on the same project since ~ April 2019).

**This repository is still actively used and altered on a daily basis.**

## Getting started

### Viewing the electronic LabBook

#### On Github

The `LabBook.org` file can be viewed with markdown-like rendering here on
Github.

Note that the file looks different depending on the branch you are on, so
you might for example want to take a look at `LabBook.org` on the `develop`
branch for insight into ongoing exploratory research.

Note also that some elements of the file get lost while rendering and some
formatting errors might happen - viewing the *Raw* version of the file will show
you how it actually looks like.

#### On your computer

You can [download the develop
repository](https://github.com/aseltmann/fluotracify/archive/develop.zip) and
open `LabBook.html`as a nice html export of the LabBook.

Alternatively, you can open `LabBook.org` using any text editor. The file is
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

Download this repository or use [git](https://git-scm.com/) to clone a local
copy of for yourself:

``` sh
git clone https://github.com/aseltmann/fluotracify
```

The **src/fluotracify** folder is a python package which I will probably share
at some point in a dedicated way (e.g. pip). At the moment, you can add the
package to your own Python environment using a small, dirty hack:

``` python
import sys
sys.path.append('/path/to/fluotracify/src/')
```

Then you can import the fluotracify modules e.g. like this:

``` python
from fluotracify.training import build_model as bm
```

### Repeating the experiment or using the reproducible workflow for yourself

If you have a running git and Emacs setup (or an alternative editor who speaks
`org-mode`), then first get a local copy for yourself:

``` sh
git clone https://github.com/aseltmann/fluotracify
```

The `LabBook.org` file will guide you through the further process on how the
actual experiments were conducted.

Using the reproducible worklfow described here - and based on
[this paper by Stanisic et al.](https://hal.inria.fr/hal-01112795/document) -
has quite a steep learning curve. There is a
[free online
course](https://www.fun-mooc.fr/courses/course-v1:inria+41016+self-paced/5d99aa3742e34d6f87eed84b71fdde74/)
on reproducible research organized in part by the authors of the paper, which
might be the way to go, if you are not a fluent `git` and/or `Emacs` user.

## Organization of this repository - A reproducible workflow

**This description shows the ideal state of the repository. As of May 2020 I
still do a lot of exploratory work, so the "develop" branch will give you most
insight on ongoing experimental work.**

### master branch:

- contains all the source code in folder `src/` which is used for experiments.
- contains the `LabBook.org` template
- contains setup- and metadata files such as `MLproject` or `conda.yaml`
- the log contains only lasting alterations on the folders and files mentioned
  above, which are e.g. used for conducting experiments or which introduce new
  features. Day-to-day changes in code

### exp# branches:

- if an experiment is done, the code and templates will be branched out from
  *master* in an *exp#* branch, # substituting some meaningful descriptor.
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
  - Provenance data, such as Metadata about the environment the code was
    executed in, the command line output of the code, and some

### remote/origin/develop branch:

- this is the branch I use for day to day work on features and exploration.
  All of my current activity can be followed here.

### remote/origin/data branch:

- contains a full cronicle of the whole research process
- all *exp#* branches are merged here. Afterwards the original branch is
  deleted and on the data branch there is a *Git tag* which shows the merge
  commit to make accessing single experiments easy.
- the *develop* branch is merged here as well.

## License

The nanosimpy module found under `src/nanosimpy` is a fork of Dominic Waithe's
original nanosimpy library - here the original licensing applies.

All other content are free in the sense of freedom and licensed under the
Apache-2.0 License - see the [LICENSE](LICENSE) file for details.
