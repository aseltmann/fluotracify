{
  description = "Flake for exp-250327-masters branch of drmed-git repository";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-25.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = {nixpkgs, flake-utils, ...}:

flake-utils.lib.eachDefaultSystem (
  system:
let
  pkgs = nixpkgs.legacyPackages.${system};
  multipletau-pypi = pkgs.python312Packages.buildPythonPackage rec {
    pname = "multipletau";
    version = "0.4.1";
    pyproject = true;
    src = pkgs.fetchPypi {
      inherit pname version;
      hash = "sha256-roP342FbjWKEtx32KSe6Ibgy0eiwLHgqoFwXUlnuVO8=";
    };
    build-system = with pkgs.python312Packages; [
      setuptools
      setuptools-scm
    ];
    dependencies = with pkgs.python312Packages; [
      numpy
    ];
  };
  polars-pypi = pkgs.python312Packages.buildPythonPackage rec {
    pname = "polars";
    version = "1.29.0";
    pyproject = true;
    src = pkgs.fetchPypi {
      inherit pname version;
      hash = "sha256-0qy3H84f8Op2219kir2Rp6bEYPr6vOmi6BdRhO+gDQI=";
    };
    # build-system = with pkgs.python312Packages; [
    #   setuptools
    #   setuptools-scm
    # ];
    # dependencies = with pkgs.python312Packages; [
    #   numpy
    # ];
  };
in
  {
    devShells.default = pkgs.mkShell {
      packages =
        with pkgs; [
          pdf2svg
          python312
        ] ++ (
          with pkgs.python312Packages; [
            click
            cython
            ipykernel
            ipywidgets
            jupyterlab
            lmfit
            matplotlib
            mlcroissant
            mlflow
            multipletau-pypi
            numpy
            pandas
            polars-pypi
            scikit-image
            scikit-learn
            scipy
            seaborn
            tensorflow
            tqdm
          ]);
      shellHook = ''
        jupyter lab
      '';
    };
  });
}
