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
  pkgs-polars = import (builtins.fetchGit {
    name = "nixpkgs-unstable-for-polars";
    url = "https://github.com/NixOS/nixpkgs/";
    ref = "refs/heads/nixos-unstable";
    rev = "5fa4e5ce4cdfb1fb002889f3f65aa23e8b0b7425";
  }) {};
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
            scikit-image
            scikit-learn
            scipy
            seaborn
            tensorflow
            tqdm
          ]) ++ (
            with pkgs-polars.python312Packages; [
              polars
            ]);
      shellHook = ''
        jupyter lab
      '';
    };
  });
}
