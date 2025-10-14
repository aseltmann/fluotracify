{
  description = "Flake for exp-250327-masters branch of drmed-git repository";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-25.05";
    nixpkgs-unstable.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = {nixpkgs, nixpkgs-unstable, flake-utils, ...}:

flake-utils.lib.eachDefaultSystem (
  system:
let
  pkgs = nixpkgs.legacyPackages.${system};
  pkgs-unstable = nixpkgs-unstable.legacyPackages.${system};
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
            with pkgs-unstable.python312Packages; [
              polars
            ]);
      shellHook = ''
        jupyter lab
      '';
    };
  });
}
