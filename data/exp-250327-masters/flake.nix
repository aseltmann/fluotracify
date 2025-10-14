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
  multipletau = pkgs.python3Packages.buildPythonPackage rec {
    pname = "multipletau";
    version = "0.4.1";
    pyproject = true;
    src = pkgs.fetchFromGitHub {
      owner = "FCS-analysis";
      repo = pname;
      rev = version;
      sha256 = "sha256-uFcHqKrLELsOxkr1WKQkzLtLjl/LFjg+vzfSW6UFezU=";
    };
    # build-system = with pkgs.python3Packages; [ setuptools ];
    dependencies = with pkgs.python3Packages; [
      setuptools
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
            multipletau
            numpy
            pandas
            polars
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
