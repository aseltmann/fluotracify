{
  description = "Flake for exp-250327-masters branch of drmed-git repository";

  inputs = {
    # version of nixos-25.05 from 2025-12-18
    nixpkgs.url = "github:nixos/nixpkgs?ref=2b0d2b456e4e8452cf1c16d00118d145f31160f9";
    # version of nixos-unstable from 2025-07-06
    nixpkgs-unstable.url = "github:nixos/nixpkgs?ref=55b0d38442aac04f892f03b6a53cd9bb4c6cfc1c";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    { self, nixpkgs, nixpkgs-unstable, flake-utils, ...}:
    # Create system-specific outputs for the standard Nix systems
    # https://github.com/numtide/flake-utils/blob/main/lib.nix#L3-L9
    flake-utils.lib.eachDefaultSystem (
      system: let
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
      in {
        devShells.default = pkgs.mkShellNoCC {
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
