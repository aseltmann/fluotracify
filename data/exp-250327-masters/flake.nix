{
  description = "Flake for exp-250327-masters branch of drmed-git repository";

  inputs = {
    # version of nixos-25.05 from 2025-12-18
    nixpkgs.url = "github:nixos/nixpkgs?ref=2b0d2b456e4e8452cf1c16d00118d145f31160f9";
    # version from 2025-07-06 for polars 1.31.0
    nixpkgs-polars.url = "github:nixos/nixpkgs?ref=55b0d38442aac04f892f03b6a53cd9bb4c6cfc1c";
    # version from 2022-08-27 for tmux 3.2a
    nixpkgs-tmux.url = "github:nixos/nixpkgs?ref=bf7d05e64d1172ad9356b87bc8c2a643f600e1f0";
    # version from 2024-05-19 for tensorflow 2.15
    nixpkgs-tensorflow.url = "github:nixos/nixpkgs?ref=bcdbb17a41d8fab94e8031a7c5e180fa5acd0809";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    { self, nixpkgs, nixpkgs-polars, nixpkgs-tmux, nixpkgs-tensorflow, flake-utils, ...}:
    # Create system-specific outputs for the standard Nix systems
    # https://github.com/numtide/flake-utils/blob/main/lib.nix#L3-L9
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = nixpkgs.legacyPackages.${system};
        pkgs-polars = nixpkgs-polars.legacyPackages.${system};
        pkgs-tmux = nixpkgs-tmux.legacyPackages.${system};
        pkgs-tensorflow = nixpkgs-tensorflow.legacyPackages.${system};
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
              python312  # 3.12.12
            ] ++ (
              with pkgs.python312Packages; [
                click  # 8.1.8
                cython  # 3.0.12
                ipykernel  # 6.29.5
                ipywidgets  # 8.1.5 (conda env: 8.1.7)
                jupyterlab  # 4.4.1
                lmfit  # 1.3.3
                matplotlib  # 3.10.1
                mlcroissant  # 1.0.17
                mlflow  # 2.20.3 (conda 2.21.3)
                multipletau-pypi  # 0.4.1
                numpy  # 2.2.5
                pandas  # 2.2.3
                scikit-image  # 0.25.2
                scikit-learn  # 1.6.1
                scipy  # 1.15.3 (conda 1.15.2)
                seaborn  # 0.13.2
                tqdm  # 4.67.1
              ]
            ) ++ (
              with pkgs-polars.python312Packages; [
                polars  # 1.31.0  # polars jumped from 1.27.1 to 1.31.0 in nixpkgs, choose higher version
              ]
            ) ++ (
              with pkgs-tensorflow.python312Packages; [
                keras # 2.15
                tensorflow-bin  # 2.15
              ]
            ) ++ (
              with pkgs-tmux; [
                tmux
              ]
            );
          # shellHook = ''

          # '';
        };
      });
}
