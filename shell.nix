with import <nixpkgs> { allowUnfree = true;}; {
	qpidEnv = stdenvNoCC.mkDerivation {
		name = "my-gcc12-environment";
		buildInputs = [
			gcc12
		];
		# shellHook = '' ????
		# 	export CUDA_PATH="/usr/local/cuda"
		# 	export LD_LIBRARY_PATH=/usr/lib/wsl/lib
		# '';}
	};
}
