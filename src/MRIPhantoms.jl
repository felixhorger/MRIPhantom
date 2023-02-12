
module MRIPhantoms
	
	using FFTW
	import FINUFFT
	import MRIRecon
	using SphericalCoordinates

	"""
		Width relative to shape

	"""
	function coil_sensitivities(shape::NTuple{N, Integer}, num_channels::NTuple{M, Integer}, width::Real) where {N, M}
		@assert N == M + 1
		# TODO: Phases?
		sensitivities = zeros(ComplexF64, shape..., prod(num_channels))
		width = width .* shape
		for (c, Φ) = enumerate(hypersphere(Val(N), num_channels))
			position = 0.25 .* shape .* (spherical2cartesian(Φ) .+ 2)
			for X in CartesianIndices(shape) # TODO: inbounds
				sensitivities[X, c] = @. exp(-0.5 * $sum((($Tuple(X) - position) / width)^2))
			end
		end
		return sensitivities
	end

	"""
		Random sensitivities, which have a large kernel in k-space to check the functionality of a recon pipeline
	"""
	function random_coil_sensitivities(shape::NTuple{N, Integer}, num_channels::Integer) where N
		return sensitivities = rand(ComplexF64, shape..., num_channels)
	end


	function ellipsoidal_shutter(shape::NTuple{N, Integer}, radii::Union{Real, NTuple{N, Real}}) where N
		shutter = Array{Bool, N}(undef, shape)
		centre = shape .÷ 2
		for X in CartesianIndices(shape)
			v = (Tuple(X) .- centre) ./ radii
			if sum(v.^2) <= 1
				shutter[X] = 1
			else
				shutter[X] = 0
			end
		end
		return shutter
	end


	"""
		a is upsampled array
		Last axis is channels
		shape is size of target kspace
	"""
	function measure(
		a::AbstractArray{<: Number, N},
		upsampling::NTuple{M, Integer},
		shape::NTuple{M, Integer}
	) where {N, M}
		@assert N == M + 1
		upsampled_shape = size(a)[1:M]
		@assert upsampled_shape == shape .* upsampling
		# Get location of target k-space
		centre_indices = MRIRecon.centre_indices.(upsampled_shape, shape)
		# FFT and extract target k-space
		kspace = fft(a, 1:M)
		kspace = fftshift(kspace, 1:M)
		kspace = @view kspace[centre_indices..., :]
		kspace = ifftshift(kspace, 1:M)
		# Normalise for different k-space sizes
		kspace ./= prod(upsampling)
		return kspace
	end
	for N ∈ 2:4
		ks = join(("k[$d, :]" for d = 1:N-1), ", ")
		@eval begin
			"""
				upsampling that _was_ applied to a
			"""
			function measure(
				a::AbstractArray{<: Number, $N},
				upsampling::NTuple{$(N-1), Integer},
				k::AbstractMatrix{<: Real};
				eps::Real=1e-12
			)
				@assert size(k, 1) == $(N-1)
				k = k ./ upsampling
				kspace = FINUFFT.$(Symbol("nufft$(N-1)d2"))($(Meta.parse(ks))..., -1, eps, a) ./ prod(upsampling)
				return kspace
			end
		end
	end


	function homogeneous(shape::NTuple{N, Integer}, upsampling::NTuple{N, Integer}) where N
		upsampled_shape = upsampling .* shape
		# Spatial profile of phantom
		highres_shutter = ellipsoidal_shutter(upsampled_shape, 0.4 .* upsampled_shape)
		# Downsample to get back to target resolution
		shutter = MRIRecon.downsample(highres_shutter, upsampling)
		return shutter, highres_shutter
	end

	function add_dynamic_dim(
		a::AbstractArray{<: Number, N},
		dynamic::AbstractVector{<: Number}
	) where N
		# Broadcast spatial dimensions and channels
		broadcasted_shape = (ntuple(_ -> 1, N)..., length(dynamic))
		a = reshape(dynamic, broadcasted_shape) .* a
		return a
	end

end

