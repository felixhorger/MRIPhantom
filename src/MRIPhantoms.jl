
module MRIPhantoms
	
	using FFTW
	using Interpolations
	import FINUFFT
	import MRIRecon


	"""
		Width relative to shape

	"""
	# TODO: There is a coordinate transforms julia package
	# TODO: Not sure whether the 2D case should be made conform with the others in terms of types
	function spherical_coordinates(dim::Val{2}, num::Integer)
		return range(0, 2π * (1 - 1/num); length=num)
	end
	function spherical_coordinates(dim::Val{N}, num::NTuple{M, Integer}) where {N, M}
		@assert N > 2
		@assert M == N - 1
		offset = π / (num[M] + 2)
		return Base.Iterators.product(
			(range(0, 2π * (1 - 1/num[i]); length=num[i]) for i = 1:M-1)...,
			range(offset, π - offset; length=num[M])
		)
	end
	@generated function spherical2cartesian(Φ::NTuple{N, Real}) where N
		expr = :()
		retval = join(ntuple(i -> "x_$i", N+1), ", ")
		for i in 1:N
			expr = quote
				$expr
				sine, cosine = sincos(Φ[$i])
				$(Symbol("x_$i")) = cosine * prod_sines
				prod_sines *= sine
			end
		end
		return quote
			local prod_sines = 1
			$expr
			$(Symbol("x_$(N+1)")) = prod_sines
			return $(Meta.parse(retval))
		end
	end
	function coil_sensitivities(shape::NTuple{N, Integer}, num_channels::NTuple{M, Integer}, width::Real) where {N, M}
		@assert N == M + 1
		# TODO: Phases?
		sensitivities = zeros(ComplexF64, shape..., prod(num_channels))
		width = width .* shape
		for (c, Φ) = enumerate(spherical_coordinates(Val(N), num_channels))
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
		Sinc interpolation downsampling, in the first M axes
	"""
	function downsample(a::AbstractArray{T, N}, downsampling::NTuple{M, Integer}) where {N, M, T <: Number}
		shape = size(a)[1:M]
		downsampled_shape = divrem.(shape, downsampling)
		@assert all(p -> p[2] == 0, downsampled_shape) "downsampling must yield no remainder when dividing the size of a (except last dimension)"
		downsampled_shape = ntuple(d -> downsampled_shape[d][1], M)
		# Fourier transform
		b = fftshift(fft(a, 1:M), 1:M)
		# Extract centre indices, cutting off high frequency components
		centre_indices = MRIRecon.centre_indices.(shape, downsampled_shape)
		b = @view b[centre_indices..., ntuple(_ -> :, N-M)...]
		# Transform back
		interpolated = ifft(ifftshift(b, 1:M), 1:M) ./ prod(downsampling)
		return interpolated
	end

	"""
		Sinc interpolation upsampling, in the first M axes
	"""
	function upsample(a::AbstractArray{T, N}, upsampling::NTuple{M, Integer}) where {N, M, T <: Number}
		@assert N == M + 1
		shape = size(a)[1:M]
		residual_shape = size(a)[M+1:N]
		upsampled_shape = shape .* upsampling
		# Zero pad the Fourier transform of a
		centre_indices = MRIRecon.centre_indices.(upsampled_shape, shape)
		b = zeros(T, upsampled_shape..., residual_shape...)
		b[centre_indices..., ntuple(_ -> :, N-M)...] = fftshift(fft(a, 1:M), 1:M)
		# Transform back
		interpolated = ifft(ifftshift(b, 1:M), 1:M) .* prod(upsampling)
		return interpolated
	end


	"""
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
				k::AbstractMatrix{<: Real},
				eps::Real
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
		shutter = downsample(highres_shutter, upsampling)
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

