
module MRIPhantoms
	
	using FFTW
	using Interpolations
	import FINUFFT
	import MRIRecon


	"""
		Width relative to shape

	"""
	function coil_sensitivities(shape::NTuple{N, Integer}, num_channels::Integer, width::Real) where N
		# TODO: Phases?
		sensitivities = zeros(ComplexF64, shape..., num_channels)
		width = width .* shape
		for (c, α) = enumerate(range(0, 2π * (1 - 1/num_channels); length=num_channels))
			sine, cosine = sincos(α)
			position = @. 0.25 * shape * ((cosine, sine) + 2)
			for X in CartesianIndices(shape) # TODO: inbounds
				sensitivities[X, c] = @. exp(-0.5 * $sum((($Tuple(X) - position) / width)^2))
			end
		end
		return sensitivities
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


	function upsample(a::AbstractArray{T, N}, upsampling::NTuple{M, Integer}) where {N, M, T <: Number}
		@assert N == M + 1
		shape = size(a)[1:M]
		upsampled_shape = shape .* upsampling
		indices = ntuple(
			d -> range(1, size(a, d) + (upsampling[d] - 1) / upsampling[d]; length=upsampled_shape[d]),
			Val(M)
		)
		interpolated = Array{T, N}(undef, upsampled_shape..., size(a, N))
		@views for i in axes(a, N)
			interpolated[:, :, i] = extrapolate(interpolate(a[:, :, i], BSpline(Linear()), OnGrid()), Flat())[indices...]
		end
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


	function homogeneous(
		upsampling::NTuple{N, Integer},
		sensitivities::AbstractArray{<: Number, M}
	) where {N,M}
		@assert N == M - 1
		shape = size(sensitivities)[1:N]
		upsampled_shape = upsampling .* shape
		# Spatial profile of phantom
		shutter = ellipsoidal_shutter(upsampled_shape, 0.4 .* upsampled_shape)
		# Upsample sensitivities
		upsampled_sensitivities = upsample(sensitivities, upsampling)
		# Get individual coil images
		coil_images = upsampled_sensitivities .* shutter
		return coil_images, shutter
	end

	function add_dynamic_dim(
		dynamic::AbstractVector{<: Number},
		phantom::AbstractArray{<: Number, N},
		kspace::AbstractArray{<: Number, M}
	) where {N,M}
		# Broadcast spatial dimensions and channels
		dims = max(N, M)
		broadcasted_shape = (ntuple(_ -> 1, dims)..., length(dynamic))
		phantom = reshape(dynamic, broadcasted_shape[dims-N+1:dims+1]) .* phantom
		kspace = reshape(dynamic, broadcasted_shape[dims-M+1:dims+1]) .* kspace
		return phantom, kspace
	end

end

