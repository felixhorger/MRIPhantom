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
		total_num_channels = prod(num_channels)
		sensitivities = Array{ComplexF64, N+1}(undef, shape..., total_num_channels)
		width = width .* shape
		centre = shape .÷ 2
		mask_width = 0.4 .* shape
		num_channels = (num_channels[1] - 1, num_channels[2:end]...)
		for (c, Φ) = enumerate(hypersphere(Val(N), num_channels))
			#position = 0.25 .* shape .* (spherical2cartesian(Φ) .+ 2)
			position = shape .* (0.35 .* spherical2cartesian(Φ) .+ 0.5)
			for X in CartesianIndices(shape) # TODO: inbounds
				#sensitivities[X, c] = exp(-0.5 * (sum(((Tuple(X) .- position) ./ width).^2 .+ ((Tuple(X) .- centre) ./ mask_width).^6)))
				sensitivities[X, c] = exp(-0.5 * sum(((Tuple(X) .- position) ./ width).^2 ))
			end
		end
		for X in CartesianIndices(shape) # TODO: inbounds
			#sensitivities[X, total_num_channels] = exp(-0.5 * sum(((Tuple(X) .- centre) ./ width).^2))
			sensitivities[X, total_num_channels] = exp(-0.5 * sum(((Tuple(X) .- centre) ./ width).^2))
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

	function stamp!(a::AbstractArray{<: Number, N}, mask::AbstractArray{Bool, N}, position::CartesianIndex{N}, v::Number) where N
		position = position - CartesianIndex{N}(size(mask) .÷ 2)
		for i in CartesianIndices(mask)
			j = i + position
			!checkbounds(Bool, a, j) && continue
			a[j] += v * mask[i]
		end
		return a
	end


	"""
		a is upsampled array
		last axis is channels, dynamic etc
		shape is size of target kspace
		TODO: isn't it redundant to have shape, as only the last dim of a is "extra" (e.g. channels), and so shpae can be inferred, there even is an assert...
	"""
	function measure(
		a::AbstractArray{<: Number, N},
		upsampling::NTuple{M, Integer},
		shape::NTuple{M, Integer}
	) where {N, M}
		@assert M + 1 == N
		upsampled_shape = size(a)[1:M]
		@assert upsampled_shape == shape .* upsampling
		# Get location of target k-space
		centre_indices = MRIRecon.centre_indices.(upsampled_shape, shape)
		# FFT and extract target k-space
		kspace = fft(a, 1:M)
		kspace = fftshift(kspace, 1:M) # TODO: allocate tmp array and use fftshift!(), as this can get very large due to oversampling and channels etc
		kspace_centre = @view kspace[centre_indices..., :]
		lowres_kspace = ifftshift(kspace_centre, 1:M)
		# Normalise for different k-space sizes
		lowres_kspace ./= prod(upsampling)
		return lowres_kspace
	end
	for N ∈ 2:4
		ks = join(("k[$d, :]" for d = 1:N-1), ", ")
		@eval begin
			"""
				upsampling that _was_ applied to a
				need to reshape a if channels and dynamic
			"""
			function measure(
				a::AbstractArray{<: Number, $N},
				upsampling::NTuple{$(N-1), Integer},
				k::AbstractMatrix{<: Real};
				eps::Real=1e-8,
				kwargs...
			)
				@assert size(k, 1) == $(N-1)
				k = k ./ upsampling
				kspace = FINUFFT.$(Symbol("nufft$(N-1)d2"))($(Meta.parse(ks))..., -1, eps, a; kwargs...) ./ prod(upsampling)
				return kspace
			end
		end
	end


	function homogeneous(shape::NTuple{N, Integer}, upsampling::NTuple{N, Integer}; radius::Union{Real, NTuple{N, Real}}=0.4) where N
		upsampled_shape = upsampling .* shape
		# Spatial profile of phantom
		highres_shutter = ellipsoidal_shutter(upsampled_shape, radius .* upsampled_shape)
		# Downsample to get back to target resolution
		shutter = MRIRecon.downsample(highres_shutter, upsampling)
		return shutter
	end

	function nist_slice(
		shape::NTuple{N, Integer};
		intensities::AbstractVector{T}=ones(9)
	) where {N, T <: Number}
		@assert N == 2
		# Spatial profile of phantom
		phantom = zeros(T, shape)
		body = ellipsoidal_shutter(shape, 0.4 .* shape)
		centre = CartesianIndex(shape .÷ 2)
		stamp!(phantom, body, centre, intensities[1])
		circle = ellipsoidal_shutter(shape, 0.05 .* shape)
		r = 0.275 .* shape # circles located on circle of this radius
		for (i, ϕ) in enumerate(range(0, 2π; length=8+1)[1:8])
			sine, cosine = sincos(ϕ)
			position = CartesianIndex{2}(floor.(Int, r .* (cosine, sine))) + centre
			stamp!(phantom, circle, position, intensities[i+1] .- 1.0)
		end
		return phantom
	end

	"""
		dynamics[time, label]
	"""
	function dynamic_from_segmentation(labels::AbstractArray{<: Integer, N}, dynamics::AbstractMatrix{T}; outtype::Type=T) where {T, N}
		phantom = Array{outtype, N+1}(undef, size(labels), size(dynamic, 1))
		for I in CartesianIndices(labels)
			@views phantom[I, :] = dynamics[:, labels[I]]
		end
		return phantom
	end

	function add_dynamic_dim(
		a::AbstractArray{<: Number, N},
		dynamic::AbstractVector{<: Number}
	) where N
		# Broadcast spatial dimensions and channels
		broadcasted_shape = (ntuple(_ -> 1, N)..., length(dynamic))
		return reshape(dynamic, broadcasted_shape) .* a
	end

	function add_dynamic_dim(
		a::AbstractArray{T, N},
		dynamics::AbstractMatrix{<: Number},
		indices::AbstractArray{<: Integer, N}
	) where {T <: Number, N}
		# Broadcast spatial dimensions and channels
		num_dynamic = size(dynamics, 1)
		ad = Array{T, N+1}(undef, size(a)..., num_dynamic)
		@views for i in CartesianIndices(indices)
			if indices[i] == 0
				ad[i, :] .= 0.0
			else
				ad[i, :] .= dynamics[:, indices[i]] .* a[i]
			end
		end
		return ad
	end

end

