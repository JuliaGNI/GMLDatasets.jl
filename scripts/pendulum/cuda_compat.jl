# Backport JuliaGPU/CUDA.jl#3087 for the CUDA version in scripts/Manifest.toml:
# https://github.com/JuliaGPU/CUDA.jl/pull/3087
# CuRef's GPU allocation must survive the device-to-host copy. CUDA 5.11.3 preserves only the
# CPU Ref, so GC can free the device pointer while a BLAS scalar result (e.g. norm) is read.
# Keep the upstream methods unchanged apart from this file's version guard. Remove this backport
# when the scripts environment moves to a CUDA release containing the upstream fix.
if pkgversion(CUDA) == v"5.11.3"
    @eval CUDA begin
        function Base.getindex(gpu::CuRefValue{T}) where {T}
            synchronize(gpu.buf)
            cpu = Ref{T}()
            GC.@preserve cpu gpu begin
                cpu_ptr = Base.unsafe_convert(Ptr{T}, cpu)
                gpu_ptr = Base.unsafe_convert(CuPtr{T}, gpu)
                unsafe_copyto!(cpu_ptr, gpu_ptr, 1; async=false)
            end
            cpu[]
        end

        function Base.getindex(gpu::CuRefArray{T}) where {T}
            synchronize(gpu.x)
            cpu = Ref{T}()
            GC.@preserve cpu gpu begin
                cpu_ptr = Base.unsafe_convert(Ptr{T}, cpu)
                gpu_ptr = pointer(gpu.x, gpu.i)
                unsafe_copyto!(cpu_ptr, gpu_ptr, 1; async=false)
            end
            cpu[]
        end
    end
end
