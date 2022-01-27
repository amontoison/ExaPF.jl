using ExaPF

using ExaPF.ForwardDiff
using ExaPF.LinearAlgebra
using ExaPF.SparseArrays
using ExaPF.CUDA
using ExaPF.CUDA.CUSPARSE
using ExaPF.KernelAbstractions

n = 1000000
d = 0.0001

A = sprandn(n,n,d)
x = ones(n)
y = zeros(n)
# Create duals
dx = ForwardDiff.Dual.(x)
dy = ForwardDiff.Dual.(y)

# Create CUDA
cuA = CuSparseMatrixCSR(A)
cux = CuVector{Float64}(x)
cuy = CuVector{Float64}(y)
dcux = cu(dx)
dcuy = cu(dy)

maxdist = 0
rowptr = Array(cuA.rowPtr)
for i in 2:n
    maxdist = max(maxdist, rowptr[i]-rowptr[i-1])
end
@show maxdist

ExaPF.LinearAlgebra.mul!(dcuy, cuA, dcux, 1.0, 0.0)
@time ExaPF.LinearAlgebra.mul!(dcuy, cuA, dcux, 1.0, 0.0)
ExaPF.mul2!(dcuy, cuA, dcux, 1.0, 0.0, maxdist)
CUDA.@time ExaPF.mul2!(dcuy, cuA, dcux, 1.0, 0.0, maxdist)
CUDA.@time ExaPF.LinearAlgebra.mul!(dcuy, cuA, dcux, 1.0, 0.0)
CUDA.@time begin
    mul!(cuy, cuA, cux)
    CUDA.synchronize()
end

using KernelAbstractions

@kernel function f_kernel(x)
    # i = @index(Global, Linear)
    # @show i
    i, j = @index(Global)
    @show i
    @show j

end

x = ones(100)
wait(f_kernel(CPU(), 3, 5)(x, ndrange=5))

