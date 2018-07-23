
import LinearAlgebra: size, mul!
import Base: *

struct ImplicitProd{T<:Number,TQ<:AbstractMatrix{T},TU<:AbstractMatrix{T}} <:AbstractMatrix{T}
    p1::AbstractVector{Int}
    Q::TQ
    p2::AbstractVector{Int}
    U::TU
end

const NOPERM = Base.OneTo(0)

function ImplicitProd(p1::AbstractVector{Int}, Q::AbstractMatrix{T}, p2::AbstractVector{Int}, U::AbstractMatrix{T}) where T<:Number
    
    @assert isempty(p1) || length(p1) == size(Q, 1) && isperm(p1)
    @assert isempty(p2) || length(p2) == size(Q, 2) && isperm(p2)
    @assert size(U, 2) <= size(U, 1) <= size(Q, 2)
    p1 = p1 == 1:length(p1) ? NOPERM : p1
    p2 = p2 == 1:length(p2) ? NOPERM : p2
    ImplicitProd{T, typeof(Q), typeof(U)}(p1, Q, p2, U) 
end

ImplicitProd(Q::AbstractMatrix{T}, U::AbstractMatrix{T}) where T =
ImplicitProd(1:0, Q, 1:0, U)

ImplicitProd(p1::AbstractVector{Int}, Q::AbstractMatrix{T}, U::AbstractMatrix{T}) where T =
ImplicitProd(p1, Q, 1:0, U)

ImplicitProd(Q::AbstractMatrix{T}, p2::AbstractVector{Int}, U::AbstractMatrix{T}) where T =
ImplicitProd(1:0, Q, p2, U)

Base.show(io::IO, A::ImplicitProd) = show(io, MIME("text/plain"), A)
Base.show(io::IO, ::MIME"text/plain", A::ImplicitProd) = Base.show_default(io, A)
Base.show(io::IO, A::Adjoint{<:Any,<:ImplicitProd}) = show(io, MIME("text/plain"), A)
Base.show(io::IO, ::MIME"text/plain", A::Adjoint{<:Any,<:ImplicitProd}) = Base.show_default(io, A)

#convert named tuple to explicit matrix
function Base.Matrix(A::ImplicitProd{T}) where T<:Number
    m, n, rk, nullity = sizes(A)
    X = zeros(T, m, n)
    X[1:rk,1:nullity] .= A.U
    for i = 1:m-rk
        X[rk+i,nullity+i] = T(1)
    end
    if ! isempty(A.p2)
        X = X[A.p2,:]
    end
    X = A.Q * X
    if ! isempty(A.p1)
        X = X[A.p1,:]
    end
    X
end

Base.Matrix(aA::Adjoint{<:Any,<:ImplicitProd}) = Matrix(Matrix(aA.parent)')

"""
    `m, n, rk, nullity = sizes(A::ImplicitProd)`

m is the column count of A.Q
n is the column count of A.X
rk is the row count of U
nullity is the column count of U
"""
@inline function sizes(A::ImplicitProd)
    rk, nullity = size(A.U)
    m = size(A.Q, 2)
    n = m - rk + nullity
    m, n, rk, nullity
end

size(A::ImplicitProd) = (size(A.Q, 1), size(A.U, 2) + size(A.Q, 2) - size(A.U, 1))

## *(A::ImplicitProd, B::AbstractVector) = mul!(similar(B, size(A, 1)), A, B)

function mul!(C::AbstractVector, A::ImplicitProd, B::AbstractVector)
    checkdims(C, A, B)
    m, n, rk, nullity = sizes(A)
    @inbounds C[1:rk] .= A.U * B[1:nullity]
    @inbounds C[rk+1:end] .= B[nullity+1:end]
    if ! isempty(A.p2)
        C[:] .= C[A.p2]
    end
    lmul!(A.Q, C)
    if ! isempty(A.p1)
        @inbounds C[:] .= C[A.p1]
    end
    C
end

## *(A::ImplicitProd, B::AbstractMatrix) = mul!(similar(B, size(A, 1), size(B, 2)), A, B)

function mul!(C::AbstractMatrix, A::ImplicitProd, B::AbstractMatrix)
    checkdims(C, A, B)
    m, n, rk, nullity = sizes(A)
    @inbounds C[1:rk,:] .= A.U * B[1:nullity,:]
    @inbounds C[rk+1:end,:] .= B[nullity+1:end,:]
    if ! isempty(A.p2)
        @inbounds C[:,:] .= C[A.p2,:]
    end
    lmul!(A.Q, C)
    if ! isempty(A.p1)
        @inbounds C[:,:] .= C[A.p1,:]
    end
    C
end

## *(A::Adjoint{<:Any,<:ImplicitProd}, B::AbstractVector) = mul!(similar(B, size(A, 1)), A, B)

function mul!(C::AbstractVector, aA::Adjoint{<:Any,<:ImplicitProd}, B::AbstractVector)
    checkdims(C, aA, B)
    A = aA.parent
    m, n, rk, nullity = sizes(A)
    if ! isempty(A.p1)
        D = similar(B)
        @inbounds D[A.p1] .= B
    else
        D = B
    end
    lmul!(A.Q', D)
    if ! isempty(A.p2)
        @inbounds D[A.p2] .= D
    end
    @inbounds C[1:nullity] = A.U' * D[1:rk]
    @inbounds C[nullity+1:end] = D[rk+1:end]
    C
end

## *(A::Adjoint{<:Any,<:ImplicitProd}, B::AbstractMatrix) = mul!(similar(B, size(A, 1), size(B, 2)), A, B)

function mul!(C::AbstractMatrix, aA::Adjoint{<:Any,<:ImplicitProd}, B::AbstractMatrix)
    checkdims(C, aA, B)
    A = aA.parent
    m, n, rk, nullity = sizes(A)
    if ! isempty(A.p1)
        D = similar(B)
        D[A.p1,:] .= B
    else
        D = B
    end
    E = A.Q' * D # mul! not defined for adjoints of Q mul!(similar(D), A.Q', D)
    if ! isempty(A.p2)
        E[A.p2,:] .= E
    end
    C[1:nullity,:] .= A.U' * E[1:rk,:]
    C[nullity+1:end,:] .= E[rk+1:end,:]
    C
end

function mul!(C::AbstractMatrix, aA::Adjoint{<:Any,<:ImplicitProd}, aB::Adjoint{<:Any,<:AbstractMatrix})
    
    mul!(C, aA, copy(aB))
end

mul!(C::AbstractArray, A::UniformScaling, B::AbstractArray) = ( C .= B; C .*= A[1,1]; C )

function checkdims(C, A, B)
    m, n = size(A, 2), size(B, 1)
    if m != n
        text = "second dimension of A, $m, does not match first dimension of B, $n"
        throw(DimensionMismatch(text))
    end
    if size(C, 1) != size(A, 1) || size(C, 2) != size(B, 2)
        text = "destination dimensions $(size(C)) do not match result dimensions ($(size(A, 1)),$(size(B, 2)))"
        throw(DimensionMismatch(text))
    end
end
