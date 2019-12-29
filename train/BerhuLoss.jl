using LearnBase

struct BerhuLoss{T<:AbstractFloat} <: DistanceLoss
    c::T   # boundary between quadratic and linear loss
    function BerhuLoss{T}(c::T) where T
        c > 0 || error("Berhu crossover parameter must be strictly positive.")
        new{T}(c)
    end
end

BerhuLoss(c::T=0.0) where {T<:AbstractFloat} = BerhuLoss{T}(c)  # default to L2
BerhuLoss(c) = BerhuLoss{Float64}(Float64(c))

function value(loss::BerhuLoss{T1}, difference::T2) where {T1,T2<:Number}
    T = promote_type(T1, T2)
    abs_diff = abs(difference)
    if abs_diff <= loss.c
        return convert(T, abs_diff)  # linear, abs diff
    else
        return convert(T, (abs2(loss.c)+abs2(abs_diff))/(2*loss.c))  # quadratic
    end
end

function deriv(loss::BerhuLoss{T1}, difference::T2) where {T1,T2<:Number}
    T = promote_type(T1, T2)
    if abs(difference) <= loss.c
        return convert(T, sign(difference))  # linear
    else
        return convert(T, difference/loss.c)  # quadratic
    end
end

function deriv2(loss::BerhuLoss{T1}, difference::T2) where {T1,T2<:Number}
    T = promote_type(T1, T2)
    abs(difference) <= loss.c ? zero(T) : sign(difference)*one(T)/loss.c
end

function value_deriv(loss::BerhuLoss{T1}, difference::T2) where {T1,T2<:Number}
    T = promote_type(T1, T2)
    abs_diff = abs(difference)
    if abs_diff <= loss.c
        val = convert(T, abs_diff)
        der = convert(T, sign(difference))
    else
        val = convert(T, (abs2(loss.c)+abs2(abs_diff))/(2*loss.c))
        der = convert(T, difference/loss.c)
    end
    return val, der
end

isdifferentiable(::BerhuLoss) = true
isdifferentiable(l::BerhuLoss, at) = true
istwicedifferentiable(::BerhuLoss) = false
istwicedifferentiable(l::BerhuLoss, at) = at != abs(l.c)
islipschitzcont(::BerhuLoss) = false  # unbounded derivative
isconvex(::BerhuLoss) = true
isstrictlyconvex(::BerhuLoss) = false
isstronglyconvex(::BerhuLoss) = false
issymmetric(::BerhuLoss) = true
