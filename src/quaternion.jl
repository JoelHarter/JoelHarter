# Copyright 2026 Joel T Harter
# Adapted from Julia's complex.jl (MIT: https://julialang.org/license).
using StaticArrays
using LinearAlgebra: norm
using Random: Random, AbstractRNG, SamplerType
import Base: Complex, +, -, *, /, \, ^, ==, isequal, hash, show, read, write, bswap, conj, abs, abs2, inv,
    real, imag, isreal, isinteger, isfinite, isnan, isinf, iszero, isone, promote_rule, widen, float,
    big, flipsign, in, angle, round, sqrt, exp, exp2, exp10, expm1, log, log2, log10, log1p,
    sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, asinh, acosh, atanh, getproperty, propertynames

"""
    Quaternion{T<:Real} <: Number

Quaternion `w + x*im + y*jm + z*km` with scalar part `w` and vector part `v = (x, y, z)`, all of
type `T`. Stored as four numbers of type `T` in a row, like two for `Complex`. The parts are
`q.w`, `q.x`, `q.y`, `q.z`, and the vector part as a whole is `q.v`.

`QuaternionF16`, `QuaternionF32` and `QuaternionF64` are aliases for
`Quaternion{Float16}`, `Quaternion{Float32}` and `Quaternion{Float64}` respectively.

Reals and complex numbers promote to quaternions, and `im` from `Complex` is reused as the first
imaginary unit, so `1 + 2im + 3jm + 4km` builds a `Quaternion{Int}`.

See also: [`Complex`](@ref), [`quaternion`](@ref), [`real`](@ref), [`imag`](@ref), [`lift`](@ref).
"""
struct Quaternion{T<:Real} <: Number
    w::T
    v::SVector{3,T}
end
Quaternion(w::Real, v::AbstractVector{<:Real}) = Quaternion{promote_type(typeof(w), eltype(v))}(w, v)
Quaternion(w::Real, x::Real, y::Real, z::Real) = Quaternion(promote(w, x, y, z)...)
Quaternion(w::T, x::T, y::T, z::T) where {T<:Real} = Quaternion{T}(w, SVector(x, y, z))
Quaternion(x::Real) = Quaternion(x, zero(x), zero(x), zero(x))
Quaternion(z::Complex) = Quaternion(real(z), imag(z), zero(real(z)), zero(real(z)))
Quaternion(v::AbstractVector{<:Real}) = Quaternion(zero(eltype(v)), v) # pure quaternion, 0 + v
Quaternion(q::Quaternion) = q

getproperty(q::Quaternion, s::Symbol) =
    s === :x ? getfield(q, :v)[1] :
    s === :y ? getfield(q, :v)[2] :
    s === :z ? getfield(q, :v)[3] : getfield(q, s)
propertynames(::Quaternion) = (:w, :x, :y, :z, :v)

"""
    jm, km

The second and third quaternion units, after `im`: `im*jm == km`, `jm*km == im`, `km*im == jm`,
and `im^2 == jm^2 == km^2 == im*jm*km == -1`.

# Examples
```jldoctest
julia> jm * km
1 + 0im + 0jm + 0km

julia> (1 + 2im + 3jm + 4km)^2
-28 + 4im + 6jm + 8km
```
"""
const jm = Quaternion(false, false, true, false)
const km = Quaternion(false, false, false, true)

const QuaternionF64 = Quaternion{Float64}
const QuaternionF32 = Quaternion{Float32}
const QuaternionF16 = Quaternion{Float16}

Quaternion{T}(w::Real, x::Real, y::Real, z::Real) where {T<:Real} = Quaternion{T}(w, SVector{3,T}(x, y, z))
Quaternion{T}(x::Real) where {T<:Real} = Quaternion{T}(x, 0, 0, 0)
Quaternion{T}(z::Complex) where {T<:Real} = Quaternion{T}(real(z), imag(z), 0, 0)
Quaternion{T}(q::Quaternion) where {T<:Real} = Quaternion{T}(q.w, q.v)
(::Type{T})(q::Quaternion) where {T<:Real} =
    isreal(q) ? T(q.w)::T : throw(InexactError(nameof(T), T, q))
Complex{T}(q::Quaternion) where {T<:Real} =
    iszero(q.y) & iszero(q.z) ? Complex{T}(q.w, q.x) : throw(InexactError(:Complex, Complex{T}, q))
Complex(q::Quaternion) = iszero(q.y) & iszero(q.z) ? Complex(q.w, q.x) : throw(InexactError(:Complex, Complex, q))

promote_rule(::Type{Quaternion{T}}, ::Type{S}) where {T<:Real,S<:Real} =
    Quaternion{promote_type(T,S)}
promote_rule(::Type{Quaternion{T}}, ::Type{Complex{S}}) where {T<:Real,S<:Real} =
    Quaternion{promote_type(T,S)}
promote_rule(::Type{Quaternion{T}}, ::Type{Quaternion{S}}) where {T<:Real,S<:Real} =
    Quaternion{promote_type(T,S)}

widen(::Type{Quaternion{T}}) where {T} = Quaternion{widen(T)}

float(::Type{Quaternion{T}}) where {T<:AbstractFloat} = Quaternion{T}
float(::Type{Quaternion{T}}) where {T} = Quaternion{float(T)}

"""
    real(q::Quaternion)

Return the scalar part `w` of the quaternion `q`.

# Examples
```jldoctest
julia> real(1 + 2im + 3jm + 4km)
1
```
"""
real(q::Quaternion) = q.w

"""
    imag(q::Quaternion)

Return the vector part `(x, y, z)` of the quaternion `q` as an `SVector`.

# Examples
```jldoctest
julia> imag(1 + 2im + 3jm + 4km)
3-element SVector{3, Int64} with indices SOneTo(3):
 2
 3
 4
```
"""
imag(q::Quaternion) = q.v

real(Q::Type{<:Quaternion}) = fieldtype(Q, 1)

isreal(q::Quaternion) = iszero(q.v)
isinteger(q::Quaternion) = isreal(q) & isinteger(q.w)
isfinite(q::Quaternion) = isfinite(q.w) & all(isfinite, q.v)
isnan(q::Quaternion) = isnan(q.w) | any(isnan, q.v)
isinf(q::Quaternion) = isinf(q.w) | any(isinf, q.v)
iszero(q::Quaternion) = iszero(q.w) & iszero(q.v)
isone(q::Quaternion) = isone(q.w) & iszero(q.v)

"""
    quaternion(w, [x, y, z])
    quaternion(w, v)

Convert real or complex numbers or arrays to quaternion. The vector part defaults to zero.

# Examples
```jldoctest
julia> quaternion(7)
7 + 0im + 0jm + 0km

julia> quaternion(1 + 2im)
1 + 2im + 0jm + 0km

julia> quaternion([1, 2])
2-element Vector{Quaternion{Int64}}:
 1 + 0im + 0jm + 0km
 2 + 0im + 0jm + 0km
```
"""
quaternion(q::Quaternion) = q
quaternion(x::Union{Real,Complex}) = Quaternion(x)
quaternion(w::Real, v::AbstractVector{<:Real}) = Quaternion(w, v)
quaternion(w::Real, x::Real, y::Real, z::Real) = Quaternion(w, x, y, z)

"""
    quaternion(T::Type)

Return an appropriate type which can represent a value of type `T` as a quaternion.
Equivalent to `typeof(quaternion(zero(T)))`.

# Examples
```jldoctest
julia> quaternion(Complex{Int})
Quaternion{Int64}

julia> quaternion(Float64)
Quaternion{Float64}
```
"""
quaternion(::Type{T}) where {T<:Real} = Quaternion{T}
quaternion(::Type{Complex{T}}) where {T<:Real} = Quaternion{T}
quaternion(::Type{Quaternion{T}}) where {T<:Real} = Quaternion{T}

flipsign(q::Quaternion, y::Real) = ifelse(signbit(y), -q, q)

function show(io::IO, q::Quaternion)
    compact = get(io, :compact, false)::Bool
    show(io, q.w)
    for (c, unit) in zip(q.v, ("im", "jm", "km"))
        if signbit(c) && !isnan(c)
            print(io, compact ? "-" : " - ")
            if isa(c,Signed) && !isa(c,BigInt) && c == typemin(typeof(c))
                show(io, -widen(c))
            else
                show(io, -c)
            end
        else
            print(io, compact ? "+" : " + ")
            show(io, c)
        end
        if !(isa(c,Integer) && !isa(c,Bool) || isa(c,AbstractFloat) && isfinite(c))
            print(io, "*")
        end
        print(io, unit)
    end
end
show(io::IO, q::Quaternion{Bool}) =
    print(io, q == im ? "im" : q == jm ? "jm" : q == km ? "km" : "Quaternion($(q.w),$(q.x),$(q.y),$(q.z))")

function Base.show_unquoted(io::IO, q::Quaternion, ::Int, prec::Int)
    if Base.operator_precedence(:+) <= prec
        print(io, "(")
        show(io, q)
        print(io, ")")
    else
        show(io, q)
    end
end

function read(s::IO, ::Type{Quaternion{T}}) where T<:Real
    w = read(s,T)
    x = read(s,T)
    y = read(s,T)
    z = read(s,T)
    Quaternion{T}(w,x,y,z)
end
function write(s::IO, q::Quaternion)
    write(s,q.w,q.x,q.y,q.z)
end

## byte order swaps: each part is swapped individually
bswap(q::Quaternion) = Quaternion(bswap(q.w), bswap.(q.v))

## equality and hashing of quaternions ##

==(q::Quaternion, r::Quaternion) = (q.w == r.w) & (q.v == r.v)
==(q::Quaternion, z::Complex) = iszero(q.y) & iszero(q.z) && (q.w == real(z)) & (q.x == imag(z))
==(z::Complex, q::Quaternion) = q == z
==(q::Quaternion, x::Real) = isreal(q) && q.w == x
==(x::Real, q::Quaternion) = isreal(q) && q.w == x

isequal(q::Quaternion, r::Quaternion) = isequal(q.w,r.w)::Bool & isequal(q.v,r.v)::Bool
isequal(q::Quaternion, z::Complex) = isequal(q.w,real(z))::Bool & isequal(q.x,imag(z))::Bool & isequal(q.y,zero(q.y))::Bool & isequal(q.z,zero(q.z))::Bool
isequal(z::Complex, q::Quaternion) = isequal(q, z)
isequal(q::Quaternion, x::Real) = isequal(q.w,x)::Bool & isequal(q.v,zero(q.v))::Bool
isequal(x::Real, q::Quaternion) = isequal(q, x)

in(q::Quaternion, r::AbstractRange{<:Real}) = isreal(q) && q.w in r

# Hashes like the complex or real number it equals: the jm and km terms vanish when y and z are 0,
# just as Base's im term vanishes when the imaginary part is 0.
if UInt === UInt64
    const h_jmag = 0x6d9c3e5b4a8f2d17
    const h_kmag = 0x2f5b8e1c7a3d9046
else
    const h_jmag = 0x4a8f2d17
    const h_kmag = 0x7a3d9046
end
const hash_0_jmag = hash(0, h_jmag)
const hash_0_kmag = hash(0, h_kmag)

function hash(q::Quaternion, h::UInt)
    hash(Complex(q.w, q.x), h ⊻ hash(q.y, h_jmag) ⊻ hash_0_jmag ⊻ hash(q.z, h_kmag) ⊻ hash_0_kmag)
end

## generic functions of quaternions ##

"""
    conj(q::Quaternion)

Compute the conjugate of a quaternion `q`, negating its vector part.

# Examples
```jldoctest
julia> conj(1 + 2im + 3jm + 4km)
1 - 2im - 3jm - 4km
```
"""
conj(q::Quaternion) = Quaternion(q.w, -q.v)
abs(q::Quaternion)  = hypot(q.w, q.x, q.y, q.z)
abs2(q::Quaternion) = abs2(q.w) + sum(abs2, q.v)
inv(q::Quaternion) = conj(q)/abs2(q)
inv(q::Quaternion{<:Integer}) = inv(float(q))
# As Base does for ComplexF64: abs2 overflows or underflows when the parts are near the edges of the
# float range, so outside the safe range scale by a power of two first, which is exact.
function inv(q::Quaternion{T}) where {T<:AbstractFloat}
    m = max(abs(q.w), maximum(abs, q.v))
    sqrt(floatmin(T)/4) <= m <= sqrt(floatmax(T)/4) && return conj(q)/abs2(q)
    (isinf(q.w) | any(isinf, q.v)) && return Quaternion(copysign(zero(q.w), q.w), flipsign.(-zero(q.v), q.v))
    (iszero(m) | isnan(m)) && return conj(q)/abs2(q)
    e = exponent(m)
    s = Quaternion(ldexp(q.w, -e), ldexp.(q.v, -e))
    r = conj(s)/abs2(s)
    Quaternion(ldexp(r.w, -e), ldexp.(r.v, -e))
end

+(q::Quaternion) = Quaternion(+q.w, +q.v)
-(q::Quaternion) = Quaternion(-q.w, -q.v)
+(q::Quaternion, r::Quaternion) = Quaternion(q.w + r.w, q.v + r.v)
-(q::Quaternion, r::Quaternion) = Quaternion(q.w - r.w, q.v - r.v)
# Hamilton product: (a + u)(b + v) = ab - u⋅v + av + bu + u×v, written out because it compiles a
# few percent faster than the dot/cross form. Not commutative.
*(q::Quaternion, r::Quaternion) = Quaternion(q.w*r.w - q.x*r.x - q.y*r.y - q.z*r.z,
                                             q.w*r.x + q.x*r.w + q.y*r.z - q.z*r.y,
                                             q.w*r.y - q.x*r.z + q.y*r.w + q.z*r.x,
                                             q.w*r.z + q.x*r.y - q.y*r.x + q.z*r.w)

# handle Bool and Quaternion{Bool}
# avoid type signature ambiguity warnings
+(x::Bool, q::Quaternion{Bool}) = Quaternion(x + q.w, q.v)
+(q::Quaternion{Bool}, x::Bool) = Quaternion(q.w + x, q.v)
-(x::Bool, q::Quaternion{Bool}) = Quaternion(x - q.w, -q.v)
-(q::Quaternion{Bool}, x::Bool) = Quaternion(q.w - x, q.v)
*(x::Bool, q::Quaternion{Bool}) = Quaternion(x * q.w, x * q.v)
*(q::Quaternion{Bool}, x::Bool) = Quaternion(q.w * x, q.v * x)

+(x::Bool, q::Quaternion) = Quaternion(x + q.w, q.v)
+(q::Quaternion, x::Bool) = Quaternion(q.w + x, q.v)
-(x::Bool, q::Quaternion) = Quaternion(x - q.w, -q.v)
-(q::Quaternion, x::Bool) = Quaternion(q.w - x, q.v)
*(x::Bool, q::Quaternion) = Quaternion(x * q.w, x * q.v)
*(q::Quaternion, x::Bool) = Quaternion(q.w * x, q.v * x)

+(x::Real, q::Quaternion{Bool}) = Quaternion(x + q.w, q.v)
+(q::Quaternion{Bool}, x::Real) = Quaternion(q.w + x, q.v)
-(x::Real, q::Quaternion{Bool}) = Quaternion(x - q.w, -q.v)
-(q::Quaternion{Bool}, x::Real) = Quaternion(q.w - x, q.v)
*(x::Real, q::Quaternion{Bool}) = Quaternion(x * q.w, x * q.v)
*(q::Quaternion{Bool}, x::Real) = Quaternion(q.w * x, q.v * x)

# adding or multiplying real & quaternion is common; complex & quaternion go through promotion
+(x::Real, q::Quaternion) = Quaternion(x + q.w, q.v)
+(q::Quaternion, x::Real) = Quaternion(x + q.w, q.v)
-(x::Real, q::Quaternion) = Quaternion(x - q.w, -q.v)
-(q::Quaternion, x::Real) = Quaternion(q.w - x, q.v)
*(x::Real, q::Quaternion) = Quaternion(x * q.w, x * q.v)
*(q::Quaternion, x::Real) = Quaternion(x * q.w, x * q.v)

# Division is not commutative either: q / r = q * inv(r) and q \\ r = inv(q) * r.
/(q::Quaternion, x::Real) = Quaternion(q.w/x, q.v/x)
/(x::Real, q::Quaternion) = x * inv(q)
/(q::Quaternion, r::Quaternion) = q * inv(r)
\(x::Real, q::Quaternion) = q / x
\(q::Quaternion, x::Real) = inv(q) * x
\(q::Quaternion, r::Quaternion) = inv(q) * r

"""
    lift(f, q::Quaternion)

Apply the complex function `f` to the quaternion `q`. Every quaternion is `w + n*u` for a unit
vector `u` with `u^2 == -1`, so on its own plane it is the complex number `w + n*im`, and any
function analytic on the complex plane extends to quaternions by `f(w + n*u) = a + b*u` where
`f(w + n*im) = a + b*im`. All the elementary functions below are defined this way, keeping
Base's branch cuts and special values. A real quaternion has no plane of its own, so it uses
`im`'s, like `Complex` does: `sqrt(-1 + 0jm) == im`.

# Examples
```jldoctest
julia> exp(π/2 * jm) ≈ jm
true

julia> lift(z -> z^3, 1 + km) == (1 + km)^3
true
```
"""
function lift(f, q::Quaternion)
    n = norm(q.v)
    z = f(Complex(q.w, n))
    iszero(n) ? Quaternion(z) : Quaternion(real(z), (imag(z)/n) * q.v)
end

for f in (:sqrt, :exp, :exp2, :exp10, :expm1, :log, :log2, :log10, :log1p,
          :sin, :cos, :tan, :asin, :acos, :atan, :sinh, :cosh, :tanh, :asinh, :acosh, :atanh)
    @eval $f(q::Quaternion) = lift($f, q)
end

"""
    angle(q::Quaternion)

Compute the angle in radians between the quaternion `q` and the real axis: the `θ` in the polar
form `q = abs(q) * (cos(θ) + u*sin(θ))` for a unit vector `u` along the vector part.
Returns a number `0 ≤ angle(q) ≤ pi`.

# Examples
```jldoctest
julia> rad2deg(angle(1 + jm))
45.0

julia> rad2deg(angle(-1 + 1e-20km))
180.0
```
"""
angle(q::Quaternion) = atan(norm(q.v), q.w)

^(q::Quaternion, p::Real) = lift(z -> z^p, q)
^(q::Quaternion, p::Rational) = lift(z -> z^p, q) # to resolve ambiguity
^(q::Quaternion, n::Integer) = n >= 0 ? Base.power_by_squaring(q,n) : Base.power_by_squaring(inv(q),-n)
^(q::Quaternion{<:Integer}, n::Integer) = Base.power_by_squaring(q,n) # DomainError for n<0
^(q::Quaternion, n::Bool) = n ? q : one(q)
^(q::Quaternion{<:Integer}, n::Bool) = n ? q : one(q) # to resolve ambiguity
^(::Irrational{:ℯ}, q::Quaternion) = exp(q)
# With a non-real exponent there is no single natural definition, since exp(log(q)*p) and
# exp(p*log(q)) differ; this is the former.
^(q::Quaternion, p::Union{Complex,Quaternion}) = exp(log(q) * p)
^(x::Union{Real,Complex}, q::Quaternion) = exp(log(x) * q)

#Rounding quaternions
#Requires two different RoundingModes for the scalar and vector parts
"""
    round(q::Quaternion[, RoundingModeScalar, [RoundingModeVector]])
    round(q::Quaternion[, RoundingModeScalar, [RoundingModeVector]]; digits=0, base=10)
    round(q::Quaternion[, RoundingModeScalar, [RoundingModeVector]]; sigdigits, base=10)

Return the nearest integral value of the same type as the quaternion `q` to `q`,
breaking ties using the specified [`RoundingMode`](@ref)s. The first
[`RoundingMode`](@ref) is used for rounding the scalar part while the
second is used for rounding the vector part.

`RoundingModeScalar` and `RoundingModeVector` default to [`RoundNearest`](@ref),
which rounds to the nearest integer, with ties (fractional values of 0.5)
being rounded to the nearest even integer.

# Examples
```jldoctest
julia> round(3.14 + 4.5jm)
3.0 + 0.0im + 4.0jm + 0.0km

julia> round(3.14 + 4.5jm, RoundUp, RoundNearestTiesUp)
4.0 + 0.0im + 5.0jm + 0.0km

julia> round(3.14159 + 4.512km; digits = 1)
3.1 + 0.0im + 0.0jm + 4.5km
```
"""
function round(q::Quaternion, rw::RoundingMode=RoundNearest, rv::RoundingMode=rw; kwargs...)
    Quaternion(round(q.w, rw; kwargs...),
               round.(q.v, rv; kwargs...))
end

float(q::Quaternion{<:AbstractFloat}) = q
float(q::Quaternion) = Quaternion(float(q.w), float(q.v))

big(::Type{Quaternion{T}}) where {T<:Real} = Quaternion{big(T)}
big(q::Quaternion{T}) where {T<:Real} = Quaternion{big(T)}(q)

## random quaternions, as Random does complex numbers ##

# rand: each part uniform on [0, 1), the unit 4-cube, like rand(ComplexF64) fills the unit square.
Random.rand(r::AbstractRNG, ::SamplerType{Quaternion{T}}) where {T<:Real} =
    Quaternion(rand(r, T), rand(r, T), rand(r, T), rand(r, T))
# randn: four normal parts of variance 1/4, so E(abs2(q)) == 1 as for randn(ComplexF64). A uniformly
# random unit quaternion, and so a uniformly random rotation, is sign(randn(QuaternionF64)).
Random.randn(r::AbstractRNG, ::Type{Quaternion{T}}) where {T<:AbstractFloat} =
    Quaternion(randn(r, T), randn(r, T), randn(r, T), randn(r, T)) / 2

## Array operations on quaternions ##

quaternion(A::AbstractArray{<:Quaternion}) = A

function quaternion(A::AbstractArray{T}) where T
    if !isconcretetype(T)
        error("`quaternion` not defined on abstractly-typed arrays; please convert to a more specific type")
    end
    convert(AbstractArray{typeof(quaternion(zero(T)))}, A)
end
