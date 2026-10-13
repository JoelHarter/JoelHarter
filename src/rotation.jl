# Copyright 2026 Joel T Harter
isdefined(@__MODULE__, :Quaternion) || include(joinpath(@__DIR__, "quaternion.jl"))
using StaticArrays
using LinearAlgebra: norm, normalize, cross, I
using Random: Random, AbstractRNG, SamplerType
import Base: *, /, \, inv, one, ==, isapprox, angle

"""
    Rotation

A rotation of 3-space in one of three forms besides a unit `Quaternion`: `Euler` angles about a
sequence of axes, a rotation vector `Rotvec`, or a rotation matrix `Rotmat`. Any form converts to
any other by calling the target type on the source, `Rotmat(q)`, `Rotvec(m)`, `Quaternion(e)`,
`Euler(m, :z, :y, :x)`. Every form is built from principal rotations, one axis and one angle at a
time, `Rotmat(:z, ψ, :y, θ, :x, φ)`, `Quaternion(:x, θ)`. An Euler sequence and a quaternion
convert to a matrix directly, as do a rotation vector and a quaternion to each other and a
rotation vector to a matrix; the rest go through the quaternion. All four forms share these
conventions:

- Rotations are active and right-handed: `rotate(r, v)` turns the vector `v` by `r` in a fixed frame,
  and a positive angle about an axis is counterclockwise looking down the axis. The passive use,
  expressing the same `v` in the rotated frame, is `rotate(inv(r), v)`.
- A quaternion `q` rotates `v` by `q * v * inv(q)`, so `q` and `-q` are the same rotation, and
  composition is the product: `a * b` applies `b` first, then `a`, for every form.
- `inv`, `\\`, `/`, `one` and `isapprox` treat each form as the rotation it stands for: `a ≈ b`
  is true when they rotate the same way, whatever the form or the sign of the quaternion. Two bare
  quaternions keep Base's `≈` as numbers, which tells `q` from `-q`; wrap one, `Rotvec(p) ≈ q`.
- A quaternion's power `q^t` for `0 ≤ t ≤ 1` is a fraction of its rotation, `slerp(a, b, t)`
  interpolates between two rotations, and the angle between two rotations is `angle(Rotvec(a \\ b))`.
- `rand(Rotmat)`, `rand(Rotvec)`, `rand(Euler{(:z, :y, :x)})` draw a rotation uniformly at random; the
  quaternion of one is `sign(randn(QuaternionF64))`.

See also: [`Euler`](@ref), [`Rotvec`](@ref), [`Rotmat`](@ref), [`rotate`](@ref), [`slerp`](@ref).
"""
abstract type Rotation end

"""
    rotate(r, v)

Rotate the vector `v` by the rotation `r`, a `Quaternion` or any `Rotation`. For a quaternion this is
`q * v * inv(q)` with `v` as a pure quaternion, computed in the cross product form that costs half
as much; `q` need not be a unit.

# Examples
```jldoctest
julia> rotate(exp(π/4 * km), SVector(1, 0, 0)) # 90° about z
3-element SVector{3, Float64} with indices SOneTo(3):
 2.220446049250313e-16
 1.0
 0.0

julia> rotate(Rotmat(:z, π/2), SVector(1, 0, 0)) ≈ SVector(0, 1, 0)
true
```
"""
function rotate(q::Quaternion, v::AbstractVector)
    t = 2 * cross(q.v, v)
    v + (q.w * t + cross(q.v, t)) / abs2(q)
end
rotate(r::Rotation, v::AbstractVector) = rotate(Quaternion(r), v)

## Rotation vector ##

"""
    Rotvec(v)
    Rotvec(x, y, z)

A rotation vector: the axis of rotation scaled by the angle in radians, `angle(r) * axis(r)`, the
logarithm of the rotation, `Rotvec(q) == Rotvec(2 * imag(log(q)))` for a unit `q`. The vector is
`r.v`. Converting a rotation to one takes the short way, an angle in `[0, π]`; `Rotvec(v)` itself
accepts any length. A small rotation is nearly its vector, `rotate(r, x) ≈ x + r.v × x`, which is
what makes it the right form for angular velocity times a time step.

# Examples
```jldoctest
julia> r = Rotvec(0, 0, π/2); # 90° about z

julia> angle(r), axis(r)
(1.5707963267948966, [0.0, 0.0, 1.0])

julia> rotate(r, SVector(1, 0, 0)) ≈ SVector(0, 1, 0)
true

julia> Rotvec(exp(π/4 * km)) ≈ r
true
```
"""
struct Rotvec{T<:Real} <: Rotation
    v::SVector{3,T}
    Rotvec{T}(v::SVector{3,T}) where {T<:Real} = new{T}(v)
end
Rotvec(v::SVector{3,T}) where {T<:Real} = Rotvec{T}(v)
Rotvec(v::AbstractVector) = Rotvec(SVector{3}(v))
Rotvec(x::Real, y::Real, z::Real) = Rotvec(SVector(promote(x, y, z)))
Rotvec{T}(v::AbstractVector) where {T<:Real} = Rotvec{T}(SVector{3,T}(v))
Rotvec{T}(x::Real, y::Real, z::Real) where {T<:Real} = Rotvec{T}(SVector{3,T}(x, y, z))
Rotvec(r::Rotvec) = r
Rotvec(q::Quaternion) = Rotvec(2 * imag(log(flipsign(q, q.w)))) # flipsign: the short way round
Rotvec(r::Rotation) = Rotvec(Quaternion(r))
Rotvec{T}(r::Union{Quaternion,Rotation}) where {T<:Real} = Rotvec{T}(Rotvec(r).v)

Quaternion(r::Rotvec) = exp(Quaternion(r.v / 2))

"""
    angle(r::Rotvec)
    axis(r::Rotvec)

The angle in radians and the unit axis of a rotation vector. For any other form of rotation, convert
first: `angle(Rotvec(q))`.
"""
angle(r::Rotvec) = norm(r.v)
axis(r::Rotvec) = normalize(r.v)

inv(r::Rotvec) = Rotvec(-r.v)
one(::Type{Rotvec}) = one(Rotvec{Float64})
one(::Type{Rotvec{T}}) where {T} = Rotvec{T}(zero(SVector{3,T}))
==(a::Rotvec, b::Rotvec) = a.v == b.v
*(a::Rotvec, b::Rotvec) = Rotvec(Quaternion(a) * Quaternion(b))

## Rotation matrix ##

"""
    Rotmat(m::AbstractMatrix)

A 3×3 rotation matrix, `rotate(r, v) == r.m * v`. `Rotmat(m)` takes `m` as given and does not check
that it is orthonormal with determinant 1; a matrix that only nearly is, from accumulated rounding,
can be cleaned by a round trip through a quaternion, `Rotmat(Quaternion(m))`. The conversion from a
matrix is Shepperd's method: whichever of `w`, `x`, `y`, `z` is largest is taken from the diagonal,
so no small number is divided by.

# Examples
```jldoctest
julia> m = Rotmat(exp(π/4 * km)); m.m # 90° about z
3×3 SMatrix{3, 3, Float64, 9} with indices SOneTo(3)×SOneTo(3):
 2.22045e-16  -1.0          0.0
 1.0           2.22045e-16  0.0
 0.0           0.0          1.0

julia> Quaternion(m) ≈ exp(π/4 * km)
true

julia> inv(m).m == m.m'
true
```
"""
struct Rotmat{T<:Real} <: Rotation
    m::SMatrix{3,3,T,9}
    Rotmat{T}(m::SMatrix{3,3,T,9}) where {T<:Real} = new{T}(m)
end
Rotmat(m::SMatrix{3,3,T,9}) where {T<:Real} = Rotmat{T}(m)
Rotmat(m::AbstractMatrix) = Rotmat(SMatrix{3,3}(m))
Rotmat{T}(m::AbstractMatrix) where {T<:Real} = Rotmat{T}(SMatrix{3,3,T}(m))
Rotmat(r::Rotmat) = r
Rotmat(r::Rotation) = Rotmat(Quaternion(r))
# Rodrigues: R = I + a[v]ₓ + b[v]ₓ² with a = sin θ/θ and b = (1 - cos θ)/θ², θ = |v|, which are 1 and
# 1/2 at θ = 0 and are read off their limits below the resolution of the angle, where their
# quotients lose digits. The way back, a matrix to its vector, goes through the quaternion:
# the angle from the trace loses its accuracy near 0 and π, which the quaternion's log keeps.
function Rotmat(r::Rotvec)
    v = r.v
    θ = norm(v)
    a, b = θ < sqrt(eps(θ)) ? (one(θ), one(θ) / 2) : (sin(θ) / θ, 2 * sin(θ / 2)^2 / θ^2)
    x, y, z = v
    K = @SMatrix [0 -z y; z 0 -x; -y x 0]
    Rotmat(SMatrix{3,3}(I) + a * K + b * (K * K))
end
Rotmat{T}(r::Union{Quaternion,Rotation}) where {T<:Real} = Rotmat{T}(Rotmat(r).m)

function Rotmat(q::Quaternion)
    w, x, y, z = q.w, q.x, q.y, q.z
    s = 2 / abs2(q)
    Rotmat(@SMatrix [1 - s*(y*y + z*z)  s*(x*y - w*z)      s*(x*z + w*y)
                     s*(x*y + w*z)      1 - s*(x*x + z*z)  s*(y*z - w*x)
                     s*(x*z - w*y)      s*(y*z + w*x)      1 - s*(x*x + y*y)])
end

function Quaternion(r::Rotmat)
    m = r.m
    t = m[1,1] + m[2,2] + m[3,3]
    # 4w² = 1 + t and 4x² = 1 + 2m₁₁ - t and so on; take the square root of the largest
    q = if t >= m[1,1] && t >= m[2,2] && t >= m[3,3]
        w = sqrt(1 + t) / 2
        Quaternion(w, (m[3,2] - m[2,3]) / 4w, (m[1,3] - m[3,1]) / 4w, (m[2,1] - m[1,2]) / 4w)
    elseif m[1,1] >= m[2,2] && m[1,1] >= m[3,3]
        x = sqrt(1 + 2m[1,1] - t) / 2
        Quaternion((m[3,2] - m[2,3]) / 4x, x, (m[1,2] + m[2,1]) / 4x, (m[1,3] + m[3,1]) / 4x)
    elseif m[2,2] >= m[3,3]
        y = sqrt(1 + 2m[2,2] - t) / 2
        Quaternion((m[1,3] - m[3,1]) / 4y, (m[1,2] + m[2,1]) / 4y, y, (m[2,3] + m[3,2]) / 4y)
    else
        z = sqrt(1 + 2m[3,3] - t) / 2
        Quaternion((m[2,1] - m[1,2]) / 4z, (m[1,3] + m[3,1]) / 4z, (m[2,3] + m[3,2]) / 4z, z)
    end
    flipsign(q, q.w)
end

rotate(r::Rotmat, v::AbstractVector) = r.m * v
inv(r::Rotmat) = Rotmat(r.m')
one(::Type{Rotmat}) = one(Rotmat{Float64})
one(::Type{Rotmat{T}}) where {T} = Rotmat{T}(SMatrix{3,3,T}(I))
==(a::Rotmat, b::Rotmat) = a.m == b.m
*(a::Rotmat, b::Rotmat) = Rotmat(a.m * b.m)

## Principal rotations ##

"""
    Rotmat(axis, θ)
    Quaternion(axis, θ)

A rotation by the angle `θ` about one of the three axes, named `:x`, `:y` or `:z`: the principal
rotation, which every other rotation is built from. As a matrix it is one `sincos` and seven
constants; as a quaternion, `sincos(θ/2)`.

    Rotmat(axis₁, θ₁, axis₂, θ₂, …)
    Quaternion(axis₁, θ₁, axis₂, θ₂, …)

A sequence of principal rotations, as many as given: the product `Rotmat(axis₁, θ₁) *
Rotmat(axis₂, θ₂) * …`, which is the first rotation followed by each next one about its axis as
it then lies (intrinsic), or equally the last followed by each earlier one about the fixed axes
(extrinsic). `Rotmat(:z, ψ, :y, θ, :x, φ)` is the aerospace yaw, pitch and roll; two rotations
take one frame to another, `Rotmat(:z, λ, :y, -φ)`; five carry a frame onto an axis, turn about
it and carry it back. An axis may follow itself, `Rotmat(:z, a, :z, b)` being `Rotmat(:z, a + b)`.
A sequence kept as its angles is an [`Euler`](@ref).

# Examples
```jldoctest
julia> rotate(Rotmat(:z, π/2), SVector(1, 0, 0)) ≈ SVector(0, 1, 0)
true

julia> Rotmat(:z, 0.3, :y, -0.2, :x, 0.1) ≈ Quaternion(:z, 0.3) * Quaternion(:y, -0.2) * Quaternion(:x, 0.1)
true

julia> Rotmat(:z, 0.3, :y, -0.2, :x, 0.1) ≈ Rotmat(:z, 0.3) * Rotmat(:y, -0.2) * Rotmat(:x, 0.1)
true
```
"""
axisindex(axis::Symbol) = axis === :x ? 1 : axis === :y ? 2 : axis === :z ? 3 :
    throw(ArgumentError("an axis is :x, :y or :z, not $(repr(axis))"))

function Rotmat(axis::Symbol, θ::Real)
    s, c = sincos(θ)
    i = axisindex(axis)
    i == 1 ? Rotmat(@SMatrix [1 0 0; 0 c -s; 0 s c]) :
    i == 2 ? Rotmat(@SMatrix [c 0 s; 0 1 0; -s 0 c]) :
             Rotmat(@SMatrix [c -s 0; s c 0; 0 0 1])
end
function Quaternion(axis::Symbol, θ::Real)
    s, c = sincos(θ / 2)
    Quaternion(c, setindex(zero(SVector{3,typeof(s)}), s, axisindex(axis)))
end
Rotmat(axis::Symbol, θ::Real, next::Symbol, rest...) = Rotmat(axis, θ) * Rotmat(next, rest...)
Quaternion(axis::Symbol, θ::Real, next::Symbol, rest...) = Quaternion(axis, θ) * Quaternion(next, rest...)
Rotmat(axis::Symbol) = throw(ArgumentError("each axis takes its angle after it, `Rotmat(:$axis, θ)`"))
Quaternion(axis::Symbol) = throw(ArgumentError("each axis takes its angle after it, `Quaternion(:$axis, θ)`"))

## Euler angles ##

"""
    Euler(axis₁, θ₁, axis₂, θ₂, …)
    Euler(r, axis₁, axis₂, axis₃)

A rotation kept as the angles of a sequence of principal rotations, one axis and one angle at a
time, as many as given, each axis `:x`, `:y` or `:z`: `Euler(:z, ψ, :y, θ, :x, φ)` is the
aerospace yaw, pitch and roll, `Euler(:z, α, :x, β, :z, γ)` a proper Euler sequence, and two or five
rotations are as welcome as three. The rotation is `Rotmat(axis₁, θ₁, axis₂, θ₂, …)`, the product of
the principals, with each rotation about its axis as it then lies. The axes are the type's,
`Euler{(:z, :y, :x)}`, so an `Euler` holds nothing but its angles, `e.angle`, as plain numbers in
a row; the angles are not a vector and are not held as one.

`Euler(r, axis₁, axis₂, axis₃)` reads the three angles about those axes out of any rotation `r`,
for any three axes with no axis twice in a row. `θ₁` and `θ₃` are in `(-π, π]`, and `θ₂` in
`[0, π]` when the first and third axes are the same (proper Euler, `:z, :x, :z`) or in
`[-π/2, π/2]` when all three differ (Tait-Bryan, `:z, :y, :x`). At gimbal lock, `θ₂` at either
end of its range, only `θ₁ ± θ₃` is defined and `θ₃` is set to zero. The extraction is direct for
every sequence, after Bernardes and Viollet (2022), with `θ₂` from an `atan` rather than an `acos`
so it keeps its accuracy near the ends of the range; a matrix goes through its quaternion first.

`inv(e)` is exact: the negated angles in the reversed sequence.

# Examples
```jldoctest
julia> e = Euler(:z, 0.3, :y, -0.2, :x, 0.1)
Euler(:z, 0.3, :y, -0.2, :x, 0.1)

julia> Rotmat(e) ≈ Rotmat(:z, 0.3) * Rotmat(:y, -0.2) * Rotmat(:x, 0.1)
true

julia> Euler(Quaternion(e), :z, :y, :x) ≈ e
true

julia> Euler(e, :x, :y, :z) # the same rotation about other axes
Euler(:x, 0.15641951308019914, :y, -0.1600272204316182, :z, 0.32260969057647504)

julia> inv(e)
Euler(:x, -0.1, :y, 0.2, :z, -0.3)
```
"""
struct Euler{A,T<:Real,N} <: Rotation
    angle::NTuple{N,T}
    function Euler{A,T,N}(angle::NTuple{N,T}) where {A,T<:Real,N}
        A isa NTuple{N,Symbol} || throw(ArgumentError("the axes of an Euler are a tuple of $N symbols, one per angle, not $(repr(A))"))
        foreach(axisindex, A)
        new{A,T,N}(angle)
    end
end
Euler{A}(angle::NTuple{N,T}) where {A,T<:Real,N} = Euler{A,T,N}(angle)
Euler{A}(angle::Real...) where {A} = Euler{A}(promote(angle...))
Euler{A,T}(angle::Real...) where {A,T<:Real} = Euler{A}(map(T, angle))
function Euler(axis::Symbol, θ::Real, rest...)
    A, θs = unzip(axis, θ, rest...)
    return Euler{A}(θs...)
end
Euler(axis::Symbol) = throw(ArgumentError("each axis takes its angle after it, `Euler(:$axis, θ)`"))
# `(:z, ψ, :y, θ)` as the axes and the angles.
unzip(axis::Symbol, θ::Real) = ((axis,), (θ,))
function unzip(axis::Symbol, θ::Real, next::Symbol, rest...)
    A, θs = unzip(next, rest...)
    return (axis, A...), (θ, θs...)
end
unzip(axis::Symbol) = throw(ArgumentError("each axis takes its angle after it, `Euler(:$axis, θ)`"))

Base.show(io::IO, e::Euler{A}) where {A} = print(io, "Euler(", join(("$(repr(a)), $(repr(θ))" for (a, θ) in zip(A, e.angle)), ", "), ")")

Euler{A}(e::Euler{A}) where {A} = e
Euler{A,T}(e::Euler{A}) where {A,T} = Euler{A,T}(e.angle...)
Euler{A}(r::Union{Quaternion,Rotation}) where {A} = Euler(r, A...)
Euler{A,T}(r::Union{Quaternion,Rotation}) where {A,T} = Euler{A,T}(Euler{A}(r).angle...)
Euler{A,T,N}(r::Union{Quaternion,Rotation}) where {A,T,N} = Euler{A,T}(r)      # `typeof(e)(q)`, as `slerp` and `rand` build one

# The products of the principals: the matrix straight from its sines and cosines, no quaternion.
function Rotmat(e::Euler{A}) where {A}
    m = Rotmat(A[1], e.angle[1])
    for k in 2:length(A)
        m = m * Rotmat(A[k], e.angle[k])
    end
    return m
end
function Quaternion(e::Euler{A}) where {A}
    q = Quaternion(A[1], e.angle[1])
    for k in 2:length(A)
        q = q * Quaternion(A[k], e.angle[k])
    end
    return q
end

function Euler(r::Union{Quaternion,Rotation}, axis₁::Symbol, axis₂::Symbol, axis₃::Symbol)
    i, j, k = axisindex(axis₁), axisindex(axis₂), axisindex(axis₃)
    i != j && j != k || throw(ArgumentError("two rotations in a row about one axis have only their sum of angles defined; the angles of a rotation need three axes with none twice in a row, not $(repr((axis₁, axis₂, axis₃)))"))
    q = Quaternion(r)
    l = 6 - i - j                          # the axis that is neither i nor j
    ε = mod(j - i, 3) == 1 ? 1 : -1        # +1 when (i, j, l) is a cyclic permutation
    w, x = q.w, q.v
    # A proper sequence i-j-i, with θ± = (θ₁ ± θ₃)/2, has q = c₂cosθ₊ + c₂sinθ₊ eᵢ + s₂cosθ₋ eⱼ + εs₂sinθ₋ eₗ.
    # A Tait-Bryan i-j-l becomes the proper i-j-i of angles (θ₁, θ₂ + π/2, -εθ₃) times a fixed
    # quarter turn: q * qⱼ(π/2) ∝ q * (1 + eⱼ), so read the same four numbers off that product.
    if k == i
        a, b, c, d = w, x[i], x[j], ε*x[l]
    else
        a, b, c, d = w - x[j], x[i] - ε*x[l], w + x[j], x[i] + ε*x[l]
    end
    ab, cd = hypot(a, b), hypot(c, d)
    θ₂ = 2 * atan(cd, ab)                  # cos(θ₂/2) ∝ hypot(a, b), sin(θ₂/2) ∝ hypot(c, d)
    if iszero(cd)                          # θ₂ = 0: only θ₁ + θ₃ is defined
        θ₁, θ₃ = 2 * atan(b, a), zero(θ₂)
    elseif iszero(ab)                      # θ₂ = π: only θ₁ - θ₃ is defined
        θ₁, θ₃ = 2 * atan(d, c), zero(θ₂)
    else
        θ₊, θ₋ = atan(b, a), atan(d, c)
        θ₁, θ₃ = θ₊ + θ₋, θ₊ - θ₋
    end
    k == i || (θ₂ -= π/2; θ₃ = -ε*θ₃)
    return Euler{(axis₁, axis₂, axis₃)}(rem2pi(θ₁, RoundNearest), θ₂, rem2pi(θ₃, RoundNearest))
end

inv(e::Euler{A}) where {A} = Euler{reverse(A)}(map(-, reverse(e.angle))...)
one(::Type{Euler{A}}) where {A} = one(Euler{A,Float64})
one(::Type{Euler{A,T}}) where {A,T} = Euler{A,T}(ntuple(_ -> zero(T), length(A))...)
==(a::Euler{A}, b::Euler{A}) where {A} = a.angle == b.angle

## Shared by all forms ##

Quaternion{T}(r::Rotation) where {T<:Real} = Quaternion{T}(Quaternion(r))

one(r::Rotation) = one(typeof(r))
# Any two forms compose, in the quaternion, the hub, when they differ.
*(a::Rotation, b::Rotation) = Quaternion(a) * Quaternion(b)
*(q::Quaternion, r::Rotation) = q * Quaternion(r)
*(r::Rotation, q::Quaternion) = Quaternion(r) * q
/(a::Union{Quaternion,Rotation}, b::Rotation) = a * inv(b)
/(a::Rotation, b::Quaternion) = a * inv(b)
\(a::Rotation, b::Union{Quaternion,Rotation}) = inv(a) * b
\(a::Quaternion, b::Rotation) = inv(a) * b

# The same rotation, whatever the form and the sign of the quaternion. Two bare quaternions keep
# Base's ≈ as numbers, which tells q from -q.
function same(a, b; kwargs...)
    p, q = sign(Quaternion(a)), sign(Quaternion(b))
    isapprox(p, q; kwargs...) || isapprox(p, -q; kwargs...)
end
isapprox(a::Rotation, b::Rotation; kwargs...) = same(a, b; kwargs...)
isapprox(a::Rotation, b::Quaternion; kwargs...) = same(a, b; kwargs...)
isapprox(a::Quaternion, b::Rotation; kwargs...) = same(a, b; kwargs...)

"""
    slerp(a, b, t)

Spherical linear interpolation from rotation `a` at `t = 0` to `b` at `t = 1` along the short way
round, at a constant rate: `a * (a \\ b)^t`. Any form of rotation; the result takes the form of `a`.

# Examples
```jldoctest
julia> slerp(one(Rotvec), Rotvec(0, 0, 1), 0.25)
Rotvec{Float64}([0.0, 0.0, 0.25])
```
"""
function slerp(a::Quaternion, b::Quaternion, t::Real)
    p = a \ b
    a * flipsign(p, p.w)^t
end
slerp(a::Rotation, b::Union{Quaternion,Rotation}, t::Real) = typeof(a)(slerp(Quaternion(a), Quaternion(b), t))
slerp(a::Quaternion, b::Rotation, t::Real) = slerp(a, Quaternion(b), t)

# A uniformly random rotation: the direction of a 4-vector of normals is uniform on the unit sphere.
Random.rand(r::AbstractRNG, ::SamplerType{R}) where {R<:Rotation} = R(sign(randn(r, QuaternionF64)))
