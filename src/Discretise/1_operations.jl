
Base.:+(a::Operator, b::Operator) = [a, b]
Base.:+(a::Vector{<:Operator}, b::Operator) = push!(a, b)

Base.:-(a::Operator, b::Operator) = begin
    @reset b.sign = -1
    [a, b]
end
Base.:-(a::Vector{<:Operator}, b::Operator) = begin
    @reset b.sign = -1
    push!(a, b)
end

# Source operations

Base.:+(a::Src, b::Src) = [a, b]
Base.:+(a::Vector{<:Src}, b::Src) = push!(a, b)

Base.:-(a::Src, b::Src) = begin
    @reset b.sign = -1
    [a, b]
end
Base.:-(a::Vector{<:Src}, b::Src) = begin
    @reset b.sign = -1
    push!(a, b)
end

# Equality operation for model wrapper

Base.:(==)(a::Operator, b::Src) = begin
    Model{1,1}((a,),(b,))
end

Base.:(==)(a::Vector{<:Operator}, b::Src) = begin
    Model{length(a),1}((a...,),(b,))
end

Base.:(==)(a::Operator, b::Vector{<:Src}) = begin
    Model{1,length(b)}((a...,),(b,))
end

Base.:(==)(a::Vector{<:Operator}, b::Vector{<:Src}) = begin
    Model{length(a), length(b)}((a...,),(b...,))
end