function free_propagation_step!(ψ,phases,plan,iplan)
    plan*ψ
    map!(*,ψ,ψ,phases)
    iplan*ψ
    nothing
end

phase_evolution(ψ,factor) = cis(abs2(ψ)*factor)*ψ

function nls_propagation_step!(ψ,phases,plan,iplan,factor,repetitions)
    for n in 1:repetitions
        map!(ψ->phase_evolution(ψ,factor/2),ψ,ψ)
        free_propagation_step!(ψ,phases,plan,iplan)
        map!(ψ->phase_evolution(ψ,factor/2),ψ,ψ)
    end
end

function partition(a::Integer,b::Integer)
    @assert a ≥ b

    result = fill(a ÷ b, b)

    for n in 1:mod(a,b)
        result[n] +=1
    end

    result
end

function get_free_propagation_phases(xs,ys,Δz,k,use_cuda)
    if use_cuda
        map(ks -> cis(-Δz*sum(abs2,ks)/(2k)), 
        Iterators.product(reciprocal_grid(xs),reciprocal_grid(ys))) |> ifftshift |> cu
    else
        map(ks -> cis(-Δz*sum(abs2,ks)/(2k)), 
        Iterators.product(reciprocal_grid(xs),reciprocal_grid(ys))) |> ifftshift
    end
end

function kerr_propagation(ψ₀,xs,ys,zs,total_steps=0;k=1,χ=1)
    #solves 2ik ∂_z ψ = - ∇² ψ - χ |ψ|² ψ with initial condition ψ₀

    @assert iszero(first(zs))

    if iszero(total_steps) || total_steps < length(zs) - 1
        total_steps = length(zs)
    end

    steps = partition(total_steps,length(zs)-1)

    results = similar(ψ₀,size(ψ₀)...,length(zs))
    results[:,:,1] = ψ₀

    use_cuda = typeof(ψ₀) <: CuArray

    plan = plan_fft!(ψ₀)
    iplan = plan_ifft!(ψ₀)

    cache = ifftshift(ψ₀)
    phases = similar(ψ₀)

    for (i,n) in enumerate(steps)
        Δz = (zs[i+1] - zs[i])/n
        phase_evolution_factor = χ*Δz/(2k)

        if i == 1 || n != steps[i-1]
            phases = get_free_propagation_phases(xs,ys,Δz,k,use_cuda)
        end

        nls_propagation_step!(cache,phases,plan,iplan,phase_evolution_factor,n)

        fftshift!(view(results,:,:,i+1), cache)
    end

    results
end