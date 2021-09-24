fetchvcat(J::judiVector) = J
fetchvcat(x) = vcat(x...)

function JUDI.time_modeling(model::Model, srcGeometry, srcData, recGeometry, recData, dm, srcnum::UnitRange{Int64}, op::Char, mode::Int64, options)

    # Broadcast common parameters
    _model = @bcast model
    _dm = isnothing(dm) ? dm : @bcast dm
    opts = make_opt([model, dm, op, srcGeometry])
    iter = make_parts(srcnum)
    # Run on azure
    results = @batchexec pmap(sx -> time_modeling_azure(_model, subsample(srcGeometry, sx), subsample(srcData, sx),
                                                        subsample(recGeometry, sx), subsample(recData, sx), _dm,
                                                        op, mode, subsample(options, sx)), iter) opts
    # Gather results
    if op=='F' || (op=='J' && mode==1)
        temp = fetch(results)
        argout1 = fetchvcat(temp)
    elseif op=='J' && mode==-1
        argout1 = fetchreduce(results; op=+, remote=true)
    else
        error("operation no defined")
    end
    return argout1
end

# FWI
function JUDI.fwi_objective(model::Model, source::judiVector, dObs::judiVector; options=JUDI.Options())
    # fwi_objective function for multiple sources. The function distributes the sources and the input data amongst the available workers.
    # Broadcast common parameters
    _model = @bcast model
    opts = make_opt([model, dObs, source])
    iter = make_parts(1:dObs.nsrc)
    results = @batchexec pmap(j -> fwi_objective_azure(_model, source[j], dObs[j], subsample(options, j)), iter) opts
    
    # Collect and reduce gradients
    obj, gradient = fetchreduce(results; op=+, remote=true)
    # first value corresponds to function value, the rest to the gradient
    return obj, gradient
end

# lsrtm

function JUDI.lsrtm_objective(model::Model, source::judiVector, dObs::judiVector, dm; options=JUDI.Options(), nlind=false)
    # lsrtm_objective function for multiple sources. The function distributes the sources and the input data amongst the available workers.

    # Broadcast common parameters
    _model = @bcast model
    _dm = isnothing(dm) ? dm : @bcast dm

    opts = make_opt([model, dObs, source, dm])
    iter = make_parts(1:dObs.nsrc)
    results = @batchexec pmap(j -> lsrtm_objective_azure(_model, source[j], dObs[j], _dm, subsample(options, j); nlind=nlind), iter) opts
    # Collect and reduce gradients
    obj, gradient = fetchreduce(results; op=+, remote=true)

    # first value corresponds to function value, the rest to the gradient
    return obj, gradient
end

function JUDI.lsrtm_objective(model::Array{Model,1}, source::Array{judiVector{T,Array{T,2}},1}, dObs::Array{judiVector{T,Array{T,2}},1}, dm::Union{Array{Array{T,1},1}, Array{PhysicalParameter{T},1}}; options=Options(), nlind=false) where T
# lsrtm_objective function for multiple sources and multiple vintages. The function distributes the sources and the input data amongst the available workers.

    println("serial lsrtm_objective on each vintage now")

    ctrl = Array{Any, 1}(undef, length(dObs))

    for i = 1:length(dObs)
        _model = model[i]
        _source = source[i]
        _dObs = dObs[i]
        _dm = dm[i]
        ctrl[i] = @batchexec pmap(j -> lsrtm_objective_azure(_model, _source[j], _dObs[j], _dm, subsample(options, j); nlind=nlind), 1:length(dObs)) opts
    end
    for i = 1:length(dObs)
        results = fetch(ctrl[i])
        obj[i], gradient[j] = results
    end

    obj1 = sum(obj)

    # first value corresponds to function value, the rest to the gradient
    return obj1, gradient
end



# Make options for unique batch ids
make_opt(a::Array) = AzureClusterlessHPC.Options(;task_name="task_$(objectid(a))")

# Batchexec function 
slicec(i::Integer, j::Integer) = i==j ? i : (i:j)
chunk(arr, n) = [arr[slicec(i, min(i + n - 1, length(arr)))] for i in 1:n:length(arr)]
make_parts(iter) = _njpi > 1 ? chunk(iter, _njpi) : iter
