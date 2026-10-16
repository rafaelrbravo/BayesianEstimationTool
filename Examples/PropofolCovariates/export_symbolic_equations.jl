#!/usr/bin/env julia
# Export the original trained symbolic-regression equations to TOML.
#
# Usage:
#   julia Examples/PropofolCovariates/export_symbolic_equations.jl /path/to/learning-pharmacometric-covariate-structures
#
# Run with the upstream repository's Julia project. The script reads the committed
# BSON model and reproduces the scaling/substitutions in symbreg/checkresults/get_results.jl.
# It does not train a model or require the patient CSV files.

using Pkg
using TOML

length(ARGS) == 1 || error("Pass the path to a checkout of wahlquisty/learning-pharmacometric-covariate-structures")
upstream = abspath(ARGS[1])
symbreg = joinpath(upstream, "symbreg")
model_path = joinpath(symbreg, "bson", "bson_ALE", "nn_thread1loss_ALE.bson")
isfile(joinpath(symbreg, "Project.toml")) || error("No symbreg/Project.toml found under: $upstream")
isfile(model_path) || error("Trained BSON model not found: $model_path")

Pkg.activate(symbreg)
Pkg.instantiate()

using BSON: @load
using Flux
using SymbolicUtils

@syms x_age x_wgt x_bmi x_gdr x_AV

# Register the custom Julia/Flux model types before deserializing BSON.
cd(symbreg)
include(joinpath("fcts", "fcts_nn.jl"))
include(joinpath("fcts", "fcts_eqreader.jl"))

@load model_path model
network = model.nn
normalization = [0.008468822, 0.013328808, 0.00356759, 0.0057506226, 0.000102905105, 9440.525]

function equations_for_sex(network, normalization, gdr)
    input = ["(x_age/88)", "(x_wgt/160)", "(x_bmi/52.84713965)", string(gdr), "x_AV"]
    chain = network[1].paths
    names = ["k10", "k12", "k13", "k21", "k31", "V1"]
    equations = Dict{String,String}()
    for i in eachindex(chain)
        m = chain[i]
        y1 = layer2string(m, 1, input, false)
        y2 = layer2string(m, 2, y1, false)
        y3 = layer2string(m, 3, y2, false)
        y4 = layer2string(m, 4, y3, false)
        y = layer2string(m, 5, y4, false)[1]
        scaled = string(normalization[i], "*", y)
        equations[names[i]] = string(expand(eval(Meta.parse(scaled))))
    end
    return equations
end

# Match the upstream script: generic GDR for common outputs, and separate
# male/female substitutions for k13 and k31.
common = equations_for_sex(network, normalization, "x_gdr")
male = equations_for_sex(network, normalization, -0.5)
female = equations_for_sex(network, normalization, 0.5)

result = Dict(
    "source_repository" => "wahlquisty/learning-pharmacometric-covariate-structures",
    "source_model" => "symbreg/bson/bson_ALE/nn_thread1loss_ALE.bson",
    "input_normalization" => Dict("AGE" => 88.0, "WGT" => 160.0, "BMI" => 52.84713965),
    "input_variables" => ["AGE", "WGT", "BMI", "GDR", "AV"],
    "output_order" => ["k10", "k12", "k13", "k21", "k31", "V1"],
    "output_units" => Dict("k10" => "1/s", "k12" => "1/s", "k13" => "1/s", "k21" => "1/s", "k31" => "1/s", "V1" => "mL"),
    "equations" => Dict(
        "common" => common,
        "male" => Dict("k13" => male["k13"], "k31" => male["k31"]),
        "female" => Dict("k13" => female["k13"], "k31" => female["k31"]),
    ),
)
output = joinpath(@__DIR__, "symbolic_equations.toml")
open(output, "w") do io
    TOML.print(io, result)
end
println("Exported equations to: ", output)
