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

# Validate the rendered expressions against the original trained Flux network
# before writing the artifact. Inputs to the original network are
# normalized; the exported equations accept raw age, weight, and BMI.
reference_functions = get_fctsfrommodel(network)
# Compare against the trained Flux network itself, not just a second
# expression produced by the same layer2string reconstruction.
test_inputs = [
    (24.0, 55.0, 18.0, -0.5, -0.5),
    (42.0, 78.0, 25.0, 0.5, 0.5),
    (63.0, 105.0, 31.0, -0.5, 0.5),
    (76.0, 135.0, 39.0, 0.5, -0.5),
    (35.0, 92.0, 29.0, 0.5, 0.5),
]
function equation_function(expression)
    return eval(Meta.parse("(x_age,x_wgt,x_bmi,x_gdr,x_AV) -> " * expression))
end
common_functions = [equation_function(common[name]) for name in ["k10", "k12", "k13", "k21", "k31", "V1"]]
male_functions = Dict(name => equation_function(male[name]) for name in ["k13", "k31"])
female_functions = Dict(name => equation_function(female[name]) for name in ["k13", "k31"])

for (age, weight, bmi, gdr, av) in test_inputs
    normalized = (age / 88, weight / 160, bmi / 52.84713965, gdr, av)
    network_rates = normalization .* get_modeloutput(network, reshape(Float32.(collect(normalized)), 5, 1))
    for i in 1:6
        expected = network_rates[i]
        reconstructed = normalization[i] * reference_functions[i](normalized...)
        isapprox(reconstructed, expected; rtol=1e-5, atol=1e-8) ||
            error("Original network and eqreader disagree at $(normalized) for output $i")
        actual = common_functions[i](age, weight, bmi, gdr, av)
        isapprox(actual, expected; rtol=1e-5, atol=1e-8) ||
            error("Equation export mismatch for $(["k10","k12","k13","k21","k31","V1"][i]) at input $(normalized): $(actual) vs $(expected)")
    end
    for (sex, functions) in ((-0.5, male_functions), (0.5, female_functions))
        sex_input = (age, weight, bmi, sex, av)
        normalized_sex = (age / 88, weight / 160, bmi / 52.84713965, sex, av)
        sex_network_rates = normalization .* get_modeloutput(network, reshape(Float32.(collect(normalized_sex)), 5, 1))
        for name in ["k13", "k31"]
            i = name == "k13" ? 3 : 5
            expected = sex_network_rates[i]
            actual = functions[name](sex_input...)
            isapprox(actual, expected; rtol=1e-8, atol=1e-10) ||
                error("Sex-specific equation export mismatch for $name at input $normalized_sex: $actual vs $expected")
        end
    end
end
println("Validated all six equations and male/female substitutions on $(length(test_inputs)) representative covariate combinations.")

result = Dict(
    "source_repository" => "wahlquisty/learning-pharmacometric-covariate-structures",
    "source_commit" => "939d30699fa6157c14c601712882c51106b82a24",
    "julia_version" => "1.8.4",
    "source_model" => "symbreg/bson/bson_ALE/nn_thread1loss_ALE.bson",
    "source_model_git_blob_sha" => "aec9732b21a2beaec315eaefebb37699546f9dfa",
    "validation" => Dict("method" => "compare reconstructed equations and eqreader against original trained Flux network", "covariate_combinations" => length(test_inputs), "relative_tolerance" => 1e-5, "absolute_tolerance" => 1e-8),
    "input_normalization" => Dict("AGE" => 88.0, "WGT" => 160.0, "BMI" => 52.84713965),
    "input_variables" => ["AGE", "WGT", "BMI", "GDR", "AV"],
    "output_order" => ["k10", "k12", "k13", "k21", "k31", "V1"],
    "output_units" => Dict("k10" => "1/s", "k12" => "1/s", "k13" => "1/s", "k21" => "1/s", "k31" => "1/s", "V1" => "mL"),
    # Store independently evaluated network outputs so the Python consumer
    # can verify numerical agreement, not merely positivity and syntax.
    "reference_predictions" => [
        Dict(
            "covariates" => collect(input),
            "rates" => collect(normalization .* get_modeloutput(network, reshape([
                input[1] / 88, input[2] / 160, input[3] / 52.84713965, input[4], input[5]
            ] .|> Float32, 5, 1))),
        ) for input in test_inputs
    ],
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
