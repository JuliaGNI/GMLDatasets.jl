# The two autoencoders the pendulum experiment trains, and how to load either from its weights.
#
# `symplectic` is the symplectic autoencoder of the paper. `standard` is a `StandardAutoencoder` of
# Dense layers at about the same number of parameters, which is there to compare latent spaces: the
# same data, loss, optimizer and checkpoint selection, and no symplectic structure in the decoder.
#
# The default width and depth of `standard` are chosen so that the two networks have about the same
# number of parameters, 4408 against the symplectic one's 4402: three layers of width 31 in the
# encoder and in the decoder. `parameter_counts` prints both.
#
# A weights file written by `train_sae.jl` records which architecture it holds in its `architecture`
# attribute, and `load_pendulum_network` builds that one. Files written before the attribute existed
# hold the symplectic autoencoder.

using GeometricMachineLearning
import HDF5

const full_dim = 4
const reduced_dim = 2

symplectic_architecture(; upscale = 20) = SymplecticAutoencoder(full_dim, reduced_dim;
    n_encoder_blocks = 2,
    n_decoder_blocks = 2,
    n_encoder_layers = 10,
    n_decoder_layers = 20,
    n_decoder_output_layers = 10,
    sympnet_upscale = upscale)

standard_architecture(; width = 31, layers = 3) = StandardAutoencoder(full_dim, reduced_dim;
    width = width, n_encoder_layers = layers, n_decoder_layers = layers)

function pendulum_architecture(kind::AbstractString; upscale = 20, width = 31, layers = 3)
    kind == "symplectic" ? symplectic_architecture(; upscale) :
    kind == "standard" ? standard_architecture(; width, layers) :
    error("the architecture must be `symplectic` or `standard`, not `$kind`")
end

"The architecture of a weights file, from its attributes."
function file_architecture(path::AbstractString)
    HDF5.h5open(path, "r") do file
        attrs = HDF5.attributes(file)
        has(name) = haskey(attrs, name)
        kind = has("architecture") ? read(attrs["architecture"]) : "symplectic"
        pendulum_architecture(kind;
            upscale = has("sympnet_upscale") ? read(attrs["sympnet_upscale"]) : 20,
            width = has("width") ? read(attrs["width"]) : 31,
            layers = has("layers") ? read(attrs["layers"]) : 3)
    end
end

load_pendulum_network(path) = load(NeuralNetwork, path, file_architecture(path))


function parameter_counts()
    for kind in ("symplectic", "standard")
        nn = NeuralNetwork(pendulum_architecture(kind))
        println(rpad(kind, 11), parameterlength(nn.model), " parameters (encoder ",
            parameterlength(encoder(nn).model), ", decoder ", parameterlength(decoder(nn).model), ")")
    end
end
