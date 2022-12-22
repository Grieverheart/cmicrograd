#include <string.h>

typedef struct
{
    Value  bias;
    Value* weights;
    size_t num_weights;
} Neuron;

void cmg_neuron_create(Neuron* neuron, size_t num_inputs)
{
    neuron->bias = cmg_val(0);
    neuron->num_weights = num_inputs;
    neuron->weights = (Value*) malloc(num_inputs * sizeof(Value));
    for(size_t i = 0; i < num_inputs; ++i)
    {
        // @todo: Random uniform.
        neuron->weights[i] = cmg_val(0.1);
    }
}

void cmg_neuron_free(Neuron* neuron)
{
    free(neuron->weights);
}

typedef struct
{
    Neuron* neurons;
    size_t num_neurons;
} Layer;

void cmg_layer_create(Layer* layer, size_t num_inputs, size_t num_outputs)
{
    layer->num_neurons = num_outputs;
    layer->neurons = (Neuron*) malloc(num_outputs * sizeof(Neuron));
    for(size_t i = 0; i < num_outputs; ++i)
        cmg_neuron_create(&layer->neurons[i], num_inputs);
}

void cmg_layer_free(Layer* layer)
{
    for(size_t ni = 0; ni < layer->num_neurons; ++ni)
        cmg_neuron_free(&layer->neurons[ni]);
    free(layer->neurons);
}

Value* cmg_layer_params(const Layer* layer, size_t* num_params)
{
    size_t _num_params = 0;
    for(size_t ni = 0; ni < layer->num_neurons; ++ni)
        _num_params += layer->neurons[ni].num_weights + 1;
    *num_params = _num_params;

    Value* layer_params = (Value*) malloc(_num_params * sizeof(Value));

    for(size_t ni = 0, i = 0; ni < layer->num_neurons; ++ni)
    {
        for(size_t wi = 0; wi < layer->neurons[ni].num_weights; ++wi, ++i)
            layer_params[i] = layer->neurons[ni].weights[wi];
        layer_params[i++] = layer->neurons[ni].bias;
    }

    return layer_params;
}

Value cmg_neuron_forward(const Neuron* neuron, const Value* x)
{
    Value r = cmg_val(0);
    for(size_t wi = 0; wi < neuron->num_weights; ++wi)
        r = cmg_add(r, cmg_mul(neuron->weights[wi], x[wi]));
    r = cmg_add(r, neuron->bias);
    return cmg_relu(r);
}

Value* cmg_layer_forward(const Layer* layer, const Value* x)
{
    Value *y = (Value*) malloc(layer->num_neurons * sizeof(Value));
    for(size_t ni = 0; ni < layer->num_neurons; ++ni)
        y[ni] = cmg_neuron_forward(&layer->neurons[ni], x);
    return y;
}

typedef struct
{
    Layer* layers;
    size_t* sizes;
    size_t num_layers;
} MLP;

void cmg_mlp_create(MLP* mlp, size_t* sizes, size_t num_sizes)
{
    size_t num_layers = num_sizes - 1;
    mlp->num_layers = num_layers;
    mlp->layers = (Layer*) malloc(num_layers * sizeof(Layer));
    mlp->sizes = (size_t*) malloc(num_sizes * sizeof(size_t));
    memcpy(mlp->sizes, sizes, num_sizes * sizeof(size_t));

    for(size_t li = 0; li < num_layers; ++li)
        cmg_layer_create(&mlp->layers[li], sizes[li], sizes[li+1]);
}

void cmg_mlp_free(MLP* mlp)
{
    free(mlp->sizes);
    for(size_t li = 0; li < mlp->num_layers; ++li)
        cmg_layer_free(&mlp->layers[li]);
    free(mlp->layers);
}

Value* cmg_mlp_forward(const MLP* mlp, const Value* x)
{
    Value* y = cmg_layer_forward(&mlp->layers[0], x);
    for(size_t li = 1; li < mlp->num_layers; ++li)
    {
        Value* r = cmg_layer_forward(&mlp->layers[li], y);
        free(y);
        y = r;
    }
    return y;
}

Value* cmg_mlp_params(const MLP* mlp, size_t* num_params)
{
    Value* params = NULL;
    size_t _num_params = 0;
    for(size_t li = 0; li < mlp->num_layers; ++li)
    {
        size_t np;
        Value* layer_params = cmg_layer_params(&mlp->layers[li], &np);
        params = (Value*) realloc((void*) params, (_num_params + np) * sizeof(Value));
        memcpy(params + _num_params, layer_params, np * sizeof(Value));
        _num_params += np;
        free(layer_params);
    }
    *num_params = _num_params;
    return params;
}

