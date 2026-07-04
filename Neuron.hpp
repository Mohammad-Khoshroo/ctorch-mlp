#include <iostream>
#include <systemc.h>
#include <vector>
#include <cmath>
#include <string>

using namespace std;
enum ActivationType { X, ReLU, SIGMOID, TANH, SOFTMAX };

// input_size == previous_layer_Neurons_Number
template <int input_size, int bias, ActivationType func>
class Neuron : public sc_module {
    
private:
    
    vector<sc_in<float>>& dendrites;
    sc_out<float> axon;
    void compute();
    
public:
    vector<float> weights;

    SC_HAS_PROCESS(Neuron);
    Neuron( sc_module_name name, sc_vector<sc_in<float>>& inputs)
    :sc_module(name),
     dendrites(inputs),
     axon("axon"),
     weights(input_size,1.0f) {

        SC_METHOD(compute);
        for (int i = 0; i < input_size; ++i)
            sensitive << dendrites[i];

    }

    float read() {return axon.read(); }
    sc_out<float>& get_output() { return axon; }

};

template<int input_size, int bias, ActivationType func>
void Neuron<input_size, bias, func>::compute() {
    
    // Activation Funcs
    auto reLU = [](float x) { return (x > 0) ? x : 0; };
    auto sigmoid = [](float x) { return 1.0 / (1.0 + exp(-x)); };
    auto tanh_fn = [](float x) { return tanh(x); };

    float product = 0.0;
    for (int i = 0; i < input_size; i++)
        product += weights[i] * dendrites[i].read();
    product += bias;

    float activated_product = 0.0;
    
    switch (func) {
        case ReLU :    activated_product = reLU(product); break;
        case SIGMOID: activated_product = sigmoid(product); break;
        case TANH:    activated_product = tanh_fn(product); break;
        default:      activated_product = product;
    }

    axon.write(activated_product);

}