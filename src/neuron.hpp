#ifndef NEURON_HPP
#define NEURON_HPP

#include <iostream>
#include <systemc.h>
#include <vector>
#include <cmath>
#include <string>
#include <cstdlib>
#include <random>

using namespace std;

enum ActivationType { X, ReLU, SIGMOID, TANH, SOFTMAX, LINEAR };

// input_size == previous_layer_Neurons_Number
template <int input_size, int bias, ActivationType func>
class Neuron : public sc_module {

public:
    
    sc_vector<sc_in<float>>& dendrites;
    sc_out<float> axon;

    vector<float> weights;

    // backpropagation memory for saving forward-pass
    float last_product;     // z = sum(w*x) + bias 
    float last_activation;  // a = f(z)

private:
    void compute();

public:
    SC_HAS_PROCESS(Neuron);
    Neuron(sc_module_name name, sc_vector<sc_in<float>>& inputs)
        : sc_module(name),
          dendrites(inputs),
          axon("axon"),
          weights(input_size, 1.0f),
          last_product(0.0f),
          last_activation(0.0f) {

        SC_METHOD(compute);
        for (int i = 0; i < input_size; ++i)
            sensitive << dendrites[i];
    }

    float read() { return axon.read(); }
    sc_out<float>& get_output() { return axon; }

    void randomize_weights(float min_val = -1.0f, float max_val = 1.0f) {
        static random_device rd;
        static mt19937 gen(rd());
        uniform_real_distribution<float> dist(min_val, max_val);
        for (int i = 0; i < input_size; ++i)
            weights[i] = dist(gen);
    }

    static float activate(float x) {
        switch (func) {
            case ReLU:    return (x > 0) ? x : 0.0f;
            case SIGMOID: return 1.0f / (1.0f + exp(-x));
            case TANH:    return tanh(x);
            case SOFTMAX: return x;
            default:      return x;  // linear
        }
    }

    static float activate_derivative(float x) {
        switch (func) {
            case ReLU:    return (x > 0) ? 1.0f : 0.0f;
            case SIGMOID: {
                float s = 1.0f / (1.0f + exp(-x));
                return s * (1.0f - s);
            }
            case TANH: {
                float t = tanh(x);
                return 1.0f - t * t;
            }
            case SOFTMAX: return 1.0f;
            default:      return 1.0f;
        }
    }
};

template<int input_size, int bias, ActivationType func>
void Neuron<input_size, bias, func>::compute() {

    float product = 0.0;
    for (int i = 0; i < input_size; i++)
        product += weights[i] * dendrites[i].read();
    product += bias;

    last_product = product;
    last_activation = activate(product);

    axon.write(last_activation);
}

#endif // NEURON_HPP