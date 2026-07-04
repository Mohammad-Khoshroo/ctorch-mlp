#ifndef LAYER_HPP
#define LAYER_HPP

#include "neuron.hpp"

class LayerBase {
public:
    virtual ~LayerBase() {}

    virtual int get_neurons_count() = 0;
    virtual int get_input_size() = 0;
    virtual ActivationType get_activation() = 0;

    virtual vector<float> get_outputs() = 0;
    virtual vector<float> get_inputs() = 0;
    virtual vector<vector<float>> get_weights_matrix() = 0;
    virtual void update_weights(const vector<float>& deltas, float learning_rate) = 0;
    virtual float activate_derivative_at(int neuron_idx) = 0;
    virtual void randomize_weights(float min_val = -1.0f, float max_val = 1.0f) = 0;

    virtual sc_out<float>& get_output_port(int neuron_idx) = 0;
    virtual sc_in<float>&  get_input_port(int input_idx) = 0; 
};

template<int input_size, int neurons_number, int bias, ActivationType func>
class Layer : public sc_module, public LayerBase {

public:
    sc_in<sc_logic> train_signal;
    
    sc_vector<sc_in<float>> inputs; 
    vector<Neuron<input_size, bias, func>*> neurons;

    SC_HAS_PROCESS(Layer);
    
    Layer(sc_module_name name)
        : sc_module(name),
          train_signal("train_signal"),
          inputs("inputs", input_size),
          neurons(neurons_number) {

        for (int i = 0; i < neurons_number; ++i) {
            string neuron_name = string(name) + "_neuron_" + to_string(i);
            neurons[i] = new Neuron<input_size, bias, func>(neuron_name.c_str(), inputs);
        }
    }

    virtual ~Layer() {
        for (auto* neuron : neurons) delete neuron;
    }

    virtual void train_weights() {}

    int get_neurons_count() override { return neurons_number; }
    int get_input_size() override { return input_size; }
    ActivationType get_activation() override { return func; }

    vector<float> get_outputs() override {
        vector<float> out(neurons_number);
        for (int i = 0; i < neurons_number; i++)
            out[i] = neurons[i]->read();
        return out;
    }

    vector<float> get_inputs() override {
        vector<float> in(input_size);
        for (int i = 0; i < input_size; i++)
            in[i] = inputs[i].read();
        return in;
    }

    vector<vector<float>> get_weights_matrix() override {
        vector<vector<float>> w(neurons_number, vector<float>(input_size));
        for (int i = 0; i < neurons_number; i++)
            for (int j = 0; j < input_size; j++)
                w[i][j] = neurons[i]->weights[j];
        return w;
    }

    float activate_derivative_at(int neuron_idx) override {
        return Neuron<input_size, bias, func>::activate_derivative(
            neurons[neuron_idx]->last_product
        );
    }

    void update_weights(const vector<float>& deltas, float learning_rate) override {
        vector<float> in = get_inputs();
        for (int i = 0; i < neurons_number; i++) {
            for (int j = 0; j < input_size; j++) {
                neurons[i]->weights[j] += learning_rate * deltas[i] * in[j];
            }
        }
    }

    void randomize_weights(float min_val = -1.0f, float max_val = 1.0f) override {
        for (int i = 0; i < neurons_number; i++)
            neurons[i]->randomize_weights(min_val, max_val);
    }

    sc_out<float>& get_output_port(int neuron_idx) override {
        return neurons[neuron_idx]->axon;
    }

    sc_in<float>& get_input_port(int input_idx) override {
        return inputs[input_idx];
    }

    vector<float> compute_hidden_deltas(
        const vector<float>& next_deltas,
        const vector<vector<float>>& next_weights,
        int next_neurons_count
    ) {
        vector<float> deltas(neurons_number, 0.0f);
        for (int i = 0; i < neurons_number; i++) {
            float error_sum = 0.0f;
            for (int j = 0; j < next_neurons_count; j++) {
                error_sum += next_weights[j][i] * next_deltas[j];
            }
            deltas[i] = error_sum * activate_derivative_at(i);
        }
        return deltas;
    }
};

template<int input_size, int neurons_number, int bias, ActivationType func>
class HiddenLayer : public Layer<input_size, neurons_number, bias, func> {
public:
    using Layer<input_size, neurons_number, bias, func>::Layer;
    void train_weights() override {}
};

template<int input_size, int neurons_number, int bias, ActivationType func>
using InputLayer = HiddenLayer<input_size, neurons_number, bias, func>;

template<int input_size, int classes_number, int bias, ActivationType func>
class OutputLayer : public Layer<input_size, classes_number, bias, func> {

public:
    sc_in<unsigned int> label;

    SC_HAS_PROCESS(OutputLayer);
    OutputLayer(sc_module_name name, sc_in<unsigned int>& label_in)
        : Layer<input_size, classes_number, bias, func>(name),
          label("label") {
        label.bind(label_in);
    }

    void train_weights() override {}

    vector<float> compute_output_deltas(unsigned int true_label) {
        vector<float> deltas(classes_number);
        vector<float> outputs = this->get_outputs();

        for (int i = 0; i < classes_number; i++) {
            float target = (i == (int)true_label) ? 1.0f : 0.0f;

            if (func == SOFTMAX) {
                deltas[i] = outputs[i] - target;
            } else {
                deltas[i] = (target - outputs[i]) * this->activate_derivative_at(i);
            }
        }
        return deltas;
    }

    int predict_class() {
        vector<float> outputs = this->get_outputs();
        int max_idx = 0;
        float max_val = outputs[0];
        for (int i = 1; i < classes_number; i++) {
            if (outputs[i] > max_val) {
                max_val = outputs[i];
                max_idx = i;
            }
        }
        return max_idx;
    }
};

#endif // LAYER_HPP
