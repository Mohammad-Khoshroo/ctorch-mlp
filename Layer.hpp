#include <Neuron.hpp>

// Super class for all layers
// input_size == previous_layer_Neurons_Number
template<int input_size, int neurons_number, int bias, ActivationType func>
class Layer : public sc_module {
public:
    sc_in<sc_logic> train_signal;
    sc_vector<sc_in<float>>& dendrites_bundles;
    vector<sc_out<float>*> axons;
    virtual void train_weights() = 0;
    
    // for each Neuron in every layer we should connect all previous layer Neurons output to its input
    vector<Neuron<input_size, bias, func>*> neurons;

    SC_HAS_PROCESS(Layer);
    Layer(sc_module_name name, sc_vector<sc_in<float>>& inputs)
        : sc_module(name),
          dendrites_bundles(inputs),
          axons(neurons_number),
          neurons(neurons_number) {      
        
        for (int i = 0; i < neurons_number; ++i) {
            string neuron_name = name+"_neuron_"+to_string(i);
            neurons[i] = new Neuron<input_size,bias,func>(neuron_name, inputs);
            axons[i] = &neurons[i]->get_output();
        }

        SC_METHOD(train_weights);
        sensitive << train_signal.pos();
        
    }

    ~Layer() { for (auto* neuron : neurons) delete neuron; }

    // Training Function 
    // for now i don't know what should i do with this :/
};








template<int input_size, int neurons_number, int bias, ActivationType func>
class HiddenLayer : public Layer<input_size, neurons_number, bias, func> {
private:
    void train_weights();
};









template<int input_size, int neurons_number, int bias, ActivationType func>
using InputLayer = HiddenLayer<input_size, neurons_number, bias, func>;













template<int input_size, int classes_number, int bias, ActivationType func>
class OutputLayer : public Layer<input_size, classes_number, bias, func> {
private:
    sc_in<unsigned int>& label;
    void train_weights();

public:

    OutputLayer(sc_module_name name, sc_vector<sc_in<float>>& inputs, sc_in<unsigned int>& label) : 
    Layer<input_size, classes_number, bias, func>(name, inputs),
    label(label){};

};

template<int input_size, int classes_number, int bias, ActivationType func>
void OutputLayer<input_size, classes_number, bias, func>::train_weights() {

  if (train_signal.read() == SC_LOGIC_1) {
        
        vector<float> neuron_outputs(classes_number);

        float max_output = -std::numeric_limits<float>::infinity();
        int predicted_class = 0;
        
        for (int i = 0; i < classes_number; i++) {
            neuron_outputs[i] = neurons[i]->read();
            if (neuron_outputs[i] > max_output) {
                max_output = neuron_outputs[i];
                predicted_class = i;
            }
        }

        // Update weights for each neuron
        for (int i = 0; i < classes_number; i++) {
            float target = (i == predicted_class) ? 1.0f : 0.0f;
            float error = target - neuron_outputs[i];
            
            // Update weights with multi-class learning rule
            for (int j = 0; j < input_size; j++) {
                float input_value = this->neurons[i]->dendrites[j].read();
                float learning_rate = 0.01f; // You can adjust this value
                
                // Update weight: w = w + η * e * x
                this->neurons[i]->weights[j] += learning_rate * error * input_value;
            }
        }
    }  
    
} 