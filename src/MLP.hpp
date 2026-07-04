#ifndef MLP_STRUCT
#define MLP_STRUCT

#include <systemc.h>
#include <vector>
#include <memory>
#include <cassert>
#include "layer.hpp"

class AbstractMLP : public sc_module {
protected:
    std::vector<LayerBase*> layers;
    std::vector<sc_vector<sc_signal<float>>*> inter_layer_signals;

public:
    SC_HAS_PROCESS(AbstractMLP);
    AbstractMLP(sc_module_name name) : sc_module(name) {}

    void build_and_connect() {
        for (size_t i = 0; i < layers.size() - 1; ++i) {
            auto* current_layer = layers[i];
            auto* next_layer = layers[i + 1];

            int outputs_count = current_layer->get_neurons_count();
            int next_inputs_count = next_layer->get_input_size();

            assert(outputs_count == next_inputs_count && "Layer size mismatch!");

            std::string sig_name = "sig_" + std::to_string(i) + "_" + std::to_string(i+1);
            auto* sigs = new sc_vector<sc_signal<float>>(sig_name.c_str(), outputs_count);
            inter_layer_signals.push_back(sigs);

            for (int j = 0; j < outputs_count; ++j) {
                current_layer->get_output_port(j).bind((*sigs)[j]);
                next_layer->get_input_port(j).bind((*sigs)[j]);
            }
        }
    }

    virtual void forward() {}

    virtual ~AbstractMLP() {
        for (auto* s : inter_layer_signals) delete s;
    }
};

#endif // MLP_STRUCT