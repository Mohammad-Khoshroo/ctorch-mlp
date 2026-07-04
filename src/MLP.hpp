#ifndef MLP_STRUCT
#define MLP_STRUCT


#include <systemc.h>
#include <vector>
#include <memory>
#include "layer.hpp" 
class AbstractMLP : public sc_module {
protected:
    
    std::vector<LayerBase*> layers; 

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

            for (int j = 0; j < outputs_count; ++j) {
                next_layer->get_output_port(j).bind(current_layer->get_output_port(j)); 
            }
        }
    }

    virtual void forward() = 0; 
};

#endif // MLP_STRUCT