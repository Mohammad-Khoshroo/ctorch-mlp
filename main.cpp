#include  "src/MLP.hpp"

class MyMLP : public AbstractMLP {
public:
    Layer<8, 16, 1, ReLU>*        hidden_layer1;
    Layer<16, 16, 1, ReLU>*       hidden_layer2;
    OutputLayer<16, 3, 1, SOFTMAX>* output_layer;

    sc_vector<sc_in<float>> mlp_inputs;
    sc_in<unsigned int> label;

    MyMLP(sc_module_name name, sc_in<unsigned int>& label_in)
        : AbstractMLP(name),
          mlp_inputs("mlp_inputs", 8),
          label("label") {

        label.bind(label_in);

        hidden_layer1 = new Layer<8, 16, 1, ReLU>("hidden1");
        hidden_layer2 = new Layer<16, 16, 1, ReLU>("hidden2");
        output_layer  = new OutputLayer<16, 3, 1, SOFTMAX>("output", label);

        layers.push_back(hidden_layer1);
        layers.push_back(hidden_layer2);
        layers.push_back(output_layer);

        for (int i = 0; i < 8; ++i) {
            hidden_layer1->get_input_port(i).bind(mlp_inputs[i]);
        }

        build_and_connect();
    }

    ~MyMLP() {
        delete hidden_layer1;
        delete hidden_layer2;
        delete output_layer;
    }
};