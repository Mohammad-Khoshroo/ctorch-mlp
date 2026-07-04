#include "src/MLP.hpp"
#include <iostream>
#include <iomanip>

using namespace std;

int sc_main(int argc, char* argv[]) {

    cout << "\n========================================" << endl;
    cout << " MLP XOR Example in SystemC" << endl;
    cout << "========================================\n" << endl;

    const int INPUT_SIZE  = 2;
    const int HIDDEN_SIZE = 4;
    const int NUM_CLASSES = 2;

    
    sc_signal<float>        in_sig[INPUT_SIZE];
    sc_signal<unsigned int> label_sig;
    sc_signal<sc_logic>     train_sig;

    
    MLP<INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES, 0, SIGMOID, SOFTMAX> mlp("mlp", 0.5f);

    
    for (int i = 0; i < INPUT_SIZE; i++)
        mlp.inputs[i].bind(in_sig[i]);
    mlp.label.bind(label_sig);
    mlp.train_signal.bind(train_sig);

    
    mlp.randomize_weights(-1.0f, 1.0f);
    mlp.print_architecture();

    
    
    
    float X[4][2] = {
        {0.0f, 0.0f},
        {0.0f, 1.0f},
        {1.0f, 0.0f},
        {1.0f, 1.0f}
    };
    unsigned int Y[4] = {0, 1, 1, 0};  

    
    
    
    const int EPOCHS = 2000;
    train_sig.write(SC_LOGIC_0);

    cout << "\nStarting training..." << endl;

    for (int epoch = 0; epoch < EPOCHS; epoch++) {

        float epoch_loss = 0.0f;

        for (int s = 0; s < 4; s++) {
            
            in_sig[0].write(X[s][0]);
            in_sig[1].write(X[s][1]);
            label_sig.write(Y[s]);
            train_sig.write(SC_LOGIC_0);
            sc_start(1, SC_NS);

            
            vector<float> out = mlp.predict();
            int target = Y[s];
            float p = max(out[target], 1e-7f);  
            epoch_loss -= log(p);

            
            train_sig.write(SC_LOGIC_1);
            sc_start(1, SC_NS);
            train_sig.write(SC_LOGIC_0);
            sc_start(1, SC_NS);
        }

        
        if ((epoch + 1) % 200 == 0 || epoch == 0) {
            cout << "Epoch " << setw(4) << (epoch + 1)
                 << " | Avg Loss: " << fixed << setprecision(4) << (epoch_loss / 4.0f)
                 << endl;
        }
    }

    
    
    
    cout << "\n========================================" << endl;
    cout << " Final Evaluation" << endl;
    cout << "========================================" << endl;
    cout << " Input    | Target | Predicted | Output Values" << endl;
    cout << "----------|--------|-----------|----------------------" << endl;

    int correct = 0;
    train_sig.write(SC_LOGIC_0);

    for (int s = 0; s < 4; s++) {
        in_sig[0].write(X[s][0]);
        in_sig[1].write(X[s][1]);
        sc_start(1, SC_NS);

        vector<float> out = mlp.predict();
        int pred = mlp.predict_class();

        cout << " [" << X[s][0] << "," << X[s][1] << "]  |   "
             << Y[s] << "    |     " << pred << "     | ["
             << fixed << setprecision(4) << out[0] << ", "
             << setprecision(4) << out[1] << "]" << endl;

        if (pred == (int)Y[s]) correct++;
    }

    cout << "\nAccuracy: " << correct << "/4 (" << (correct * 100 / 4) << "%)" << endl;
    cout << "========================================\n" << endl;

    return 0;
}