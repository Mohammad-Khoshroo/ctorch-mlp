#include "neuron.hpp"

SC_MODULE(TestBench) {
    
    sc_signal<float> in_signals[3];
    sc_signal<float> out_signal;

    Neuron<3, 1, ReLU> *neuron;

    SC_CTOR(TestBench) {
        
        neuron = new Neuron<3, 1, ReLU>("neuron");

        for (int i = 0; i < 3; ++i)
            neuron->inputs[i](in_signals[i]);

        neuron->output(out_signal);

        SC_THREAD(run);
    }

    void run() {

        neuron->weights[0] = 0.5;
        neuron->weights[1] = -0.3;
        neuron->weights[2] = 0.8;

        in_signals[0].write(0.0);
        in_signals[1].write(0.0);
        in_signals[2].write(0.0);

        wait(SC_ZERO_TIME);
        wait(SC_ZERO_TIME);

        cout << "Time: " << sc_time_stamp() << "Output: " << out_signal.read() << endl;

        wait(4 , SC_NS);

        in_signals[0].write(1.0);
        in_signals[1].write(0.0);
        in_signals[2].write(1.0);

        wait(SC_ZERO_TIME);
        wait(SC_ZERO_TIME);

        cout << "Time: " << sc_time_stamp() << "Output: " << out_signal.read() << endl;

        wait(50, SC_NS);

        in_signals[0].write(4.0);

        wait(SC_ZERO_TIME);
        wait(SC_ZERO_TIME);

        cout <<"Time: " << sc_time_stamp() << "Output: " << out_signal.read() << endl;
        

        sc_stop();
    }
};

int sc_main(int argc, char* argv[]) {
    
    TestBench tb("tb");
    sc_start();
    
    return 0;
}
