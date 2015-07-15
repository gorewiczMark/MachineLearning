package Classifiers.NNC;

import java.util.ArrayList;
import java.util.List;

public class Layer {
    List<Neuron> neurons = new ArrayList<>();

    public Layer(int neuronCount, int inputCount) {
        for (int i = 0; i < neuronCount; i++) {
            neurons.add(new Neuron(inputCount));
        }
    }

    public List<Double> produceOutputs(List<Double> inputs) {
        List<Double> outputs = new ArrayList<>();
        for (Neuron neuron : neurons) {
            outputs.add(neuron.produceOutput(inputs));
        }

        return outputs;
    }
}