package Classifiers.NNC;

import java.util.ArrayList;
import java.util.List;

public class Network{
    List<Layer> layers = new ArrayList<>();

    public Network(int inputCount, List<Integer> neuronCountsPerLayer) {
        if (neuronCountsPerLayer.isEmpty()) {
            throw new UnsupportedOperationException("neuronCountsPerLayer is empty");
        }

        // 1 for the bias
        layers.add(new Layer(neuronCountsPerLayer.get(0), inputCount + 1));

        for (int i = 1; i < neuronCountsPerLayer.size(); i++) {
            layers.add(new Layer(neuronCountsPerLayer.get(i),
                    neuronCountsPerLayer.get(i - 1) + 1));
        }
    }

    public List<Double> getOutputs(List<Double> inputs) {
        List<Double> outputs = new ArrayList<>(inputs);

        for (Layer layer : layers) {
            // Add bias
            outputs.add(1.0);

            outputs = layer.produceOutputs(outputs);
        }

        return outputs;
    }

    public double train(List<Double> inputs, double classification, double learningValue) {
        ArrayList<List<Double>> allOutputs = new ArrayList<>();
        List<Double> outputs = new ArrayList<>(inputs);
        // feed forward to calculate outputs

        for (Layer layer : layers) {
            outputs.add(1.0);
            outputs = layer.produceOutputs(outputs);
            allOutputs.add(outputs);
        }

        ArrayList<ArrayList<Double>> allErrors = new ArrayList<>();
        // work backwards to calculate errors

        // do output nodes
        ArrayList<Double> error = new ArrayList<>();
        List<Double> currentOutputs = allOutputs.get(allOutputs.size() - 1);
        Layer current = layers.get(layers.size() - 1);
        
        for (int i = 0; i < current.neurons.size(); i++) {

            double expected = (classification == i ? 1 : 0);
            double errorVal = currentOutputs.get(i) * (1 - currentOutputs.get(i)) * (currentOutputs.get(i) - expected);           
            error.add(errorVal);
        }

        allErrors.add(error);

        // hidden nodes are a different equation
        for (int i = layers.size() - 2; i >= 0; i--) {
            // hidden layer
            current = layers.get(i);
            error = new ArrayList<>();
            outputs = allOutputs.get(i);
            ArrayList<Double> followingError = allErrors.get(0);
            for (int j = 0; j < current.neurons.size(); j++) {
                // neuron in current hidden layer
                double sumError = 0;
                Layer nextLayer = layers.get(i + 1);
                for (int k = 0; k < followingError.size(); k++) {
                    // neuron in following layer
                    sumError += followingError.get(k) * nextLayer.neurons.get(k).weights.get(j);
                }

                double errorVal = outputs.get(j) * (1 - outputs.get(j)) * sumError;
                error.add(errorVal);
            }

            allErrors.add(0, error);
        }

        // feed forward to update weights based on errors
        inputs.add(1.0);
        allOutputs.add(0, inputs);
        for (int i = 0; i < layers.size(); i++) {
            // layer
            current = layers.get(i);
            for (int j = 0; j < current.neurons.size(); j++) {
                // neuron in layer
                Neuron neuron = current.neurons.get(j);
                for (int k = 0; k < neuron.weights.size(); k++) {
                    // weight in neuron

                    double newWeight = neuron.weights.get(k) - allOutputs.get(i).get(k) * allErrors.get(i).get(j) * learningValue;
                    neuron.weights.set(k, newWeight);
                }

                current.neurons.set(j, neuron);
            }

            layers.set(i, current);
        }

        // total error
        double totalError = 0;
        for (List<Double> l : allErrors) {
            for (Double d : l) {
                totalError += Math.abs(d);
            }
        }

        return totalError;
    }
}