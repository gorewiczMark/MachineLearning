package Classifiers.NNC;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Neuron {
    List<Double> weights = new ArrayList<>();
    static Random random = new Random(42);

    public Neuron(int inputCount) {
         double oneOver = 1.0 / Math.sqrt(inputCount);
        for (int i = 0; i < inputCount; i++) {
            weights.add(random.nextDouble() * 2.0 * oneOver - oneOver);
        }
    }

    public double produceOutput(List<Double> inputs) {
        if (inputs.size() != weights.size()) {
            throw new UnsupportedOperationException("wrong number of inputs. Expected "
                + weights.size() + " and got " + inputs.size());
        }

        double sum = 0;
        for (int i = 0; i < weights.size(); i++) {
            sum += weights.get(i) * inputs.get(i);
        }

        return (1.0 / (1.0 + Math.exp(-sum)));
    }
}