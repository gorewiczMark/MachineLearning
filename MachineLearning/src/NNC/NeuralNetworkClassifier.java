package Classifiers.NNC;
import javafx.util.Pair;
import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class NeuralNetworkClassifier extends AbstractClassifier {
    Network network;
    int layers = 3;
    int iterations = 50;
    double learningFactor = 0.3;

    public NeuralNetworkClassifier(int layers, int iterations, double learningFactor) {
        this.layers = layers;
        this.iterations = iterations;
        this.learningFactor = learningFactor;
    }

    @Override
    public void buildClassifier(Instances instances) throws Exception {

        int inputCount = instances.numAttributes() - 1;

        List<Integer> nodesPerLayer = new ArrayList<>();

        for (int i = 0; i < layers - 1; i++) {
            nodesPerLayer.add(inputCount);
        }

        nodesPerLayer.add(instances.numDistinctValues(instances.classIndex()));

        network = new Network(inputCount, nodesPerLayer);

        ArrayList<Double> errorsPerIteration = new ArrayList<>();
        for (int j = 0; j < iterations; j++) {
            double errorsPer = 0;
            for (int k = 0; k < instances.numInstances(); k++) {
                Instance instance = instances.instance(k);

                List<Double> input = new ArrayList<>();

                for (int i = 0; i < instance.numAttributes(); i++) {
                    if (i == instance.classIndex()) {
                    } else if (Double.isNaN(instance.value(i))) {
                        input.add(0.0);
                    } else {
                        input.add(instance.value(i));
                    }

                }

                errorsPer += network.train(input, instance.value(instance.classIndex()), learningFactor);
            }

            errorsPerIteration.add(errorsPer);
        }

        for (Double d : errorsPerIteration) {
            System.out.println(d);
        }
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        List<Double> input = new ArrayList<>();
        for (int i = 0; i < instance.numAttributes(); i++) {
            if (Double.isNaN(instance.value(i)) && i != instance.classIndex())
                input.add(0.0);
            else if (i != instance.classIndex())
                input.add(instance.value(i));

        }

        List<Double> outputs = network.getOutputs(input);
        double largeVal = -1;
        int index = -1;
        for (int i = 0; i < outputs.size(); i++) {
            double tmp = outputs.get(i);
            if (tmp > largeVal) {
                largeVal = tmp;
                index = i;
            }
        }

        return index;
    }
}



