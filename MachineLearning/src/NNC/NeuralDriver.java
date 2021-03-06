package Classifiers.NNC;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.Standardize;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.RemovePercentage;

import Classifiers.NNC.NeuralNetworkClassifier;

public class NeuralDriver
{	
	String data = "iris.csv";
			
	public Instances createInstance() throws Exception
	{
		DataSource source = new DataSource(data);
        Instances dataSetPre = source.getDataSet();

        dataSetPre.setClassIndex(dataSetPre.numAttributes() - 1);

        Standardize stand = new Standardize();
        stand.setInputFormat(dataSetPre);

        Discretize discretize = new Discretize();
        discretize.setInputFormat(dataSetPre);

        NumericToNominal ntb = new NumericToNominal();
        ntb.setInputFormat(dataSetPre);

        Instances dataSet = dataSetPre;

        dataSet = Filter.useFilter(dataSet, discretize);
        dataSet = Filter.useFilter(dataSet, stand);
        dataSet = Filter.useFilter(dataSet, ntb);
       
        return dataSet;
	}
	
	public Instances randomizeData(Instances pData) throws Exception
	{
		String[] options = new String[2];
		options[0] = "-S";                                    
		options[1] = "42";                                     
		Randomize random = new Randomize();                    
		random.setOptions(options);                           
		random.setInputFormat(pData);                          
		Instances newData = Filter.useFilter(pData, random);   
        return newData; 	
	}
	
	public Instances splitData(Instances pData, String percentCut, 
			Boolean invert) throws Exception
	{
		String inversion;
		if ( invert )
		{
		   inversion = "-V";
		}
		else
			inversion = "";
		String[] options = new String[3];
		options[0] = "-P";                                  
		options[1] = percentCut;       
		options[2] = inversion;
		RemovePercentage percent = new RemovePercentage();                 
		percent.setOptions(options);                         
		percent.setInputFormat(pData);                       
		Instances newData = Filter.useFilter(pData, percent);
        return newData; 
	}
	
	public void train(Instances pData) throws Exception
	{
		String[] options = new String[1];
		options[0] = "-U";
		J48 tree = new J48();
		tree.setOptions(options);
		tree.buildClassifier(pData);
	}
	
	public static void main(String args[]) throws Exception
	{
		NeuralDriver driver = new NeuralDriver();
	
		NeuralNetworkClassifier classifier = new NeuralNetworkClassifier(3,20000,0.5);

		Instances data = driver.createInstance();
		Instances newData = driver.randomizeData(data);
		
		Instances trainData = driver.splitData(newData,"30",false);
		Instances testData = driver.splitData(newData,"30",true);

		classifier.buildClassifier(trainData);
		Evaluation eval = new Evaluation(trainData);
		eval.evaluateModel(classifier, testData);
		System.out.println(eval.toSummaryString("\nResults\n======\n", false));

		
		
	}
}