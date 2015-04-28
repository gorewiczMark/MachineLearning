import weka.classifiers.*;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.*;

import java.io.IOException;
import java.io.File;

public class HardCodedClassifier extends AbstractClassifier
{ 
	public void csvToarff(String source, String destination) throws IOException
	{
		CSVLoader loader = new CSVLoader();
		loader.setSource(new File(source));
		Instances data = loader.getDataSet();
		
		ArffSaver saver = new ArffSaver();
		saver.setInstances(data);
		saver.setFile(new File(destination));
		saver.writeBatch();		
	}

	public Instances createInstance() throws Exception
	{
		DataSource source = new DataSource("iris.arff");
		Instances data = source.getDataSet();
		if (data.classIndex() == -1)
		{
			data.setClassIndex(data.numAttributes() - 1);
		}
		
		return data;
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
	
	@Override
	public void buildClassifier(Instances data) throws Exception 
	{
		return;	
	}	
	
	@Override
	public double classifyInstance(Instance data)
	{
		return 0;
	}
	
	public static void main(String args[]) throws Exception
	{
		HardCodedClassifier classifier = new HardCodedClassifier();
		classifier.csvToarff("iris.csv", "iris.arff");
		Instances data = classifier.createInstance();
		Instances newData = classifier.randomizeData(data);
		Instances trainData = classifier.splitData(newData,"30",false);
		Instances testData = classifier.splitData(newData,"30",true);

		classifier.buildClassifier(trainData);
		Evaluation eval = new Evaluation(trainData);
		eval.evaluateModel(classifier, testData);
		System.out.println(eval.toSummaryString("\nResults\n======\n", false));
		
	}


}
