package Classifiers;
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

@SuppressWarnings("serial")
public class HardCodedClassifier extends AbstractClassifier
{   
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
	
}
