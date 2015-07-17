package Classifiers;
import weka.classifiers.*;
import weka.core.Instance;
import weka.core.Instances;

@SuppressWarnings("serial")
public class HardCodedClassifier extends Classifier
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
