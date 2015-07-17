package Classifiers;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;
import java.util.Map.Entry;

@SuppressWarnings("serial")
public class KNNClassifier extends Classifier 
{
	final int k;
	Instances saved;
	
	public KNNClassifier(int k)
	{
		this.k = k;
	}
	
	@Override
	public void buildClassifier(Instances instances) throws Exception 
	{
		saved = new Instances(instances);
	}
	
	@Override
    public double classifyInstance(Instance instance) throws Exception 
	{
        HashMap<Instance, Double> map = new HashMap<>();
        List<Instance> kNearest = new ArrayList<>();
        
        for (int i = 0; i < saved.numInstances(); i++) 
        {
            Instance tmp = saved.instance(i);
            map.put(tmp, distance(tmp, instance));
        }

        dance: for (Entry<Instance, Double> inst : entriesSortedByValues(map)) 
        {
            kNearest.add(inst.getKey());
            if (kNearest.size() >= k) 
            {
                break dance;
            }
        }
        return getClassification(kNearest);
    }
	
	static <K,V extends Comparable<? super V>> List<Entry<K, V>> entriesSortedByValues(Map<K,V> map) 
	{
        List<Entry<K,V>> sortedEntries = new ArrayList<>(map.entrySet());
        
        Collections.sort(sortedEntries, new Comparator<Entry<K, V>>() 
        		{
                    @Override
                    public int compare(Entry<K, V> e1, Entry<K, V> e2) 
                    {
                        return e1.getValue().compareTo(e2.getValue());
                    }
                }
        );

        return sortedEntries;
    }
	
	private static double distance(Instance a, Instance b)
	{
		double total = 0;
		int totalAttri = a.numAttributes() - 1;
		double difference; 
		
		for (int i = 0; i < totalAttri; i++)
		{
			if (a.classIndex() == i)
			{
				continue;
			}
		    
			difference = 0;
			
			if (a.attribute(i).isNumeric())
			{
				difference = Math.abs(a.value(i) - b.value(i));
			}
			else
			{
				if (!a.stringValue(i).equals(b.stringValue(i)))
				{
					difference = 1;
				}
			}
			
			total += Math.pow(difference, totalAttri);	
		}
		return Math.pow(total, 1.0/totalAttri);
	}
	
	public static double getClassification(Instances instances) 
	{
        ArrayList<Instance> instanceList = new ArrayList<>(instances.numInstances());
        for (int i = 0; i < instances.numInstances(); i++) 
        {
            instanceList.add(instances.instance(i));
        }
        return getClassification(instanceList);
    }
	
	public static double getClassification(List<Instance> instances) 
	{
        int index = instances.get(0).classIndex();
        HashMap<Double, Integer> counts = new HashMap<>();
        int maxCount = 0;
        double maxRValue = 0;
        
        for (Instance instance : instances) 
        {
            double val = instance.value(index);
            
            if (!counts.containsKey(val)) 
            {
                counts.put(val, 1);
            }
            else 
            {
                counts.put(val, counts.get(val) + 1);
            }
        }
        
        for (Entry<Double, Integer> entry : counts.entrySet()) 
        {
            if (entry.getValue() > maxCount) 
            {
                maxCount = entry.getValue();
                maxRValue = entry.getKey();
            }
        }
        return maxRValue;
    }

}
