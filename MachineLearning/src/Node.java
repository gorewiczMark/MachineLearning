package Classifiers;
import weka.core.Attribute;
import weka.core.Instance;

import java.util.*;

public class Node 
{
	List<Instance> instances;
	private Map<Node, Double> child = new HashMap<>();
	private Map<Double,Node> child2 = new HashMap<>();
	Attribute attribute;
	boolean isLeaf = false;
	double leafValue;

	public Node(double pLeafValue)
	{
		isLeaf = true;
		this.leafValue = pLeafValue;
	}
	
	public Node(List<Instance> pInstances, Attribute pAttribute)
	{
		this.instances = pInstances;
		this.attribute = pAttribute;
	}
	
	public boolean isLeaf()
	{
		return isLeaf;
	}
	
	public Attribute getAttribute()
	{
		return attribute;
	}
	
	public void addChild(Double pValue, Node pN)
	{
		child.put(pN, pValue);
		child2.put(pValue,pN);
	}
	
	public Double get(Node pN)
	{
		return child.get(pN);
	}
	
	public Node get(Double pD)
	{
		if (child2.get(pD) != null)
		{
			return child2.get(pD);
		}
		else
		{
			Double close = null;
			double dist = Double.MAX_VALUE;
			for (Double key : child2.keySet()) 
			{
                if (Math.abs(pD - key) < dist) 
                {
                    dist = Math.abs(pD - key);
                    close = key;
                }
            }

            return child2.get(close);
		}
	}
	
	public Set<Node> getChildren()
	{
		return child.keySet();
	}
}
