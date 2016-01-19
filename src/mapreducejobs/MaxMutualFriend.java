package mapreducejobs;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map.Entry;
import java.util.PriorityQueue;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

public class MaxMutualFriend extends Configured implements Tool {
	public static void main(String[] args) throws Exception {
		int res = ToolRunner.run(new Configuration(), new MaxMutualFriend(), args);
		System.exit(res);
	}
	
	@Override
	public int run(String[] args) throws Exception {
		System.out.println(Arrays.toString(args));
		Job job = new Job(getConf(), "MaxMutualFriend");
		job.setJarByClass(MaxMutualFriend.class);

		FileInputFormat.addInputPath(job, new Path(args[0]));
		job.setInputFormatClass(TextInputFormat.class);

		FileOutputFormat.setOutputPath(job, new Path(args[1]));
		job.setOutputFormatClass(TextOutputFormat.class);

		job.setMapperClass(Map.class);
		job.setMapOutputKeyClass(IntWritable.class);
		job.setMapOutputValueClass(NodePairWritable.class);

		job.setReducerClass(Reduce.class);
		job.setOutputKeyClass(IntWritable.class);
		job.setOutputValueClass(Text.class);

		job.waitForCompletion(true);

		return 0;
	}
	
	// composite value writable object
	public static class NodePairWritable implements Writable {
		int NodeId1;
		int NodeId2;
		int distanceDegree;  

		@Override
		public void readFields(DataInput in) throws IOException {
			NodeId1 = in.readInt();
			NodeId2 = in.readInt();
			distanceDegree = in.readInt();
		}

		@Override
		public void write(DataOutput out) throws IOException {
			out.writeInt(NodeId1);
			out.writeInt(NodeId2);
	        out.writeInt(distanceDegree);
		}
		
		public static NodePairWritable make(int nodeId1, int nodeId2,int distance) {
			NodePairWritable w = new NodePairWritable();
			w.NodeId1 = nodeId1;
			w.NodeId2 = nodeId2;
			w.distanceDegree = distance;
			return w;
		}
		
		public int getNodeId1(){
			return NodeId1;
		}
		
		public int getNodeId2(){
			return NodeId2;
		}
		
		public int getDistance(){
			return distanceDegree;
		}
		
	}
	
	public static class Map extends Mapper<LongWritable, Text, IntWritable, NodePairWritable> {

		@Override
		public void map(LongWritable key, Text value, Context context)
				throws IOException, InterruptedException {
			
			String[] row = value.toString().split("\\t");
			
			int sourceNode = Integer.parseInt(row[0]);
			
			if(row.length < 2){
				// it has no node connecting
				context.write(new IntWritable(sourceNode), NodePairWritable.make(sourceNode, sourceNode, 0));
				return;
			}
			
			String[] targetNodesStr = row[1].split(",");
			
			// convert to int once for all
			int[] targetIds = new int[targetNodesStr.length];
			for(int i = 0; i < targetNodesStr.length; i++){
				targetIds[i] = Integer.parseInt(targetNodesStr[i]);
			}
			
			for(int i = 0; i < targetIds.length; i++){
				int targetNodeId0 = targetIds[i];
				
				//emit immediate neighbors (distance 0)
				context.write(new IntWritable(sourceNode), NodePairWritable.make(sourceNode, targetNodeId0, 0));
				
				//emit the potential distance = 1 nodes
				for(int j = i + 1; j < targetIds.length; j++){
					int targetNodeId1 = targetIds[j];
					context.write(new IntWritable(targetNodeId0), NodePairWritable.make(targetNodeId0, targetNodeId1, 1));
				}
			}
		}
	}

	public static class Reduce extends Reducer<IntWritable, NodePairWritable, IntWritable, Text> {
		HashMap<Integer, Integer> frequencyMap = new HashMap<Integer, Integer>();
		PriorityQueue<Entry<Integer, Integer>> queue = new PriorityQueue<Entry<Integer, Integer>>(10, new NodeComparator());
		Text output = new Text();
		
		@Override
		public void reduce(IntWritable key, Iterable<NodePairWritable> values, Context context)
				throws IOException, InterruptedException {
			frequencyMap.clear(); //to save memory
			queue.clear();
			
			//count unique Text values here (Assuming Sorted?)
			for (NodePairWritable val : values) {
				//accumulate frequency
				int targetId = val.getNodeId2();
				int distance = val.getDistance();
				
				if(!frequencyMap.containsKey(targetId)){
					frequencyMap.put(targetId, distance);
				}else{
					if(distance == 0){
						frequencyMap.put(targetId, distance);
					}else if(frequencyMap.get(targetId) != 0){
						frequencyMap.put(targetId, frequencyMap.get(targetId) +  1);
					}
				}
			}
			
			//sort for final output
			Iterator<Entry<Integer, Integer>> it = frequencyMap.entrySet().iterator();
		    while (it.hasNext()) {
		        Entry<Integer, Integer> pair = it.next();
		        int count = pair.getValue();
		        
		        //priority queue
		        if(count > 0){ //0 means they are already friends
		        	queue.add(pair);
		        	// ideally we should remove the end of the queue here (will improve if running into memory constraint)
		        }
		        
		        it.remove(); // avoids a ConcurrentModificationException
		    }
		    
		    //write to output
		    String outputString = "";
			for(int i = 0; i < 10; i++){
				if(queue.size() > 0){
					Entry<Integer, Integer> pair = queue.poll();
					int targetId = pair.getKey();
					outputString += targetId;
					//outputString += (targetId + "[" + pair.getValue() + "]");//for debugging only
					if(i < 9 && queue.size() > 0){
						outputString += ",";
					}
				}
			}
			
			output.set(outputString);
			context.write(key, output);
		}
	}
	
	public static class NodeComparator implements Comparator<Entry<Integer, Integer>>
    {
		@Override
		public int compare(Entry<Integer, Integer> o1, Entry<Integer, Integer> o2) {
			if(o1.getValue() > o2.getValue()){
				return -1;
			}else if(o1.getValue() < o2.getValue()){
				return 1;
			}else{
				//when equal
				return o1.getKey() - o2.getKey();
			}
		}            
         
     }
}