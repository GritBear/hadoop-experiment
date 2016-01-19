package mapreducejobs;

import java.io.BufferedReader;
import java.io.DataInput;
import java.io.DataOutput;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.HashMap;

import mapreducejobs.MaxMutualFriend.NodePairWritable;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

public class AssociationSupport extends Configured implements Tool {
	private  static int supportThreshold = 100; 

	public static void main(String[] args) throws Exception {
		int res = ToolRunner.run(new Configuration(), new AssociationSupport(), args);
		System.exit(res);
	}

	@Override
	public int run(String[] args) throws Exception {
		System.out.println(Arrays.toString(args));
		//switch for jobs
		switch(args[0]){
		case "pairCount":
			runPairCountJob(args);
			break;
		case "tripleCount":
			runTripleCountJob(args);
			break;
		case "itemCount":
		default:
			runItemCountJob(args);
		}

		return 0;
	}
	
	/**
	 * Performing item counting with pre-defined support threshold
	 * @param args
	 * @throws Exception
	 */
	
	public void runItemCountJob(String[] args) throws Exception{
		System.out.println("start item count job");
		Job job = new Job(getConf(), "AssociationSupport-ItemCount");
		job.setJarByClass(AssociationSupport.class);

		FileInputFormat.addInputPath(job, new Path(args[1]));
		job.setInputFormatClass(TextInputFormat.class);

		FileOutputFormat.setOutputPath(job, new Path(args[2]));
		job.setOutputFormatClass(TextOutputFormat.class);

		job.setMapperClass(MapItem.class);
		job.setMapOutputKeyClass(Text.class);
		job.setMapOutputValueClass(IntWritable.class);

		job.setReducerClass(ReduceItem.class);
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(IntWritable.class);

		job.waitForCompletion(true);
	}

	public static class MapItem extends Mapper<LongWritable, Text, Text, IntWritable> {
		private final static IntWritable ONE = new IntWritable(1);
		private Text word = new Text();

		@Override
		public void map(LongWritable key, Text value, Context context)
				throws IOException, InterruptedException {
			for (String token: value.toString().split("\\s+")) {
				word.set(token);
				context.write(word, ONE);
			}
		}
	}

	public static class ReduceItem extends Reducer<Text, IntWritable, Text, IntWritable> {
		@Override
		public void reduce(Text key, Iterable<IntWritable> values, Context context)
				throws IOException, InterruptedException {
			int sum = 0;
			for (IntWritable val : values) {
				sum += val.get();
			}
			if(sum >= supportThreshold){
				context.write(key, new IntWritable(sum));
			}
		}
	}
	/********************************************************************************************/
	
	/**
	 * Using item count output as candidate items to find frequent pairs that meets support threshold
	 * @param args
	 * @throws Exception
	 */
	static HashMap<String, Integer> itemCandidate = new HashMap<String, Integer>();
	
	public static class PairWritableComparable implements WritableComparable<PairWritableComparable> {
		Text NodeId1 = new Text();
		Text NodeId2 = new Text();

		@Override
		public void readFields(DataInput in) throws IOException {
			NodeId1.readFields(in);
			NodeId2.readFields(in);
		}

		@Override
		public void write(DataOutput out) throws IOException {
			NodeId1.write(out);
			NodeId2.write(out);
		}
		
		public static PairWritableComparable make(String nodeId1, String nodeId2) {
			PairWritableComparable w = new PairWritableComparable();
			w.NodeId1.set(nodeId1);
			w.NodeId2.set(nodeId2);
			return w;
		}
		
		public Text getNodeId1(){
			return NodeId1;
		}
		
		public Text getNodeId2(){
			return NodeId2;
		}

		@Override
		public int compareTo(PairWritableComparable o) {
			if(NodeId1.compareTo(o.getNodeId1()) == 0){
				return NodeId2.compareTo(o.getNodeId2());
			}else{
				return NodeId1.compareTo(o.getNodeId1());
			}
		}
	}
	
	
	public void runPairCountJob(String[] args) throws Exception{
		System.out.println("start pair count job");
		loadItemCandidates(args[1], getConf());
		
		Job job = new Job(getConf(), "AssociationSupport-PairCount");
		job.setJarByClass(AssociationSupport.class);

		FileInputFormat.addInputPath(job, new Path(args[2]));
		job.setInputFormatClass(TextInputFormat.class);

		FileOutputFormat.setOutputPath(job, new Path(args[3]));
		job.setOutputFormatClass(TextOutputFormat.class);

		job.setMapperClass(MapPair.class);
		job.setMapOutputKeyClass(PairWritableComparable.class);
		job.setMapOutputValueClass(IntWritable.class);
		
		if(args.length > 4){
			if(args[4].equals("countOnly")){
				job.setReducerClass(ReducePairCount.class);
				job.setOutputKeyClass(Text.class);
				job.setOutputValueClass(IntWritable.class);
				job.waitForCompletion(true);
				return;
			}
		}
		
		job.setReducerClass(ReducePair.class);
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(FloatWritable.class);

		job.waitForCompletion(true);
		
	}
	
	// load itemCandidates
	public static void loadItemCandidates(String pathStr, Configuration conf) throws FileNotFoundException, IOException{
		itemCandidate.clear();
		Path p = new Path(pathStr);
		FileSystem fs = FileSystem.get(conf);
		
		FSDataInputStream in = fs.open(p);
		
		try (BufferedReader br = new BufferedReader(new InputStreamReader(in))) {
		    String line;
		    while ((line = br.readLine()) != null) {
		       String[] strs = line.split("\\t");
		       //debug
		       //System.out.println(strs[0]);
		       itemCandidate.put(strs[0], Integer.parseInt(strs[1]));
		    }
		}
	}
	
	public static class MapPair extends Mapper<LongWritable, Text, PairWritableComparable, IntWritable> {
		private final static IntWritable ONE = new IntWritable(1);

		@Override
		public void map(LongWritable key, Text value, Context context)
				throws IOException, InterruptedException {
			
			String[] tokens = value.toString().split("\\s+");
			
			if(tokens.length < 2){
				return;
			}
			
			for(int i = 0; i < tokens.length -1; i++){
				String t1 = tokens[i];
				if(!itemCandidate.containsKey(t1)){
					//debug
					//System.out.println("bypass non candidate");
					continue;
				}
				
				for(int j = i + 1; j < tokens.length; j++){
					String t2 = tokens[j];
					if(!itemCandidate.containsKey(t2)){
						//debug
						//System.out.println("bypass non candidate");
						continue;
					}
					
					/*
					//there should be no dup
					if(t1.compareTo(t2) == 0){
						continue;
					}
					*/
					
					PairWritableComparable pair;
					//making sure the order is correct so we have no dup pairs
					if(t1.compareTo(t2) < 0){
						pair = PairWritableComparable.make(t1, t2);
					}else{
						pair = PairWritableComparable.make(t2, t1);
					}
					context.write(pair, ONE);
				}
			}
		}
	}

	public static class ReducePair extends Reducer<PairWritableComparable, IntWritable, Text, FloatWritable> {
		Text outputKey = new Text();
		FloatWritable confidence = new FloatWritable();
		
		@Override
		public void reduce(PairWritableComparable key, Iterable<IntWritable> values, Context context)
				throws IOException, InterruptedException {
			int sum = 0;
			for (IntWritable val : values) {
				sum += val.get();
			}
			
			if(sum >= supportThreshold){
				String node1 = key.getNodeId1().toString();
				String node2 = key.getNodeId2().toString();
				
				float nodeCount1 = itemCandidate.get(node1);
				float nodeCount2 = itemCandidate.get(node2);
				
				outputKey.set(node1 + "->" + node2);
				confidence.set(((float)sum)/nodeCount1);
				context.write(outputKey, confidence);
				
				outputKey.set(node2 + "->" + node1);
				confidence.set(((float)sum)/nodeCount2);
				context.write(outputKey, confidence);
			}
		}
	}
	
	public static class ReducePairCount extends Reducer<PairWritableComparable, IntWritable, Text, IntWritable> {
		Text outputKey = new Text();
		IntWritable count = new IntWritable();
		
		@Override
		public void reduce(PairWritableComparable key, Iterable<IntWritable> values, Context context)
				throws IOException, InterruptedException {
			int sum = 0;
			for (IntWritable val : values) {
				sum += val.get();
			}
			
			if(sum >= supportThreshold){
				String node1 = key.getNodeId1().toString();
				String node2 = key.getNodeId2().toString();
				
				outputKey.set(node1 + "," + node2);
				count.set(sum);
				context.write(outputKey, count);
			}
		}
	}
	
	
	/*********************************************************************************************/
	public void runTripleCountJob(String[] args) throws Exception{
		System.out.println("start triple count job");

	}
}