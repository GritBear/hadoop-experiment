package mapreducejobs;

import java.io.BufferedReader;
import java.io.DataInput;
import java.io.DataOutput;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.AbstractMap.SimpleEntry;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.Map.Entry;
import java.util.Set;

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
	private static final int supportThreshold = 100; 

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
		Configuration conf = getConf();
		conf.setInt("supportThreshold", supportThreshold);
		
		Job job = Job.getInstance(conf);
		job.setJobName("itemCount");
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
			
			String[] tokens = value.toString().split("\\s+");
			Set<String> set = new LinkedHashSet<String>(Arrays.asList(tokens));
			tokens = set.toArray(new String[0]);
			
			for (String token: tokens) {
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
			
			int supportThreshold = context.getConfiguration().getInt("supportThreshold", 100);
			
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
		
		Configuration conf = getConf();
		conf.setInt("supportThreshold", supportThreshold);
		conf.set("CondidatePath", args[1]);
		
		Job job = Job.getInstance(conf);
		job.setJobName("AssociationSupport-PairCount");
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
	
	// load candidateTables
	public static HashMap<String, Integer> loadcandidateTables(String pathStr, Configuration conf) throws FileNotFoundException, IOException{
		HashMap<String, Integer> candidateTable = new HashMap<String, Integer>();
		candidateTable.clear();
		Path p = new Path(pathStr);
		FileSystem fs = FileSystem.get(conf);
		
		FSDataInputStream in = fs.open(p);
		
		try (BufferedReader br = new BufferedReader(new InputStreamReader(in))) {
		    String line;
		    while ((line = br.readLine()) != null) {
		       String[] strs = line.split("\\t");
		       //debug
		       //System.out.println(strs[0]);
		       candidateTable.put(strs[0], Integer.parseInt(strs[1]));
		    }
		}
		
		return candidateTable;
	}
	
	public static class MapPair extends Mapper<LongWritable, Text, PairWritableComparable, IntWritable> {
		private final static IntWritable ONE = new IntWritable(1);
		private static HashMap<String, Integer> candidateTable;
		
		public void setup(Context context){
			try {
				candidateTable = loadcandidateTables(context.getConfiguration().get("CondidatePath"), context.getConfiguration());
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		
		@Override
		public void map(LongWritable key, Text value, Context context)
				throws IOException, InterruptedException {
			
			String[] tokens = value.toString().split("\\s+");
			Set<String> set = new LinkedHashSet<String>(Arrays.asList(tokens));
			tokens = set.toArray(new String[0]);
			
			if(tokens.length < 2){
				return;
			}
			
			Arrays.sort(tokens);
			
			for(int i = 0; i < tokens.length -1; i++){
				String t1 = tokens[i];
				if(!candidateTable.containsKey(t1)){
					//debug
					//System.out.println("bypass non candidate");
					continue;
				}
				
				for(int j = i + 1; j < tokens.length; j++){
					String t2 = tokens[j];
					if(!candidateTable.containsKey(t2)){
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
					
					PairWritableComparable pair = PairWritableComparable.make(t1, t2);
					context.write(pair, ONE);
				}
			}
		}
	}

	public static class ReducePair extends Reducer<PairWritableComparable, IntWritable, Text, FloatWritable> {
		Text outputKey = new Text();
		FloatWritable confidence = new FloatWritable();
		private static HashMap<String, Integer> candidateTable;
		
		@Override
		public void setup(Context context){
			Configuration conf = context.getConfiguration();
			try {
				candidateTable = loadcandidateTables(conf.get("CondidatePath"), conf);
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		
		@Override
		public void reduce(PairWritableComparable key, Iterable<IntWritable> values, Context context)
				throws IOException, InterruptedException {
			int sum = 0;
			for (IntWritable val : values) {
				sum += val.get();
			}
			
			int supportThreshold = context.getConfiguration().getInt("supportThreshold", 100);
			
			if(sum >= supportThreshold){
				String node1 = key.getNodeId1().toString();
				String node2 = key.getNodeId2().toString();
				
				float nodeCount1 = candidateTable.get(node1);
				float nodeCount2 = candidateTable.get(node2);
				
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
	
	/***
	 * 
	 * @param args
	 * @throws Exception
	 */
	
	public static class TripleWritableComparable implements WritableComparable<TripleWritableComparable> {
		Text NodeId1 = new Text();
		Text NodeId2 = new Text();
		Text NodeId3 = new Text();

		@Override
		public void readFields(DataInput in) throws IOException {
			NodeId1.readFields(in);
			NodeId2.readFields(in);
			NodeId3.readFields(in);
		}

		@Override
		public void write(DataOutput out) throws IOException {
			NodeId1.write(out);
			NodeId2.write(out);
			NodeId3.write(out);
		}
		
		public static TripleWritableComparable make(String nodeId1, String nodeId2, String nodeId3) {
			TripleWritableComparable w = new TripleWritableComparable();
			w.NodeId1.set(nodeId1);
			w.NodeId2.set(nodeId2);
			w.NodeId3.set(nodeId3);
			return w;
		}
		
		public Text getNodeId1(){
			return NodeId1;
		}
		
		public Text getNodeId2(){
			return NodeId2;
		}
		
		public Text getNodeId3(){
			return NodeId3;
		}
		
		@Override
		public int compareTo(TripleWritableComparable o) {
			if(NodeId1.compareTo(o.getNodeId1()) == 0){
				if(NodeId2.compareTo(o.getNodeId2()) == 0){
					return NodeId3.compareTo(o.getNodeId3());
				}else{
					return NodeId2.compareTo(o.getNodeId2());
				}
			}else{
				return NodeId1.compareTo(o.getNodeId1());
			}
		}
	}
	
	public void runTripleCountJob(String[] args) throws Exception{
		System.out.println("start triple count job");
		Configuration conf = getConf();
		conf.setInt("supportThreshold", supportThreshold);
		conf.set("CondidatePath", args[1]);
		
		Job job = Job.getInstance(conf);
		job.setJobName("AssociationSupport-TripleCount");
		job.setJarByClass(AssociationSupport.class);

		FileInputFormat.addInputPath(job, new Path(args[2]));
		job.setInputFormatClass(TextInputFormat.class);

		FileOutputFormat.setOutputPath(job, new Path(args[3]));
		job.setOutputFormatClass(TextOutputFormat.class);

		job.setMapperClass(MapTriple.class);
		job.setMapOutputKeyClass(TripleWritableComparable.class);
		job.setMapOutputValueClass(IntWritable.class);
		
		if(args.length > 4){
			if(args[4].equals("countOnly")){
				job.setReducerClass(ReduceTripleCount.class);
				job.setOutputKeyClass(Text.class);
				job.setOutputValueClass(IntWritable.class);
				job.waitForCompletion(true);
				return;
			}
		}
		
		job.setReducerClass(ReduceTriple.class);
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(FloatWritable.class);

		job.waitForCompletion(true);
	}
	
	public static class MapTriple extends Mapper<LongWritable, Text, TripleWritableComparable, IntWritable> {
		private final static IntWritable ONE = new IntWritable(1);
		private static HashMap<String, Integer> candidateTable;
		
		public void setup(Context context){
			try {
				candidateTable = loadcandidateTables(context.getConfiguration().get("CondidatePath"), context.getConfiguration());
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		
		@Override
		public void map(LongWritable key, Text value, Context context)
				throws IOException, InterruptedException {
			
			String[] tokens = value.toString().split("\\s+");
			Set<String> set = new LinkedHashSet<String>(Arrays.asList(tokens));
			tokens = set.toArray(new String[0]);
			
			if(tokens.length < 3){
				return;
			}
			
			Arrays.sort(tokens);
			
			for(int i = 0; i < tokens.length -2; i++){
				String t1 = tokens[i];
				for(int j = i + 1; j < tokens.length-1; j++){
					String t2 = tokens[j];
					if(!candidateTable.containsKey(t1 + "," + t2)){
						continue;
					}
					for(int k = j + 1; k < tokens.length; k++){
						String t3 = tokens[k];
						if(!candidateTable.containsKey(t1 + "," + t3) || !candidateTable.containsKey(t2 + "," + t3) ){
							continue;
						}
						

						TripleWritableComparable triple = TripleWritableComparable.make(t1, t2, t3);
						context.write(triple, ONE);
					}
					
				}
			}
		}
	}

	public static class ReduceTriple extends Reducer<TripleWritableComparable, IntWritable, Text, FloatWritable> {
		Text outputKey = new Text();
		FloatWritable confidence = new FloatWritable();
		private static HashMap<String, Integer> candidateTable;
		
		@Override
		public void setup(Context context){
			Configuration conf = context.getConfiguration();
			try {
				candidateTable = loadcandidateTables(conf.get("CondidatePath"), conf);
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		
		@Override
		public void reduce(TripleWritableComparable key, Iterable<IntWritable> values, Context context)
				throws IOException, InterruptedException {
			int sum = 0;
			for (IntWritable val : values) {
				sum += val.get();
			}
			
			int supportThreshold = context.getConfiguration().getInt("supportThreshold", 100);
			
			if(sum >= supportThreshold){
				String node1 = key.getNodeId1().toString();
				String node2 = key.getNodeId2().toString();
				String node3 = key.getNodeId3().toString();
				
				float nodeCount12 = candidateTable.get(node1 + "," + node2);
				float nodeCount13 = candidateTable.get(node1 + "," + node3);
				float nodeCount23 = candidateTable.get(node2 + "," + node3);
				
				outputKey.set(node1 + "," + node2 + "->" + node3);
				confidence.set(((float)sum)/nodeCount12);
				context.write(outputKey, confidence);
				
				outputKey.set(node1 + "," + node3 + "->" + node2);
				confidence.set(((float)sum)/nodeCount13);
				context.write(outputKey, confidence);
				
				outputKey.set(node2 + "," + node3 + "->" + node1);
				confidence.set(((float)sum)/nodeCount23);
				context.write(outputKey, confidence);
				
			}
		}
	}
	
	public static class ReduceTripleCount extends Reducer<TripleWritableComparable, IntWritable, Text, IntWritable> {
		Text outputKey = new Text();
		IntWritable count = new IntWritable();
		
		@Override
		public void reduce(TripleWritableComparable key, Iterable<IntWritable> values, Context context)
				throws IOException, InterruptedException {
			int sum = 0;
			for (IntWritable val : values) {
				sum += val.get();
			}
			
			if(sum >= supportThreshold){
				String node1 = key.getNodeId1().toString();
				String node2 = key.getNodeId2().toString();
				String node3 = key.getNodeId3().toString();
				
				outputKey.set(node1 + "," + node2 + "," + node3);
				count.set(sum);
				context.write(outputKey, count);
			}
		}
	}
}