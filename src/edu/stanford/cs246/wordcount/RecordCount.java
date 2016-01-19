package edu.stanford.cs246.wordcount;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.Mapper.Context;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

public class RecordCount extends Configured implements Tool {
   public static void main(String[] args) throws Exception {
      System.out.println(Arrays.toString(args));
      int res = ToolRunner.run(new Configuration(), new RecordCount(), args);
      
      System.exit(res);
   }

   @Override
   public int run(String[] args) throws Exception {
      System.out.println(Arrays.toString(args));
      Job job = new Job(getConf(), "RecordCount");
      job.setJarByClass(RecordCount.class);
      job.setOutputKeyClass(Text.class);
      job.setOutputValueClass(Text.class);

      job.setMapperClass(Map.class);
      job.setReducerClass(Reduce.class);

      job.setInputFormatClass(TextInputFormat.class);
      job.setOutputFormatClass(TextOutputFormat.class);

      FileInputFormat.addInputPath(job, new Path(args[0]));
      FileOutputFormat.setOutputPath(job, new Path(args[1]));

      job.waitForCompletion(true);
      
      return 0;
   }
   
   public static class Map extends Mapper<LongWritable, Text, Text, Text> {
	      private Text recordValue = new Text();
	      private Text record = new Text();

	      @Override
	      public void map(LongWritable key, Text value, Context context)
	              throws IOException, InterruptedException {
	    	 String[] worldArray = value.toString().split("\\s+");
	    			 
	    	 record.set(worldArray[0]);
	    	 recordValue.set(worldArray[2] + worldArray[3]);
	    	 context.write(record, recordValue);
	    	 
	      }
	   }

	   public static class Reduce extends Reducer<Text, Text, Text, IntWritable> {
		  HashMap<String, Boolean> strsMap = new HashMap<String, Boolean>();
		   
	      @Override
	      public void reduce(Text key, Iterable<Text> values, Context context)
	              throws IOException, InterruptedException {
	         int sum = 0;
	         
	         // values are not sorted
	         // using hashmap is safer at low volume
	         strsMap.clear();
	         
	         //count unique Text values here (Assuming Sorted?)
	         for (Text val : values) {
	        	 String str = val.toString();
	        	 if(!strsMap.containsKey(str)){
	        		 strsMap.put(str, true);
	        		 sum++;
	        	 }
	          }
	         
	         context.write(key, new IntWritable(sum));
	      }
	   }
}