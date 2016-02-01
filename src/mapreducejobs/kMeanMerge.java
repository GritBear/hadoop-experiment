package mapreducejobs;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.ArrayPrimitiveWritable;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.join.TupleWritable;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

public class kMeanMerge extends Configured implements Tool {
	static final int MAX_ITER = 20;
	static final String centroidKey = "CENTROID_KEY";

	public static void main(String[] args) throws Exception {
		int res = ToolRunner.run(new Configuration(), new kMeanMerge(), args);
		System.exit(res);
	}

	@Override
	public int run(String[] args) throws Exception {
		System.out.println(Arrays.toString(args));
		
		String inputDir = args[0];
		String outputDir = args[1]+"/iter-" + 1; 
		String centroidDir = args[2];

		for(int i = 0; i < MAX_ITER; i++){
			Configuration conf = getConf();
			conf.set(centroidKey, centroidDir);
			
			Job job = Job.getInstance(conf);
			job.setJobName("kMeanMerge");
			job.setJarByClass(kMeanMerge.class);
			
			FileInputFormat.addInputPath(job, new Path(inputDir));
			job.setInputFormatClass(TextInputFormat.class);

			FileOutputFormat.setOutputPath(job, new Path(outputDir));
			job.setOutputFormatClass(TextOutputFormat.class);

			job.setMapperClass(Map.class);
			job.setMapOutputKeyClass(IntWritable.class);
			job.setMapOutputValueClass(TupleWritable.class);

			job.setReducerClass(Reduce.class);
			job.setOutputKeyClass(Text.class);
			job.setOutputValueClass(Text.class);
			job.setNumReduceTasks(1); //ensure there is only 1 global output file

			boolean success = job.waitForCompletion(true);
			
			centroidDir = outputDir+"/part-r-00000";
			outputDir = args[1]+"/iter-" + (i + 1);
			
			if(!success){
				return 1;
			}
		}

		return 0;
	}

	// load centroids
	public static ArrayList<float[]> loadCentroidsTables(String pathStr, Configuration conf) throws FileNotFoundException, IOException{
		ArrayList<float[]> centroids = new ArrayList<float[]>(10);
		Path p = new Path(pathStr);
		FileSystem fs = FileSystem.get(conf);

		FSDataInputStream in = fs.open(p);

		try (BufferedReader br = new BufferedReader(new InputStreamReader(in))) {
			String line;
			
			while ((line = br.readLine()) != null) {
				String[] strs = line.split("\\s+");
				//TODO add some sanity check
				
				float[] centroid = new float[strs.length];
				for(int i = 0; i < strs.length; i++){
					centroid[i] = Float.parseFloat(strs[i]);
				}
				centroids.add(centroid);
			}
		}

		return centroids;
	}

	//distance computation
	//output L2^2, which is more handy than L2 itself. It also saves some computation costs.
	public static float calcL2SquareDistance(float[] vec1, float[] vec2){
		assert (vec1.length == vec2.length);
		
		float dis = 0;
		
		for(int i = 0; i < vec1.length; i++){
			float t = vec1[i] - vec2[i];
			dis += t * t;
		}
		
		return dis;
	}

	public static class Map extends Mapper<LongWritable, Text, IntWritable, TupleWritable> {
		private IntWritable outKey = new IntWritable();
		private ArrayPrimitiveWritable outRecord1 = new ArrayPrimitiveWritable(float.class);
		private FloatWritable outRecord2 = new FloatWritable();
		private ArrayList<float[]> centroids;
		
		@Override
		public void setup(Context context){
			try {
				centroids = loadCentroidsTables(context.getConfiguration().get(centroidKey), context.getConfiguration());
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		
		@Override
		public void map(LongWritable key, Text value, Context context)
				throws IOException, InterruptedException {
			String[] wordArray = value.toString().split("\\s+");
			
			// compute distance
			// convert string to vector
			float[] vec = new float[wordArray.length];
			for(int i = 0; i < wordArray.length; i++){
				vec[i] = Float.parseFloat(wordArray[i]);
			}
			
			int minCentroid = -1;
			float minDistance = Float.MAX_VALUE;
			
			// perform linear search for this centroid
			for(int i = 0; i < centroids.size(); i++){
				float[] centroid = centroids.get(i);
				
				//TODO add distance switch here
				float distance = calcL2SquareDistance(vec, centroid);
				
				if(distance < minDistance){
					minCentroid = i;
					minDistance = distance;
				}
			}
			
			if(minCentroid >= 0){
				outKey.set(minCentroid);
				outRecord1.set(vec);
				outRecord2.set(minDistance);
				Writable[] valueArr = {outRecord1,outRecord2};
				TupleWritable outValue = new TupleWritable(valueArr);
				context.write(outKey, outValue);
			}
		}
	}
	
	//compute the loss and new centroids
	// starting with initial loss (no merging) as 0 in the loss array, the i-th iteration loss can be found at the ith position
	
	public static class Reduce extends Reducer<IntWritable, TupleWritable, Text, Text> {
		private ArrayList<float[]> centroids;
		
		@Override
		public void setup(Context context){
			try {
				centroids = loadCentroidsTables(context.getConfiguration().get(centroidKey), context.getConfiguration());
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		
		@Override
		public void reduce(IntWritable key, Iterable<TupleWritable> values, Context context)
				throws IOException, InterruptedException {
			
			float[] centroid = centroids.get(key.get());
			
			//count unique Text values here (Assuming Sorted?)
			for (TupleWritable val : values) {
				
				float[] vec = (float[])((ArrayPrimitiveWritable)val.get(0)).get();
				float distance = (float)((FloatWritable)val.get(1)).get();
				
				
			}

			//context.write(key, new IntWritable(sum));
		}
	}
}