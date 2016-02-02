package mapreducejobs;

import java.io.BufferedReader;
import java.io.DataInput;
import java.io.DataOutput;
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
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

public class kMeanMerge extends Configured implements Tool {
	static final int MAX_ITER = 3;
	static final String centroidKey = "CENTROID_KEY";

	public static void main(String[] args) throws Exception {
		int res = ToolRunner.run(new Configuration(), new kMeanMerge(), args);
		System.exit(res);
	}

	@Override
	public int run(String[] args) throws Exception {
		System.out.println(Arrays.toString(args));

		String inputDir = args[0];
		String outputDir = args[1]+"/iter-" + 0; 
		String centroidDir = args[2];

		for(int i = 0; i < MAX_ITER; i++){
			System.out.println("Iteration: " + i);
			System.out.println("Centroid Dir: " + centroidDir);

			Configuration conf = getConf();
			conf.set(centroidKey, centroidDir);

			Job job = Job.getInstance(conf);
			job.setJobName("kMeanMerge-iter-" + i);
			job.setJarByClass(kMeanMerge.class);

			FileInputFormat.addInputPath(job, new Path(inputDir));
			job.setInputFormatClass(TextInputFormat.class);

			FileOutputFormat.setOutputPath(job, new Path(outputDir));
			job.setOutputFormatClass(TextOutputFormat.class);

			job.setMapperClass(Map.class);
			job.setMapOutputKeyClass(IntWritable.class);
			job.setMapOutputValueClass(NodeWritable.class);

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
		int dim = -1;
		Path p = new Path(pathStr);
		FileSystem fs = FileSystem.get(conf);
		FSDataInputStream in = fs.open(p);

		try (BufferedReader br = new BufferedReader(new InputStreamReader(in))) {
			String line;

			while ((line = br.readLine()) != null) {
				String[] row = line.toString().split("\\t");
				String value = row[row.length - 1];
				
				String[] strs = value.split("\\s+");
				
				if(dim < 0){
					dim = strs.length;
				}else if(dim != strs.length){
					continue;
				}
				
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

	public static class Map extends Mapper<LongWritable, Text, IntWritable, NodeWritable> {
		private IntWritable outKey = new IntWritable();
		private ArrayList<float[]> centroids;

		@Override
		public void setup(Context context){
			try {
				Configuration conf = context.getConfiguration();
				centroids = loadCentroidsTables(conf.get(centroidKey), context.getConfiguration());
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
				assert centroid.length == vec.length;

				//TODO add distance switch here
				float distance = calcL2SquareDistance(vec, centroid);

				if(distance < minDistance){
					minCentroid = i;
					minDistance = distance;
				}
			}

			if(minCentroid >= 0){
				outKey.set(minCentroid);
				NodeWritable outValue = NodeWritable.make(vec,minDistance);
				context.write(outKey, outValue);
			}
		}
	}

	//compute the loss and new centroids
	// starting with initial loss (no merging) as 0 in the loss array, the i-th iteration loss can be found at the ith position

	public static class Reduce extends Reducer<IntWritable, NodeWritable, Text, Text> {
		private ArrayList<float[]> centroids;
		private static float loss;
		private Text outputkey = new Text();
		private Text output = new Text();

		@Override
		public void setup(Context context){
			loss = 0;
			try {
				centroids = loadCentroidsTables(context.getConfiguration().get(centroidKey), context.getConfiguration());
			} catch (Exception e) {
				e.printStackTrace();
			}
		}

		@Override
		public void reduce(IntWritable key, Iterable<NodeWritable> values, Context context)
				throws IOException, InterruptedException {

			float[] centroid = centroids.get(key.get());

			float[] newCentroid = new float[centroid.length]; //this is initialized to 0.0 by default
			int count = 0;

			//count unique Text values here (Assuming Sorted?)
			for (NodeWritable val : values) {

				float[] vec = val.getNode();
				float distance = val.getLoss();
				
				loss += distance;

				//optimize centroid
				// add up the points
				assert newCentroid.length == vec.length;
				for(int i = 0; i < newCentroid.length; i++){
					newCentroid[i] += vec[i];
				}

				count ++;
			}

			// getting the average
			String outputStr = "";
			for(int i = 0; i < newCentroid.length; i++){ 
				newCentroid[i] = newCentroid[i]/(float)count;
				outputStr += newCentroid[i];
				outputStr += " ";
			}

			outputkey.set(Integer.toString(key.get()));
			output.set(outputStr);
			context.write(outputkey, output);
		}

		@Override
		public void cleanup(Context context)throws IOException, InterruptedException{
			outputkey.set("Loss");
			output.set(Float.toString(loss));
			context.write(outputkey, output);
			System.out.println("Loss: " + loss);
		}
	}

	public static class NodeWritable implements Writable {
		private ArrayPrimitiveWritable outRecord1 = new ArrayPrimitiveWritable();
		private FloatWritable outRecord2 = new FloatWritable();

		@Override
		public void readFields(DataInput in) throws IOException {
			outRecord1.readFields(in);
			outRecord2.readFields(in);
		}

		@Override
		public void write(DataOutput out) throws IOException {
			outRecord1.write(out);
			outRecord2.write(out);
		}

		public static NodeWritable make(float[] node, float loss) {
			NodeWritable w = new NodeWritable();
			w.outRecord1.set(node);
			w.outRecord2.set(loss);
			return w;
		}

		public float[] getNode(){
			return (float[])outRecord1.get();
		}

		public float getLoss(){
			return outRecord2.get();
		}
	}

}