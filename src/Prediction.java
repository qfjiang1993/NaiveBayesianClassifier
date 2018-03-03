import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.examples.MultiFileWordCount;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.hbase.util.Bytes;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.CombineFileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.CombineFileRecordReader;
import org.apache.hadoop.mapreduce.lib.input.CombineFileSplit;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.LazyOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.MultipleOutputs;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.mapreduce.lib.reduce.IntSumReducer;
import org.apache.hadoop.util.LineReader;

import java.io.IOException;
import java.net.URI;
import java.text.DecimalFormat;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;

/**
 * @author QFJiang 2016/11/15
 *
 * 统计测试集每个文档的词频，预测文档所属类型
 */
public class Prediction {

    private static HashMap<String, Double> priorMap = new HashMap<>();
    private static HashMap<String, Integer> sizeMap = new HashMap<>();
    private static HashSet<String> dictSet = new HashSet<>();
    private static HashMap<String, HashMap<String, Double>> condMap = new HashMap<>();
    private static HashMap<String, HashMap<String, Double>> resultMap = new HashMap<>();

    public static void main(String[] args) throws Exception {

        long start = System.currentTimeMillis();
        if (args.length < 4) {
            System.err.println("Usage : Prediction <hdfs> <in> <train> <out>");
            System.exit(2);
        }

        Configuration conf = new Configuration();
        conf.set("mapreduce.input.fileinputformat.input.dir.recursive", "true");
        FileSystem fs = FileSystem.get(URI.create(args[0]), conf);

        FSDataInputStream input;
        FSDataOutputStream output;
        LineReader reader;
        Text line = new Text();

        // 读取先验概率
        input = fs.open(new Path(args[0] + args[2], "prior.txt"));
        reader = new LineReader(input);
        while (reader.readLine(line) > 0) {
            String[] split = line.toString().split("\t");
            // 取对数，用于后续计算测试文档的条件概率
            double p = Math.log(Double.valueOf(split[1]));
            priorMap.put(split[0], p);
        }

        // 读取字典集合
        input = fs.open(new Path(args[0] + args[2], "dictSet.txt"));
        reader = new LineReader(input);
        while (reader.readLine(line) > 0) {
            dictSet.add(line.toString());
        }

        // 读取字典集合及分类文档大小，用于计算未出现在字典中的词条的条件概率
        input = fs.open(new Path(args[0] + args[2], "classSize.txt"));
        reader = new LineReader(input);
        while (reader.readLine(line) > 0) {
            String[] split = line.toString().split("\t");
            sizeMap.put(split[0], Integer.valueOf(split[1]));
        }

        // 读取条件概率
        input = fs.open(new Path(args[0] + args[2], "cond.txt"));
        reader = new LineReader(input);
        while (reader.readLine(line) > 0) {
            String[] split = line.toString().split("\t");
            HashMap<String, Double> map = new HashMap<>();
            for (int i = 1; i < split.length; i++) {
                String pair[] = split[i].split(":");
                map.put(pair[0], Double.valueOf(pair[1]));
            }
            condMap.put(split[0], map);
        }

        input.close();
        reader.close();

        Job job = Job.getInstance(conf, "Prediction");
        job.setJarByClass(Prediction.class);
        job.setInputFormatClass(MyInputFormat.class);
        job.setOutputFormatClass(FileOutputFormat.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);
        job.setMapperClass(MyMapper.class);
        job.setCombinerClass(IntSumReducer.class);
        job.setReducerClass(MultiOutputReducer.class);
        LazyOutputFormat.setOutputFormatClass(job, TextOutputFormat.class);
        FileInputFormat.addInputPath(job, new Path(args[0], args[1]));
        FileOutputFormat.setOutputPath(job, new Path(args[0], args[3] + "/MR-out"));

        int ret = job.waitForCompletion(true) ? 0 : 1;

        if (ret == 0) {

            output = fs.create(new Path(args[3], "prediction.txt"));
            System.out.println("Prediction Result stored in " + args[3] + "/prediction.txt");

            for (String clsDoc : resultMap.keySet()) {
                HashMap<String, Double> map = resultMap.get(clsDoc);
                double pMax = Integer.MIN_VALUE;
                String predClass = "";
                String record = "";
                // 找出概率最大的分类
                for (String className : map.keySet()) {
                    double p = map.get(className);
                    if (p > pMax) {
                        pMax = p;
                        predClass = className;
                    }
                    record += className + ":" + new DecimalFormat("0.00").format(p) + "\t";
                }
                record = clsDoc.split("-")[1] + "\t" + clsDoc.split("-")[0] + "\t"
                        + predClass + "\t" + record + System.lineSeparator();
                output.write(Bytes.toBytes(record));
            }
            output.close();
        }

        long end = System.currentTimeMillis();
        System.out.println("Total time: " + (end - start) + "ms");
        System.exit(ret);
    }

    private static class MyInputFormat extends CombineFileInputFormat<MultiFileWordCount.WordOffset, Text> {

        public RecordReader<MultiFileWordCount.WordOffset, Text> createRecordReader(InputSplit split, TaskAttemptContext context) throws IOException {
            return new CombineFileRecordReader<>((CombineFileSplit) split, context, MyRecordReader.class);
        }
    }

    private static class MyRecordReader extends MultiFileWordCount.CombineFileLineRecordReader {

        String pathName;
        String parentName;

        public MyRecordReader(CombineFileSplit split, TaskAttemptContext context, Integer index) throws IOException {
            super(split, context, index);
            pathName = split.getPath(index).getName();
            parentName = split.getPath(index).getParent().getName();
        }

        @Override
        public Text getCurrentValue() throws IOException, InterruptedException {
            // map输出格式：className-pathName    item
            String record = parentName + "-" + pathName + "\t" + super.getCurrentValue().toString();
            return new Text(record);
        }
    }

    private static class MyMapper extends Mapper<MultiFileWordCount.WordOffset, Text, Text, IntWritable> {

        private final static IntWritable ONE = new IntWritable(1);
        private Text word = new Text();

        public void map(MultiFileWordCount.WordOffset key, Text value, Context context) throws IOException, InterruptedException {
            word.set(value);
            context.write(word, ONE);
        }
    }

    private static class MultiOutputReducer extends Reducer<Text, IntWritable, Text, IntWritable> {

        private IntWritable count = new IntWritable();
        private MultipleOutputs<Text, IntWritable> multipleOutputs;

        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            super.setup(context);
            multipleOutputs = new MultipleOutputs<>(context);
        }

        @Override
        protected void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable val : values) {
                sum += val.get();
            }
            count.set(sum);
            String[] split = key.toString().split("\t");
            String pathName = split[0];     // 分类名称:测试文档名称
            String item = split[1];         // 词条
            // pathName相同的record输出到同一个reduce处理
            multipleOutputs.write(new Text(item), count, pathName);

            // 根据pathName统计每个文档属于不同分类的条件概率
            if (!resultMap.containsKey(pathName)) {
                HashMap<String, Double> map = new HashMap<>();
                for (Map.Entry entry : priorMap.entrySet()) {
                    String className = (String) entry.getKey(); // 分类名
                    double prior = (double) entry.getValue();   // 先验概率
                    double cond;
                    if (!dictSet.contains(item)) {
                        cond = (double) 1 / (sizeMap.get("DICT") + sizeMap.get(className));
                    } else {
                        cond = condMap.get(item).get(className);
                    }
                    map.put(className, prior + sum * Math.log(cond));
                }
                resultMap.put(pathName, map);
            } else {
                HashMap<String, Double> map = resultMap.get(pathName);
                for (Map.Entry entry : priorMap.entrySet()) {
                    String className = (String) entry.getKey();
                    double cond;
                    if (!dictSet.contains(item)) {
                        cond = (double) 1 / (sizeMap.get("DICT") + sizeMap.get(className));
                    } else {
                        cond = condMap.get(item).get(className);
                    }
                    map.replace(className, map.get(className) + sum * Math.log(cond));
                }
                resultMap.replace(pathName, map);
            }
        }

        @Override
        protected void cleanup(Context context) throws IOException, InterruptedException {
            multipleOutputs.close();
        }
    }

}
