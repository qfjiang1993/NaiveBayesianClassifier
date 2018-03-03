import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.examples.MultiFileWordCount;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileStatus;
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

import java.io.IOException;
import java.net.URI;
import java.text.DecimalFormat;
import java.util.HashMap;
import java.util.HashSet;

/**
 * @author QFJiang 2016/11/14
 *
 * 统计训练集每个分类的词频，计算每个分类的先验概率及每个词条的条件概率
 *
 * 参考 org.apache.hadoop.examples.Training
 *
 * 主要做如下修改：
 * 1、CombineFileLineRecordReader类的getCurrentValue()方法
 *    使其返回格式如 <className    item    count>，即 “类名    词条”，用"\t"分隔
 * 2、MapClass类的map()方法，record格式为 <className    item>
 *    因此，直接将record输出，不使用Tokenizer分词，否则会将类名、词条分为两条record
 * 3、MultiOutputReducer类，根据map输出的不同className输出到不同的reduce
 *    最后输出文件名为className-r-xxxxx，record格式 <item   count>
 */
public class Training {

    private static HashSet<String> dictSet = new HashSet<>();
    private static HashMap<String, Integer> classSize = new HashMap<>();
    private static HashMap<String, HashMap<String, Integer>> classItemCount = new HashMap<>();

    public static void main(String[] args) throws Exception {

        long start = System.currentTimeMillis();
        if (args.length < 3) {
            System.err.println("Usage : Training <hdfs> <in> <out>");
            System.exit(2);
        }

        Configuration conf = new Configuration();
        conf.set("mapreduce.input.fileinputformat.input.dir.recursive", "true");
        FileSystem fs = FileSystem.get(URI.create(args[0]), conf);

        HashMap<String, Integer> docsMap = new HashMap<>(); // 分类文档计数
        HashMap<String, Double> priorMap = new HashMap<>(); // 先验概率
        HashMap<String, String> condMap = new HashMap<>();  // 条件概率
        int total = 0;                                      // 全部文档总数

        // 步骤一：统计分类文档数、文档总数，并计算分类的先验概率
        FileStatus[] listStatus = fs.listStatus(new Path(args[1]));
        for (FileStatus status : listStatus) {
            // 目录名称即为分类名称，文档个数即为分类个数
            String name = status.getPath().getName();
            int num = fs.listStatus(status.getPath()).length;
            docsMap.put(name, num);
            total += num;
        }
        for (String className : docsMap.keySet()) {
            double p = docsMap.get(className) / (double) total;
            priorMap.put(className, p);
        }

        // 步骤二：使用MR统计分类词条总数、每个词条的词频，及字典集合
        Job job = Job.getInstance(conf, "Training");
        job.setJarByClass(Training.class);
        job.setInputFormatClass(MyInputFormat.class);
        job.setOutputFormatClass(FileOutputFormat.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);
        job.setMapperClass(MyMapper.class);
        job.setCombinerClass(IntSumReducer.class);
        job.setReducerClass(MultiOutputReducer.class);
        LazyOutputFormat.setOutputFormatClass(job, TextOutputFormat.class);
        FileInputFormat.addInputPath(job, new Path(args[0], args[1]));
        FileOutputFormat.setOutputPath(job, new Path(args[0], args[2] + "/MR-out"));

        int ret = job.waitForCompletion(true) ? 0 : 1;

        // MR正确完成，则保存计算结果并计算词条的条件概率
        if (ret == 0) {

            FSDataOutputStream output;

            // 步骤三：计算每个词条属于不同分类的条件概率
            //记录格式：item    classA:pA    classB:pB    ···
            double dictSize = (double) dictSet.size();
            for (String item : dictSet) {
                String record = "";
                for (String className : classSize.keySet()) {
                    Object o = classItemCount.get(className).get(item);
                    int count = o == null ? 0 : (int) o;
                    double d = (count + 1) / (dictSize + classSize.get(className));
                    record += className + ":" + new DecimalFormat("0.00000000").format(d) + "\t";
                }
                condMap.put(item, record);
            }

            // 步骤四：保存计算结果
            // 遍历priorMap得到先验概率, 保存到prior.txt
            output = fs.create(new Path(args[2], "prior.txt"));
            System.out.println("Prior Probability stored in " + args[2] + "/prior.txt");
            for (String cls : priorMap.keySet()) {
                String p = new DecimalFormat("0.0000").format(priorMap.get(cls));
                String record = cls + "\t" + p + System.lineSeparator();
                output.write(Bytes.toBytes(record));
                //System.out.print(record);
            }
            // 遍历dictSet得到字典集合, 保存到dictSet.txt
            output = fs.create(new Path(args[2], "dictSet.txt"));
            System.out.println("Dictionary Sets stored in " + args[2] + "/dictSet.txt");
            for (String item : dictSet) {
                String record = item + System.lineSeparator();
                output.write(Bytes.toBytes(record));
            }
            // 遍历classSize得到字典大小及分类词条总数, 保存到classSize.txt
            output = fs.create(new Path(args[2], "classSize.txt"));
            System.out.println("Class Size stored in " + args[2] + "/classSize.txt");
            String s = "DICT" + "\t" + dictSet.size() + System.lineSeparator();
            output.write(Bytes.toBytes(s));
            //System.out.println("DICT" + "\t" + dictSet.size());
            for (String cls : classSize.keySet()) {
                String record = cls + "\t" + classSize.get(cls) + System.lineSeparator();
                output.write(Bytes.toBytes(record));
                //System.out.print(record);
            }
            // 遍历condMap得到词条属于不同分类的条件概率，保存到cond.txt
            output = fs.create(new Path(args[2], "cond.txt"));
            System.out.println("Conditional Probability stored in " + args[2] + "/cond.txt");
            for (String item : condMap.keySet()) {
                String record = item + "\t" + condMap.get(item) + System.lineSeparator();
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

        String parentName;

        public MyRecordReader(CombineFileSplit split, TaskAttemptContext context, Integer index) throws IOException {
            super(split, context, index);
            parentName = split.getPath(index).getParent().getName();
        }

        @Override
        public Text getCurrentValue() throws IOException, InterruptedException {
            // map输出中间结果格式：className    item
            String record = parentName + "\t" + super.getCurrentValue().toString();
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
        protected void reduce(Text key, Iterable<IntWritable> values, Context context)
                throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable val : values) {
                sum += val.get();
            }
            count.set(sum);
            // map输出中间结果格式：className    item
            String[] split = key.toString().split("\t");
            String className = split[0];
            String item = split[1];
            // className相同的record输出到同一个reduce处理
            multipleOutputs.write(new Text(item), count, className);

            // 将不同的词条添加到字典集合
            if (!dictSet.contains(item)) {
                dictSet.add(item);
            }
            // 根据className统计每个分类的词条总数
            if (!classSize.containsKey(className)) {
                classSize.put(className, sum);
            } else {
                classSize.replace(className, sum + classSize.get(className));
            }
            // 根据className和item更新每个分类的每个词条数
            if (!classItemCount.containsKey(className)) {
                HashMap<String, Integer> map = new HashMap<>();
                map.put(item, sum);
                classItemCount.put(className, map);
            } else {
                HashMap<String, Integer> map = classItemCount.get(className);
                map.put(item, sum);
                classItemCount.replace(className, map);
            }
        }

        @Override
        protected void cleanup(Context context) throws IOException, InterruptedException {
            multipleOutputs.close();
        }
    }

}
