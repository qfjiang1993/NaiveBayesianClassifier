import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.hbase.util.Bytes;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.util.LineReader;

import java.io.IOException;
import java.net.URI;
import java.text.DecimalFormat;
import java.util.HashSet;

/**
 * @author QFJiang 2016/11/11
 *
 * 展示预测结果，计算并保存评估参数 P、R、F1
 */
public class Result {

    public static void main(String args[]) throws IOException {

        if (args.length < 3) {
            System.err.println("Usage : Result <hdfs> <train> <result>");
            System.exit(2);
        }

        Configuration conf = new Configuration();
        conf.set("mapreduce.input.fileinputformat.input.dir.recursive", "true");
        FileSystem fs = FileSystem.get(URI.create(args[0]), conf);

        HashSet<String> classSet = new HashSet<>();
        FSDataInputStream input;
        FSDataOutputStream output;
        LineReader reader;
        Text line = new Text();

        // 读取分类信息
        input = fs.open(new Path(args[1], "prior.txt"));
        reader = new LineReader(input);
        while (reader.readLine(line) > 0) {
            classSet.add(line.toString().split("\t")[0]);
        }
        Object[] classArray = classSet.toArray();
        int[][] count = new int[classSet.size()][4];

        // 读取预测结果，预测结果每一行的记录格式如下：
        // docName  realClass  predClass  class1:p1  class2:p2  ···
        input = fs.open(new Path(args[2], "prediction.txt"));
        reader = new LineReader(input);
        while (reader.readLine(line) > 0) {
            String[] split = line.toString().split("\t");
            String real = split[1];
            String pred = split[2];
            for (int i = 0; i < classArray.length; i++) {
                String className = classArray[i].toString();
                if (real.equals(className) && pred.equals(className)) {
                    count[i][0]++;
                } else if (!real.equals(className) && pred.equals(className)) {
                    count[i][1]++;
                } else if (real.equals(className) && !pred.equals(className)) {
                    count[i][2]++;
                } else {
                    count[i][3]++;
                }
            }
        }

        output = fs.create(new Path(args[2], "result.txt"));
        System.out.println("Evaluation Result stored in " + args[2] + "/result.txt" + "\n");

        // 输出每个分类的邻接矩阵、及联合邻接矩阵
        int[] total = new int[]{0, 0, 0, 0};
        double pAll = 0;
        double rAll = 0;
        double fAll = 0;
        for (int i = 0; i < classSet.size(); i++) {
            double p = (double) count[i][0] / (count[i][0] + count[i][1]);
            double r = (double) count[i][0] / (count[i][0] + count[i][2]);
            double f1 = (double) 2 * count[i][0] / (2 * count[i][0] + count[i][1] + count[i][2]);
            total[0] += count[i][0];
            total[1] += count[i][1];
            total[2] += count[i][2];
            total[3] += count[i][3];
            pAll += p;
            rAll += r;
            fAll += f1;
            println(classArray[i], output);
            println("TP:" + count[i][0] + "\t\tFP:" + count[i][1], output);
            println("FN:" + count[i][2] + "\t\tTN:" + count[i][3], output);
            println("P:" + dbl4(p) + "\t" + "R:" + dbl4(r) + "\t" + "F1:" + dbl4(f1), output);
            println("", output);
        }

        // 计算微平均与宏平均
        println("Micro Avg", output);
        println("P:" + dbl4(pAll / classArray.length) + "\t"
                + "R:" + dbl4(rAll / classArray.length) + "\t"
                + "F1:" + dbl4(fAll / classArray.length), output);
        println("", output);
        println("Macro Avg", output);
        double ap = (double) total[0] / (total[0] + total[1]);
        double ar = (double) total[0] / (total[0] + total[1]);
        double af = (double) 2 * total[0] / (2 * total[0] + total[1] + total[2]);
        println("TP:" + total[0] + "\t\tFP:" + total[1], output);
        println("FN:" + total[2] + "\t\tTN:" + total[3], output);
        println("P:" + dbl4(ap) + "\t" + "R:" + dbl4(ar) + "\t" + "F1:" + dbl4(af), output);

        input.close();
        output.close();
        reader.close();
    }

    // 输出四位小数格式化数据
    private static String dbl4(double d) {
        return new DecimalFormat("0.0000").format(d);
    }

    // 打印数据到控制台，及输出流
    private static void println(Object o, FSDataOutputStream out) throws IOException {
        System.out.println(o);
        out.write(Bytes.toBytes(o.toString() + System.lineSeparator()));
    }
}
