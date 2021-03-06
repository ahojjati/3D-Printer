import java.io.*;
import java.util.*;

class svm_train {
	private svm_parameter param;		// set by parse_command_line
	private svm_problem prob;		// set by read_problem
	private svm_model model;
	private String input_file_name;		// set by parse_command_line
	private String model_file_name;		// set by parse_command_line
	private int cross_validation = 0;
	private int nr_fold;

	private static void exit_with_help()
	{
		System.out.print(
		 "Usage: svm-train [options] training_set_file [model_file]\n"
		+"options:\n"
		+"-s svm_type : set type of SVM (default 0)\n"
		+"	0 -- C-SVC\n"
		+"	1 -- nu-SVC\n"
		+"	2 -- one-class SVM\n"
		+"	3 -- epsilon-SVR\n"
		+"	4 -- nu-SVR\n"
		+"-t kernel_type : set type of kernel function (default 2)\n"
		+"	0 -- linear: u'*v\n"
		+"	1 -- polynomial: (gamma*u'*v + coef0)^degree\n"
		+"	2 -- radial basis function: exp(-gamma*|u-v|^2)\n"
		+"	3 -- sigmoid: tanh(gamma*u'*v + coef0)\n"
		+"-d degree : set degree in kernel function (default 3)\n"
		+"-g gamma : set gamma in kernel function (default 1/k)\n"
		+"-r coef0 : set coef0 in kernel function (default 0)\n"
		+"-c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)\n"
		+"-n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)\n"
		+"-p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)\n"
		+"-m cachesize : set cache memory size in MB (default 40)\n"
		+"-e epsilon : set tolerance of termination criterion (default 0.001)\n"
		+"-h shrinking: whether to use the shrinking heuristics, 0 or 1 (default 1)\n"
		+"-wi weight: set the parameter C of class i to weight*C, for C-SVC (default 1)\n"
		+"-v n: n-fold cross validation mode\n"
		);
		System.exit(1);
	}

	private void do_cross_validation()
	{
		int i;
		int total_correct = 0;
		double total_error = 0;
		double sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;

		// random shuffle
		for(i=0;i<prob.l;i++)
		{
			int j = (int)(Math.random()*(prob.l-i));
			svm_node[] tx;
			double ty;

			tx = prob.x[i];
			prob.x[i] = prob.x[j];
			prob.x[j] = tx;

			ty = prob.y[i];
			prob.y[i] = prob.y[j];
			prob.y[j] = ty;
		}

		for(i=0;i<nr_fold;i++)
		{
			int begin = i*prob.l/nr_fold;
			int end = (i+1)*prob.l/nr_fold;
			int j,k;
			svm_problem subprob = new svm_problem();

			subprob.l = prob.l-(end-begin);
			subprob.x = new svm_node[subprob.l][];
			subprob.y = new double[subprob.l];

			k=0;
			for(j=0;j<begin;j++)
			{
				subprob.x[k] = prob.x[j];
				subprob.y[k] = prob.y[j];
				++k;
			}
			for(j=end;j<prob.l;j++)
			{
				subprob.x[k] = prob.x[j];
				subprob.y[k] = prob.y[j];
				++k;
			}

			if(param.svm_type == svm_parameter.EPSILON_SVR ||
			   param.svm_type == svm_parameter.NU_SVR)
			{
				svm_model submodel = svm.svm_train(subprob,param);
				double error = 0;
				for(j=begin;j<end;j++)
				{
					double v = svm.svm_predict(submodel,prob.x[j]);
					double y = prob.y[j];
					error += (v-y)*(v-y);
					sumv += v;
					sumy += y;
					sumvv += v*v;
					sumyy += y*y;
					sumvy += v*y;
				}
				System.out.print("Mean squared error = "+error/(end-begin)+"\n");
				total_error += error;			
			}
			else
			{
				svm_model submodel = svm.svm_train(subprob,param);
				int correct = 0;
				for(j=begin;j<end;j++)
				{
					double v = svm.svm_predict(submodel,prob.x[j]);
					if(v == prob.y[j])
						++correct;
				}
				System.out.print("Accuracy = "+100.0*correct/(end-begin)+"% ("+correct+"/"+(end-begin)+")\n");
				total_correct += correct;
			}
		}		
		if(param.svm_type == svm_parameter.EPSILON_SVR || param.svm_type == svm_parameter.NU_SVR)
		{
			System.out.print("Cross Validation Mean squared error = "+total_error/prob.l+"\n");
			System.out.print("Cross Validation Squared correlation coefficient = "+
				((prob.l*sumvy-sumv*sumy)*(prob.l*sumvy-sumv*sumy))/
				((prob.l*sumvv-sumv*sumv)*(prob.l*sumyy-sumy*sumy))+"\n"
				);
		}
		else
			System.out.print("Cross Validation Accuracy = "+100.0*total_correct/prob.l+"%\n");
	}
	
	private void run(String argv[]) throws IOException
	{
		parse_command_line(argv);
		read_problem();
		if(cross_validation != 0)
		{
			do_cross_validation();
		}
		else
		{
			model = svm.svm_train(prob,param);
			svm.svm_save_model(model_file_name,model);
		}
	}

	public static void main(String argv[]) throws IOException
	{
		svm_train t = new svm_train();
		t.run(argv);
	}

	private static double atof(String s)
	{
		return Double.valueOf(s).doubleValue();
	}

	private static int atoi(String s)
	{
		return Integer.parseInt(s);
	}

	private void parse_command_line(String argv[])
	{
		int i;

		param = new svm_parameter();
		// default values
		param.svm_type = svm_parameter.C_SVC;
		param.kernel_type = svm_parameter.RBF;
		param.degree = 3;
		param.gamma = 0;	// 1/k
		param.coef0 = 0;
		param.nu = 0.5;
		param.cache_size = 40;
		param.C = 1;
		param.eps = 1e-3;
		param.p = 0.1;
		param.shrinking = 1;
		param.nr_weight = 0;
		param.weight_label = new int[0];
		param.weight = new double[0];

		// parse options
		for(i=0;i<argv.length;i++)
		{
			if(argv[i].charAt(0) != '-') break;
			++i;
			switch(argv[i-1].charAt(1))
			{
				case 's':
					param.svm_type = atoi(argv[i]);
					break;
				case 't':
					param.kernel_type = atoi(argv[i]);
					break;
				case 'd':
					param.degree = atof(argv[i]);
					break;
				case 'g':
					param.gamma = atof(argv[i]);
					break;
				case 'r':
					param.coef0 = atof(argv[i]);
					break;
				case 'n':
					param.nu = atof(argv[i]);
					break;
				case 'm':
					param.cache_size = atof(argv[i]);
					break;
				case 'c':
					param.C = atof(argv[i]);
					break;
				case 'e':
					param.eps = atof(argv[i]);
					break;
				case 'p':
					param.p = atof(argv[i]);
					break;
				case 'h':
					param.shrinking = atoi(argv[i]);
					break;
				case 'w':
					++param.nr_weight;
					{
						int[] old = param.weight_label;
						param.weight_label = new int[param.nr_weight];
						System.arraycopy(old,0,param.weight_label,0,param.nr_weight-1);
					}

					{
						double[] old = param.weight;
						param.weight = new double[param.nr_weight];
						System.arraycopy(old,0,param.weight,0,param.nr_weight-1);
					}

					param.weight_label[param.nr_weight-1] = atoi(argv[i-1].substring(2));
					param.weight[param.nr_weight-1] = atof(argv[i]);
					break;
				case 'v':
					cross_validation = 1;
					nr_fold = atoi(argv[i]);
					if(nr_fold < 2)
					{
						System.err.print("n-fold cross validation: n must >= 2\n");
						exit_with_help();
					}
					break;
				default:
					System.err.print("unknown option\n");
					exit_with_help();
			}
		}

		// determine filenames

		if(i>=argv.length)
			exit_with_help();

		input_file_name = argv[i];

		if(i<argv.length-1)
			model_file_name = argv[i+1];
		else
		{
			int p = argv[i].lastIndexOf('/');
			++p;	// whew...
			model_file_name = argv[i].substring(p)+".model";
		}
	}

	// read in a problem (in svmlight format)

	private void read_problem() throws IOException
	{
		BufferedReader fp = new BufferedReader(new FileReader(input_file_name));
		Vector vy = new Vector();
		Vector vx = new Vector();
		int max_index = 0;
		
		while(true)
		{
			String line = fp.readLine();
			if(line == null) break;

			StringTokenizer st = new StringTokenizer(line," \t\n\r\f:");

			vy.addElement(st.nextToken());
			int m = st.countTokens()/2;
			svm_node[] x = new svm_node[m];
			for(int j=0;j<m;j++)
			{
				x[j] = new svm_node();
				x[j].index = atoi(st.nextToken());
				x[j].value = atof(st.nextToken());
			}
			if(m>0) max_index = Math.max(max_index, x[m-1].index);
			vx.addElement(x);
		}

		prob = new svm_problem();
		prob.l = vy.size();
		prob.x = new svm_node[prob.l][];
		for(int i=0;i<prob.l;i++)
			prob.x[i] = (svm_node[])vx.elementAt(i);
		prob.y = new double[prob.l];
		for(int i=0;i<prob.l;i++)
			prob.y[i] = atof((String)vy.elementAt(i));

		if(param.gamma == 0)
			param.gamma = 1.0/max_index;

		fp.close();
	}
}
