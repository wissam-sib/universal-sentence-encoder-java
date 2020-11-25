package useRepresentation;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Tensor;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.Arrays;

import javax.xml.crypto.Data;

import java.io.UnsupportedEncodingException;
import org.tensorflow.ConcreteFunction;
import org.tensorflow.Signature;
import org.tensorflow.TensorFlow;
import org.tensorflow.ndarray.FloatNdArray;
import org.tensorflow.ndarray.NdArraySequence;
import org.tensorflow.ndarray.NdArrays;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Constant;
import org.tensorflow.op.core.Placeholder;
import org.tensorflow.op.math.Add;
import org.tensorflow.types.TInt32;
import org.tensorflow.types.TString;
import org.tensorflow.types.TFloat32;


public class UseRepresentation {

	public String modelPath;
	SavedModelBundle savedModelBundle;
	
	

	public UseRepresentation(String modelPath) {
		super();
		this.modelPath = modelPath;
		this.savedModelBundle = SavedModelBundle.load(modelPath, "serve");
	}
	
	

	public float[][] embed(String[] values) throws UnsupportedEncodingException {

		byte[][] input = new byte[values.length][];
		for (int i = 0; i < values.length; i++) {
			String val = values[i];
			input[i] = val.getBytes(StandardCharsets.UTF_8);

		}	
		
		Tensor<TString> t = TString.tensorOfBytes(NdArrays.vectorOfObjects(input));

		Tensor<TFloat32> result = this.savedModelBundle.session().runner().feed("input", t).fetch("output").run().get(0).expect(TFloat32.DTYPE);

		float[][] output = new float[values.length][512];
		
		TFloat32 output_raw = result.data();

		long[] idx = new long[2];
		
		for(int i = 0;i<output.length;i++) {
			for(int j = 0;j<output[0].length;j++) {
				idx[0] = i;
				idx[1] = j;
				output[i][j] = result.data().getFloat(idx);
			}
		}
		
		return output;
	}
	
	

	public static void main(String[] args) {

		
		UseRepresentation model = new UseRepresentation("C:\\use_java\\universal-sentence-encoder-4-java");
		String[] myStringArray = new String[] { "Hello World", "I am going to be converted to an embedding", "For various NLP tasks" };
		
		try {
			float[][] vectors = model.embed(myStringArray);
			System.out.println(Arrays.deepToString(vectors).replace("], ", "]\n"));
		} catch (UnsupportedEncodingException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}
}