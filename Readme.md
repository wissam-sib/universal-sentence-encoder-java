# Using Universal Sentence Encoder in Java

Universal Sentence Encoder (USE) is transformer-based model that turns natural language sentences into fixed size float vectors. 

This repository contains a minimal Java project (with Maven to manage tensorflow dependencies) that provides a class (UseRepresentation) to apply USE on English sentences and turn them into a float array of dimension 512.
These vectors can then be used so solve several NLP tasks such as classification, similarity and so on. I adapted solutions from [here](https://github.com/tensorflow/hub/issues/194) to implement this project. 

Tested on Windows 7 and Ubuntu 20.x.

## Step 1 : Prepare the model

Note : You can skip this part and directly download the model [here](https://drive.google.com/file/d/1X6j8keyG0Hhc6CbOlc25gHwrYE_s9_NF/view?usp=sharing) (just unzip the folder ``universal-sentence-encoder-4-java`` at your favorite location)

To prepare the model, I used the tensorflow hub in python to download into a ``KerasLayer``. Then I simply saved it while specifying the inputs and outputs:

```python
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
tf.disable_v2_behavior() 
url = "https://tfhub.dev/google/universal-sentence-encoder/4"
save_path = "path/to/universal-sentence-encoder-4-java/"
with tf.Graph().as_default():
    module = hub.KerasLayer(url)
    model_input = tf.placeholder(tf.string, name="input")
    model_output = tf.identity(module(model_input), name="output")
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        tf.saved_model.simple_save(
            session,
            save_path,
            inputs={'input': model_input},
            outputs={'output': model_output},
            legacy_init_op=tf.initializers.tables_initializer(name='init_all_tables'))
```

I used tensorflow_hub 0.9.0 and tensorflow 2.3.1 

## Step 2 : Import the project in Java and run it

The project uses Maven for dependencies. In the pom.xml, you'll find the Java version (1.8 but can probably work with higher) and a Tensorflow Snapshot (Tensorflow version is 2.3.1).

In UseRepresentation.java, you can edit the main method (specify the path to "universal-sentence-encoder-4-java/") and then run it to test the model :

```java
UseRepresentation model = new UseRepresentation("path/to/universal-sentence-encoder-4-java");
String[] myStringArray = new String[] { "Hello World", "I am going to be converted to an embedding", "For various NLP tasks" };

try {
	float[][] vectors = model.embed(myStringArray);
	System.out.println(Arrays.deepToString(vectors).replace("], ", "]\n"));
} catch (UnsupportedEncodingException e) {
	// TODO Auto-generated catch block
	e.printStackTrace();
}
```

Loading the model takes a few seconds (do it only once in your project) but conversion to vector is rather fast. 
