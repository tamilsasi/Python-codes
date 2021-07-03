import tensorflow as tf

@tf.function
def pow(x):
  return x ** 2

# Construct a basic model.
root = tf.train.Checkpoint()
root.pow = pow

# Create the concrete function.
input_data = tf.constant(1., shape=[1])
concrete_function = root.pow.get_concrete_function(input_data)

# Save the model.
export_dir = "/a/"
tf.saved_model.save(root, export_dir, concrete_function)
print(export_dir)

#input_data2 = tf.constant(2., shape=[2, 2])
#root.mull.get_concrete_function(input_data2)

# Convert the model.
converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
tflite_model = converter.convert()
open("sav_model.tflite","wb").write(tflite_model)