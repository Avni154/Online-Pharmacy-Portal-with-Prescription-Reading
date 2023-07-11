The model has been trained for a total of 150 epochs
# Second conv block.
x = keras.layers.Conv2D(
  64,
  (3, 3),
  activation="relu", 
  kernel_initializer="he_normal", 
  padding="same", 
  name="Conv2",
)(x)
x = keras.layers.MaxPooling2D((2, 2), name="pool2")(x)

# We have used two max pool with pool size and strides 2.
# Hence, downsampled feature maps are 4x smaller. The number of # filters in the last layer is 64. Reshape accordingly before
# passing the output to the RNN part of the model.
new_shape = ((image_width // 4), (image_height // 4) * 64)
x = keras.layers.Reshape(target_shape=new_shape,name="reshape")(x)
x = keras.layers.Dense(64, activation="relu", name="dense1")(x) 
x = keras.layers.Dropout(0.2)(x)

# RNNs.
x = keras.layers.Bidirectional(
  keras.layers.LSTM(128, return_sequences=True, dropout=0.25) )(x)
x = keras.layers.Bidirectional(
  keras.layers.LSTM(64, return_sequences=True, dropout=0.25)
)(x)
# +2 is to account for the two special tokens introduced by the CTC loss.
# The recommendation comes here: https://git.io/J0eXP. 
x = keras.layers.Dense(
  len(char_to_num.get_vocabulary()) + 2, activation="softmax", name="dense2"
)(x)
# Add CTC layer for calculating CTC loss at each step. 
output = CTCLayer(name="ctc_loss")(labels, x)
# Define the model.
model = keras.models.Model(
  inputs=[input_img, labels], outputs=output, name="handwriting_recognizer"
)
# Optimizer.
opt = keras.optimizers.Adam() 
# Compile the model and return. 
model.compile(optimizer=opt) 
return model
