from transformers import  GPT2TokenizerFast,  TFGPT2LMHeadModel
import tensorflow as tf

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", mask_token='#')
model = TFGPT2LMHeadModel.from_pretrained("gpt2")
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(optimizer=optimizer, loss="passthrough", metrics=[])
