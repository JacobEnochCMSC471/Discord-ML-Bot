import preprocess  # Stores all pre-processing code for user strings
import discord
import os
from transformers import AutoModel, AutoTokenizer
import keras
import numpy as np

# Creating client instance - this will be used to interact with the Discord API (connection to Discord)
# intents = discord.Intents.all()

token = os.getenv("DISCORD_TOKEN")
# Saw the command below in the Discord API Man Pages. Manpages said to use default intents.
intents = discord.Intents.default()
client = discord.Client(intents=intents)
# secret_key = ''

# my_guild = os.getenv("DISCORD_GUILD")


BERT_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
BERT_model = AutoModel.from_pretrained('bert-base-uncased')

# Loads the 2D CNN model saved as an hdf5 file.
moderation_model = keras.models.load_model('80pct_recall_conv2d.h5')

# The on_ready event happens when the bot comes online
@client.event
async def on_ready():
    # Verify that the bot has successfully come online
    print(f"Bot logged in as {client.user}")


# The on_message event happens when a message gets sent on the server
@client.event
async def on_message(message):
    # Only classify the message if it is not sent by the bot
    if message.author != client.user:

        user_string = message.content

        model_list = []

        preprocessed_string = preprocess.preprocess_user_comment(user_string)
        trimmed_preprocessed = preprocess.trim_sent_length(preprocessed_string)
        tokenized_comment = preprocess.tokenize_user_comment(trimmed_preprocessed, BERT_tokenizer)
        dirty_comment_embedding = preprocess.generate_comment_embedding(tokenized_comment, BERT_model)
        cleaned_comment_embedding = preprocess.extract_embedding_values(dirty_comment_embedding)
        reshaped_embedding = np.reshape(cleaned_comment_embedding, (24, 32))

        model_list.append(reshaped_embedding)

        model_list = np.array(model_list)

        model_prediction = moderation_model.predict(model_list)[0][0]

        # print(model_prediction)
        print(model_prediction)
        # Code to send messages. We'll be using this function to have our bot send a message flagging somehting as toxic.
        await message.channel.send(model_prediction)


# client.run(secret_key)  # Running the bot
client.run(token)