###############################################################
# File: Semi-Automatic Discord Moderation Bot
# Author: Austin John, Jacob Enoch
# Date: 12/20/2022
# Description: Leverages the Discord.py API to build a Discord bot that reads in messages and labels messages 
#              as toxic or not with the help of a CNN Model we implemented with a 94% accuracy. 
#              There are muliple other models that we've implemented which can be found under the models folder on our github site.

###############################################################

import preprocess  # Stores all pre-processing code for user strings
import discord
from discord.ext import commands
import os

from better_profanity import profanity  # Library containing list of profane words
from dotenv import load_dotenv # Library used to read in .env files
from transformers import AutoModel, AutoTokenizer  
import keras
import numpy as np

# Declare channel ID's as constants
GENERALCHANNEL = 1051642140729540710
MODERATIONCHANNEL = 1051683399720505385

# Uses the load_dotenv function to load .env file where our Private Discord Token is stored.
load_dotenv()
token = os.getenv("DISCORD_TOKEN")

# Set discord Intents and create client instance - this will be used to interact with the Discord API (connection to Discord)
intents = discord.Intents.all()
client = discord.Client(intents=intents)

# Used to set a predefined guild / server for the bot
# my_guild = os.getenv("DISCORD_GUILD")

# Loads a text file of profane words and their misspellings that people use to circumvent moderation bots
profanity.load_censor_words()

BERT_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
BERT_model = AutoModel.from_pretrained('bert-base-uncased')

# Loads the 2D CNN model saved as an hdf5 file.
moderation_model = keras.models.load_model('best_2d_cnn.h5')

# The on_ready event happens when the bot comes online
@client.event
async def on_ready():
    # Verify that the bot has successfully come online
    print(f"Bot logged in as {client.user}")


@client.event
async def on_member_join(member):
    # Get the channel ID for the General Channel
    channel = client.get_channel(GENERALCHANNEL)
    embed = discord.Embed(title=f"Welcome {member.name}", description=f"There are no rules because I am the allseer and will squash your hopes and dreams of being toxic.")
    # Send a message in the channel as an embedded message with the afforementioned message.
    await channel.send(embed=embed)

# The on_message event happens when a message gets sent on the server
@client.event
async def on_message(message):
    # Does not scan attachments or links.
    if len(message.attachments) > 0:
        return
    
    if 'http' in message.content:
        return

    if 'gif' in message.content:
        return

    if message.author.id in [client.user.id]:
        return

    # Ignores automated messages by Discord or other bots
    if message.author.bot:
        return

    # Only classify the message if it is sent by a user.
    if message.author != client.user:

        # Message content is saved to user_string.
        user_string = message.content

        model_list = []

        # The user message is then preprocessed.
        preprocessed_string = preprocess.preprocess_user_comment(user_string)
        trimmed_preprocessed = preprocess.trim_sent_length(preprocessed_string)
        tokenized_comment = preprocess.tokenize_user_comment(trimmed_preprocessed, BERT_tokenizer)
        dirty_comment_embedding = preprocess.generate_comment_embedding(tokenized_comment, BERT_model)
        cleaned_comment_embedding = preprocess.extract_embedding_values(dirty_comment_embedding)
        # Input has to be reshaped before passing into the CNN Model.
        reshaped_embedding = np.reshape(cleaned_comment_embedding, (24, 32))

        model_list.append(reshaped_embedding)

        model_list = np.array(model_list)

        model_prediction = moderation_model.predict(model_list)[0][0] * 100


        # Prints the prediction to the console.
        print(model_prediction)

        # Checks if any word in the message is profane - checks against a list of profane words.
        if profanity.contains_profanity(message.content):
            # Deletes the message if it contains profanity.
            await message.delete()
            # await message.channel.send(f'{message.author.mention} said *{profanity.censor(message.content)}*. \nDo not send profane messages in the chat.')
            # Warns a user in the channel where the initial message was sent asking the user to not send profanity.
            await message.channel.send(f'{message.author.mention} Do not send profane messages in the chat.')
            # Sends a message in the Moderation Queue channel as a log.
            await client.get_channel(MODERATIONCHANNEL).send(f'**Deleted** \n`{message.content}` sent by {message.author.mention} in {message.channel.mention}. \n**Reason**: Profanity')
        
        # If there are no profane words in the list, then the bot will use the model to check for toxicity.
        # If the prediction is between 50 and 80, then it will send the message in the moderation channel for human moderators to intervene. 
        elif model_prediction > 50 and model_prediction < 80:
            await client.get_channel(MODERATIONCHANNEL).send(f'**Requires Moderation** \nText: `{message.content}` \nLink: {message.channel.jump_url}')


        elif model_prediction >= 80:
            # Deletes the message from the channel if the model flags the message as assuredly toxic.
            await message.delete()
            # Sends a message in the Moderation Queue Channel as a way to keep a log of deleted/flagged messages.
            await client.get_channel(MODERATIONCHANNEL).send(f'**Deleted** \n`{message.content}` sent by {message.author.mention} in {message.channel.mention}. \n**Reason**: Toxicity')       
            # Warns the user asking them to not send toxic messages in the chat.
            await message.channel.send(f'{message.author.mention} Abstain from sending toxic messages in the chat.')

# Run the bot with the Secret Discord Token.
client.run(token)