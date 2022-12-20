# CMSC 473 Discord Moderation Bot
## Authors: Jacob Enoch, Austin John

This Discord Bot uses a pre-trained Keras CNN model with 91% accuracy and 79% positive class recall to classify comments that are read in from a channel. 
Messages are checked against a list of profane words and are deleting instantly if profanity is detected.
Messages are then passed through our model with 2 cases implemented:

<ol>
  <li> If the probability is between 50 and 80, the message is placed in a moderation queue awaiting a human moderator's input. </li>
  <li> If the probability is greater than 80, then the message is deleted instantly and a log of the message is sent in the appropriate channel if logs of said messages are ever needed in the future.</li>
</ol>

Our paper `CMSC_473_Discord_Moderation_Bot_John_Enoch.pdf` and the Jupyter Notebook `Jacob Enoch Austin John CMSC473 Project.ipynb` file has been uploaded explaining our experimentation and thought process behind each model we implemented.

Enter `pip install -r requirements.txt` in the terminal/console to install the necessary dependencies to run the notebook.

It should be noted that this notebook was primarily ran and tested using Colab - [link to original Colab notebook can be found here](https://colab.research.google.com/drive/1LFFE5-HH5Lw_gFClVcy7YSQMmaOqlz7D?authuser=1). Simply upload the required files (common_contractions.txt, train/dev/test embeddings CSV files, as well as any trained models that will be loaded in) to the runtime and run the Notebook as normal. All dependencies will be installed for the current Colab session. We did test it on a Linux distribution as well - the Colab link is provided in case issues arise with local machines. We also recommend running the Notebook on Colab because of the high RAM requirements. A standard, free Colab session is able to run data pre-processing and model training/evaluation without any RAM overloading. 

One final note - we recommend not letting the Embeddings section of the model code run fully. Each data set takes approximately 45-50 minutes to generate embeddings for. These were generated and saved in the provided embedding CSV files. The files that will be created when Embeddings section is ran are different from the provided to prevent the provided embeddings from getting over-written or appended to (i.e. new embeddings are written to train_embeddings_NEW.csv instead of the provided train_embeddings.csv).
