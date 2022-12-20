# CMSC 473 Discord Moderation Bot
## Authors: Jacob Enoch, Austin John

The bot leverages the Discord.py API to build the Discord Bot. We also saved our developer API Discord Token as an environment variable in a `.env` file.

To run the bot on your machine:
* Make a new file named `.env` with just one line in the file.
       `DISCORD_TOKEN= <YOUR API TOKEN>`</li>
* Change the `GENERALCHANNEL=` and `MODERATIONCHANNEL=` constants in `main.py` with the channel ID's for your respective channels in the server.</li>

The program should automatically load the secret API token as an environment variable everytime the bot boots up.

Enter `pip install -r requirements.txt` in your terminal/console to install the necessary dependencies to run the bot.
