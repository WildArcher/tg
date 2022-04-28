from telethon import TelegramClient, events, sync

import configparser
import json
import asyncio
from datetime import date, datetime

from telethon import TelegramClient
from telethon.errors import SessionPasswordNeededError
from telethon.tl.functions.messages import (GetHistoryRequest)
from telethon.tl.types import PeerChannel
import streamlit 


start_date = streamlit.date_input(
     "start date")


config = configparser.ConfigParser()
config.read("config.ini")

# Setting configuration values
api_id = config['Telegram']['api_id']
api_hash = config['Telegram']['api_hash']

api_hash = str(api_hash)

phone = config['Telegram']['phone']
username = config['Telegram']['username']

channel_name = streamlit.text_input('print channel_name')

channel_name = input('print channel_name')

streamlit.write('OK')

@client.on(events.NewMessage(chats=f'{channel_name}'))
async def my_event_handler(event):
    streamlit.write(event.raw_text)

client.start()
client.run_until_disconnected()
