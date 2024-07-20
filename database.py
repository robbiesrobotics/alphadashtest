from deta import Deta
import streamlit as st

#Env Variables
DETA_KEY = st.secrets["DETA_USER_AUTH"]


#Initialize Product Key
deta = Deta(DETA_KEY)

#Create/Connect Database
db = deta.Base("UserAuth") 

def insert_user(username, name, password):
    return db.put({"key": username, "name": name, "password": password})

def fetch_all_users():
    # Return a Dictionary of all users
    res = db.fetch()
    return res.items


def get_user(username):
    ## If not found, the function will return None ##
    return db.get(username)


def update_user(username, updates):
    ## If the item is updated, returns none. Otherwise, and exception is raised
    return db.update(updates, username)
    
def delete_user(username):
    ## Always returns none, even if key doesn't exist ##
      return db.delete(username)
    
    
    
#print(fetch_all_users())
