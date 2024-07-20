import streamlit_authenticator as stauth
import database as db

usernames = ["rsanchez", 
             "acaruso", 
             "bmaple", 
             "hcjones",
             "kcaradonna"
             ]

names = ["Rob Sanchez",
         "Alex Caruso", 
         "Bob Maple", 
         "Hugo Calvert-Jones",
         "Kate Caradonna"]

passwords = ["xrjf", 
             "abc123", 
             "abc123", 
             "abc123",
             "abc123"]


hashed_passwords = stauth.Hasher(passwords).generate()

for (username, name, hashed_password) in zip(usernames, names, hashed_passwords):
    db.insert_user(username, name, hashed_password)