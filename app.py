import streamlit as st
import sqlite3

# Create a connection object
conn = sqlite3.connect('data.db')

# Create a cursor object
c = conn.cursor()

# Create a table
c.execute('''
    CREATE TABLE IF NOT EXISTS users (
        name TEXT NOT NULL,
        email TEXT,
        feedback TEXT NOT NULL,
        rating INTEGER NOT NULL
    )
''')


def add_feedback(name, email, feedback, rating):
    c.execute('''
        INSERT INTO users (name, email, feedback, rating)
        VALUES (?, ?, ?, ?)
    ''', (name, email, feedback, rating))
    conn.commit()

def feedback_form():
    name = st.text_input('Name')
    email = st.text_input('Email')
    feedback = st.text_area('Feedback')
    rating = st.slider('Rating', 1, 5)
    if st.button('Submit'):
        add_feedback(name, email, feedback, rating)
        st.success('Feedback submitted successfully!')

def main():
    feedback_form()

if __name__ == '__main__':
    main()