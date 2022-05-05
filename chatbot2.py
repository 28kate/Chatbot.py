import sqlite3
from tqdm import tqdm
import random
import re
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from string import punctuation
from tts import speak


NAME = "bot626"
PUNCT = list(punctuation)


con = sqlite3.connect('./{}.db'.format(NAME))
print("Connected to {} database...".format(NAME))
c = con.cursor()


# tables needed to store the data set/sets...
def create_table():
    tables = [
    "CREATE TABLE pairs(parent TEXT NOT NULL, reply TEXT NOT NULL, word_id INT NOT NULL, word_mass REAL NOT NULL, instance INT NOT NULL DEFAULT 1)",
    "CREATE TABLE words(word TEXT NOT NULL)"
    ]

    try:
        for i in tables:
            c.execute(i)
    except:
        pass


def get_id(word):
    c.execute("SELECT rowid from words WHERE word = ?", (word,))
    rowid = c.fetchone()

    if rowid:
        return rowid[0]
    else:
        c.execute("INSERT INTO words(word) VALUES (?)", (word,))
        con.commit()
        #latest insert of word will be at end, hence lastrowid
        return c.lastrowid                                                          


#function used to determine a word's holding mass in a sentence e.g 1 word/8 words
def get_mass(word_list, word):
    total_no_of_words = len(word_list)

    instan = 0
    for w in word_list:
        if word == w:
            instan += 1

    mass = float(instan)/float(total_no_of_words)
    return mass

#cleans text to make it easier for the machine to understand and train the model....the function below was taken from python environment analytics.
def clean_text(text):
   

    text = text.lower()
    
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "that is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)

    return text
    

#function used to send the split data into the database
def train(parent, reply):
    
    raw_parent_words = word_tokenize(clean_text(parent))

    
    parent_words = [p for p in raw_parent_words if p not in PUNCT]

 
    parent_id = []
    parent_weight = []

    for word in parent_words:
        word_id = get_id(word)    
        parent_id.append(word_id)
        word_weight = get_mass(parent_words, word)
        parent_weight.append(word_weight)

    # will search for the existing data in the database
    c.execute("SELECT instance, rowid FROM pairs WHERE parent = ? AND reply = ?", (parent, reply,))
    row_info = c.fetchone()
    #increases the number of instances for that info if it already exists
    if row_info:
        new_instan = int(row_info[0]) + 1

        c.execute("UPDATE pairs SET (instance) = (?) WHERE rowid = ?",
            (str(new_instan), str(row_info[1]),))
    #will insert new data in if it does not already exist in the database
    else:
        c.execute("INSERT INTO pairs(parent, reply, word_id, word_mass) VALUES (?, ?, ?, ?)",
            (parent, reply, str(parent_id), str(parent_weight),))

    con.commit()



#splits the data into individual comments and their corresponding responses,
def prep_reddit_data():
    raw_data = open("./botdata/reddit_convos.txt", 'r', encoding='utf-8').read().split("\n")

    reddit_parent = raw_data[0::2]
    reddit_reply = raw_data[1::2]
    
    make_new_file = open("./botdata/reddit_parent.txt", 'w', encoding='utf-8').write("\n".join(reddit_parent))
    make_new_file = open("./botdata/reddit_reply.txt", 'w', encoding='utf-8').write("\n".join(reddit_reply))
    

def train_redditCorpus():
    parent_file = open("./botdata/reddit_parent.txt", 'r', encoding='utf-8').read().split("\n")
    reply_file = open("./botdata/reddit_reply.txt", 'r', encoding='utf-8').read().split("\n")

    try:
        last_reddits = open('reddit_progress.txt', 'r').read()
        last_reddits = int(last_reddits)
    except:
        last_reddits = 0

    try:
        if len(parent_file) == len(reply_file):
            print("Training on comments...")
            print("Starting from:", last_reddits)

            for i in tqdm(range(last_reddits, len(parent_file))):
                train(parent_file[i], reply_file[i])
                last_reddits = i

    except Exception as e:
        print("Lattest training progress:", last_reddits)
        final_last_reddits = open("reddit_progress.txt", 'w').write(str(last_reddits))
        con.commit()
        print(e)


def get_response(sentence):
    try:
        raw_sentence = sentence.lower()
        words = word_tokenize(raw_sentence)

        filtered_words = [f for f in words if f not in PUNCT]
        word_id = []

        for w in filtered_words:
            w_id = get_id(w)
            word_id.append(w_id)


        c.execute("SELECT * FROM pairs")
        db_data = c.fetchall()
        # FORMAT of raw_data
        # [parent, reply, word_id, word_weight, occurence]

        selected_rows = []
        for r in db_data:
            raw_word_id = eval(r[2])

            for i in word_id:
                if i in raw_word_id:
                    selected_rows.append(r)
                    break

        selected_row_weight = []
        for s in selected_rows:
            weight = 0
            used_word_id = []
            raw_word_id = eval(s[2])
            raw_word_weight = eval(s[3])

            for i in range(len(raw_word_id)):
                if raw_word_id[i] in word_id and raw_word_id[i] not in used_word_id:
                    weight += raw_word_weight[i]
                    used_word_id.append(raw_word_id[i])

            selected_row_weight.append(weight)

        favored_weight = 1.0
        higest_weight = min(selected_row_weight, key=lambda x:abs(x-favored_weight))
        best_rows = []

        for i in range(len(selected_row_weight)):
            if selected_row_weight[i] == higest_weight:
                best_rows.append(selected_rows[i][1])

        best_rows = [best_rows, higest_weight*100]

        return best_rows

    # response for the least occurance
    except:
        least_occur = []
        all_occur = 0

        for r in db_data:
            all_occur += r[4]

        average_occur = int(all_occur/len(db_data))

        for r in db_data:
            if r[4] <= average_occur:
                least_occur.append(r[1])

        least_occur = [least_occur, 0.0]

        return least_occur


def get_final_reply(sentence):
    sentence = sentence.lower()
    best_replies = get_response(sentence)

    final_reply = random.choice(best_replies[0])
    return [final_reply, best_replies[1]]


def user_interaction():
    try:
        bot_input = ""
        user_input = ""
        while True:
            if user_input == "":
                user_input = input("[USER]: ")
                speak(user_input)
            else:
                bot_input = get_final_reply(user_input)
                print("[{}]:".format(NAME), bot_input[0], )
                speak(bot_input[0])
                user_input = input("[You]: ")
                speak(user_input)
                train(bot_input[0], user_input)

    except Exception as e:
        print("\n[ERRORS]:\n")
        print(str(e))
        print("Closing the conversation...")
        con.commit()




create_table()
#prep_reddit_data()
#train_redditCorpus()
user_interaction()
