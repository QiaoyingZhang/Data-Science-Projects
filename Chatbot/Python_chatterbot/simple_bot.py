# learn chatbot using Python chatterbot library
# import libraries
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer

train_corpus = []

# clean dialogs
with open('dialogs.txt','r') as f:
    # the question follows the response from previous conversation
    # avoid the question starting from the second line
    first_line = f.readline().split('\t')
    
    train_corpus.append(first_line[0])
    train_corpus.append(first_line[1])
    
    for line in f:
        line  =  line.split('\t')
        train_corpus.append(line[1]) 

# train chatterbot with cleaned corpus
chatbot = ChatBot('Daily Chat')
trainer = ListTrainer(chatbot)
trainer.train(train_corpus)

exit_conditions = (':q', 'quit', 'exit')
while True:
    query = input('> ')
    
    if query in exit_conditions:
        break
    else:
        print(f'{chatbot.get_response(query)}')