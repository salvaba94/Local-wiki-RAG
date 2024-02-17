if __name__ == "__main__":
    from raggy.chat import MultiModalChat
    chat = MultiModalChat()
    #print("Salva")
    chat.ingest("./docs/Generalization.pdf")

    #chat.ask("Waht is the average precision as a function of the distance?")