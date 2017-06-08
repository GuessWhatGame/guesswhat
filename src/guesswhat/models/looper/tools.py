import numpy as np

def get_index(l, index, default=-1):
    try:
        return l.index(index)
    except ValueError:
        return default

def clear_after_stop_dialogue(dialogues, tokenizer):
    stop_indices = []
    final_dialogues = []

    for i, dialogue in enumerate(dialogues):
        stop_dialogue_index = get_index(dialogue.tolist(), tokenizer.stop_dialogue, default=len(dialogue)-1)
        answers_index = [j for j,token in enumerate(dialogue[:stop_dialogue_index+1]) if token in tokenizer.answers]
        if answers_index:
            final_dialogues.append(dialogue[:stop_dialogue_index+1])
            stop_indices.append(stop_dialogue_index)
        else:
            final_dialogues.append([])
            stop_indices.append(0)


    return final_dialogues, stop_indices


def list_to_padded_tokens(dialogues, tokenizer):

    # compute the length of the dialogue
    seq_length = [len(d) for d in dialogues]

    # Get dialogue numpy max size
    batch_size = len(dialogues)
    max_seq_length = max(seq_length)

    # Initialize numpy array
    padded_tokens = np.full((batch_size, max_seq_length), tokenizer.padding_token, dtype=np.int32)

    # fill the padded array with word_id
    for i, (one_path, l) in enumerate(zip(dialogues, seq_length)):
       padded_tokens[i, 0:l] = one_path

    return padded_tokens, seq_length