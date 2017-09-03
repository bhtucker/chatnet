"""
"""


def assemble_messages(row, message_generator, value_filter=lambda v: True):
    sequence = []
    for message in message_generator(row):
        if isinstance(message, (unicode, str)) and value_filter(message):
            sequence += message.split()
        else:
            return sequence
    return sequence


def get_wide_columns_message_generator(message_cols):
    # Returns a message generator
    # given a list of column names where messages can be found
    def message_generator(row):
        for key in message_cols:
            yield row[key]
    return message_generator


def get_dense_column_message_generator(column):
    # Returns a function that gets a message generator from a row
    # given a column name which points to an array of messages

    def message_generator(row):
        for cell in row[column]:
            yield cell
    return message_generator
