import math
import time
import os
import logging
import boto3
import json
from botocore.exceptions import ClientError
from email.parser import BytesParser
import string
import sys
import numpy as np
import re

from hashlib import md5

if sys.version_info < (3,):
    maketrans = string.maketrans
else:
    maketrans = str.maketrans

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

""" --- Main handler --- """

def lambda_handler(event, context):
    """
    DO: Extracts content from client and prepares SQS message
    TODO: Update SQS url accordingly
    """
    # # Set up params
    ENDPOINT_NAME = "sms-spam-classifier-mxnet-2021-04-06-02-49-12-273" #replace with your endpoint name.
    try: 
        ENDPOINT_NAME = os.environ["ENDPOINT_NAME"]
    except:
        pass
    runtime= boto3.client('runtime.sagemaker')

    # # # Get email from S3
    email_event = event["Records"][0]["s3"]
    bucket = email_event["bucket"]["name"]
    key = email_event["object"]["key"]

    # Extract email body and details
    s3 = boto3.client('s3')
    response = s3.get_object(
        Bucket=bucket,
        Key=key,
    )
    email = response["Body"].read()
    parser =  BytesParser()
    formatted_email = parser.parsebytes(email)
    print("email is: ", type(formatted_email))
    body = ''
    if formatted_email.is_multipart():
        for payload in formatted_email.get_payload():
            # if payload.is_multipart(): ...
            body += payload.get_payload()
    else:
        body += formatted_email.get_payload()
    
    from_addr =  re.search('\<(.*?)\>', formatted_email['From']).group(0)[1:-1]
    to_addr = formatted_email['To']
    sample_body = body
    #Concatenate sample_body wherever necessary
    if len(body) > 240:
        sample_body = sample_body[:240]
    print("From:", from_addr)
    print("To:", to_addr)
    print("Email body: ", body)

    # Encode message into ingestable format
    vocabulary_length = 9013
    # test_messages = ["FreeMsg: Txt: CALL to No: 86888 & claim your reward of 3 hours talk time to use from your phone now! ubscribe6GBP/ mnth inc 3hrs 16 stop?txtStop"]
    test_messages = [body]
    one_hot_test_messages = one_hot_encode(test_messages, vocabulary_length)
    encoded_test_messages = vectorize_sequences(one_hot_test_messages, vocabulary_length)

    # Invoke endpoint and extract predictions
    response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                   ContentType='application/json',
                                   Body=json.dumps(encoded_test_messages.tolist()))
    response = json.loads(response["Body"].read().decode())
    print("Response: ", response)
    print("Classification:", response["predicted_label"][0][0])
    if response["predicted_label"][0][0] == 1.0:
        classification = "Not Spam"
    else:
        classification = "Spam"
    print("Proba:", str(response["predicted_probability"][0][0]))

    # Send SES Email
    reply = "We received your email sent at {} with the subject {}. \n Here is a 240 character sample of the email body:\n {}. \nThis email was classified as {} with a {}% confidence".format(formatted_email['Date'], formatted_email['Subject'], sample_body, classification, str(response["predicted_probability"][0][0]))
    ses_reply_response = send_SES_email(to_addr, from_addr,reply)

    return ({
        'statusCode': 200,
        'body': {
            "Sagemaker_response": response,
            "SES_reply_response": ses_reply_response
        }
    })

def send_SES_email(sender, recipient, body_text=""):
    # Replace sender@example.com with your "From" address.
    # This address must be verified with Amazon SES.
    SENDER = "Assignment 3 Server <{}>".format(sender)

    # Replace recipient@example.com with a "To" address. If your account 
    # is still in the sandbox, this address must be verified.
    RECIPIENT = recipient
    # RECIPIENT = "pn2363@columbia.edu"

    # If necessary, replace us-west-2 with the AWS Region you're using for Amazon SES.
    AWS_REGION = "us-east-1"

    # The subject line for the email.
    SUBJECT = "Amazon SES Spam Classification Results"

    # The email body for recipients with non-HTML email clients.
    BODY_TEXT = (body_text)
                
    # The HTML body of the email.
    BODY_HTML = """<html>
    <head></head>
    <body>
    <h1>Amazon SES Test (SDK for Python)</h1>
    <p>{}<p
    <p>This email was sent with
        <a href='https://aws.amazon.com/ses/'>Amazon SES</a> using the
        <a href='https://aws.amazon.com/sdk-for-python/'>
        AWS SDK for Python (Boto)</a>.</p>
    </body>
    </html>
            """.format(body_text)            

    # The character encoding for the email.
    CHARSET = "UTF-8"

    # Create a new SES resource and specify a region.
    client = boto3.client('ses',region_name=AWS_REGION)

    # Try to send the email.
    try:
        #Provide the contents of the email.
        response = client.send_email(
            Destination={
                'ToAddresses': [
                    RECIPIENT,
                ],
            },
            Message={
                'Body': {
                    'Html': {
                        'Charset': CHARSET,
                        'Data': BODY_HTML,
                    },
                    'Text': {
                        'Charset': CHARSET,
                        'Data': BODY_TEXT,
                    },
                },
                'Subject': {
                    'Charset': CHARSET,
                    'Data': SUBJECT,
                },
            },
            Source=SENDER,
        )
    # Display an error if something goes wrong.	
    except ClientError as e:
        print(e.response['Error']['Message'])
    else:
        print("Email sent! Message ID:"),
        print(response['MessageId'])
    return response
    
def vectorize_sequences(sequences, vocabulary_length):
    results = np.zeros((len(sequences), vocabulary_length))
    for i, sequence in enumerate(sequences):
       results[i, sequence] = 1. 
    return results

def one_hot_encode(messages, vocabulary_length):
    data = []
    for msg in messages:
        temp = one_hot(msg, vocabulary_length)
        data.append(temp)
    return data

def text_to_word_sequence(text,
                          filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                          lower=True, split=" "):
    """Converts a text to a sequence of words (or tokens).
    # Arguments
        text: Input text (string).
        filters: list (or concatenation) of characters to filter out, such as
            punctuation. Default: `!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n`,
            includes basic punctuation, tabs, and newlines.
        lower: boolean. Whether to convert the input to lowercase.
        split: str. Separator for word splitting.
    # Returns
        A list of words (or tokens).
    """
    if lower:
        text = text.lower()

    if sys.version_info < (3,):
        if isinstance(text, unicode):
            translate_map = dict((ord(c), unicode(split)) for c in filters)
            text = text.translate(translate_map)
        elif len(split) == 1:
            translate_map = maketrans(filters, split * len(filters))
            text = text.translate(translate_map)
        else:
            for c in filters:
                text = text.replace(c, split)
    else:
        translate_dict = dict((c, split) for c in filters)
        translate_map = maketrans(translate_dict)
        text = text.translate(translate_map)

    seq = text.split(split)
    return [i for i in seq if i]

def one_hot(text, n,
            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
            lower=True,
            split=' '):
    """One-hot encodes a text into a list of word indexes of size n.
    This is a wrapper to the `hashing_trick` function using `hash` as the
    hashing function; unicity of word to index mapping non-guaranteed.
    # Arguments
        text: Input text (string).
        n: int. Size of vocabulary.
        filters: list (or concatenation) of characters to filter out, such as
            punctuation. Default: `!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n`,
            includes basic punctuation, tabs, and newlines.
        lower: boolean. Whether to set the text to lowercase.
        split: str. Separator for word splitting.
    # Returns
        List of integers in [1, n]. Each integer encodes a word
        (unicity non-guaranteed).
    """
    return hashing_trick(text, n,
                         hash_function='md5',
                         filters=filters,
                         lower=lower,
                         split=split)


def hashing_trick(text, n,
                  hash_function=None,
                  filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                  lower=True,
                  split=' '):
    """Converts a text to a sequence of indexes in a fixed-size hashing space.
    # Arguments
        text: Input text (string).
        n: Dimension of the hashing space.
        hash_function: defaults to python `hash` function, can be 'md5' or
            any function that takes in input a string and returns a int.
            Note that 'hash' is not a stable hashing function, so
            it is not consistent across different runs, while 'md5'
            is a stable hashing function.
        filters: list (or concatenation) of characters to filter out, such as
            punctuation. Default: `!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n`,
            includes basic punctuation, tabs, and newlines.
        lower: boolean. Whether to set the text to lowercase.
        split: str. Separator for word splitting.
    # Returns
        A list of integer word indices (unicity non-guaranteed).
    `0` is a reserved index that won't be assigned to any word.
    Two or more words may be assigned to the same index, due to possible
    collisions by the hashing function.
    The [probability](
        https://en.wikipedia.org/wiki/Birthday_problem#Probability_table)
    of a collision is in relation to the dimension of the hashing space and
    the number of distinct objects.
    """
    if hash_function is None:
        hash_function = hash
    elif hash_function == 'md5':
        hash_function = lambda w: int(md5(w.encode()).hexdigest(), 16)

    seq = text_to_word_sequence(text,
                                filters=filters,
                                lower=lower,
                                split=split)
    return [int(hash_function(w) % (n - 1) + 1) for w in seq]

# test = lambda_handler(None,None)
# response = send_SES_email("test@coms6998cloudcomputinga3.com", "phananh1096@yahoo.com.vn", "this is a test")
# print(test)
# print("Works!")
