# This is the python script for the image processing module.


"""
# Import python packages.

Often you need to import packages to do fancy works.
In this tutorial, the tool (Replit) takes care of the package installation for you.
But, in the future you may find yourself having the need to install packages.
In that situation, you can use a package manager, such as pip (https://github.com/pypa/pip).
"""
import json
import requests


"""
# Specify the API URL and key of the Hugging Face API.

In this tutorial, we are going to use the Hugging Face API.
API means the Application Programming Interface, which allows computer programs to talk to each other.
API_URL is the Hugging Face API URL that points to a model that we want to use.
For more information about the model, see the following page:
- https://huggingface.co/google/vit-base-patch16-224

API_KEY is the Hugging Face API key for authentication (only used for this course).
Please do not make the API key public.
For more information about how to use the API, see the following page:
- https://api-inference.huggingface.co/docs/python/html/quicktour.html
"""
API_URL = "https://api-inference.huggingface.co/models/google/vit-base-patch16-224"
API_KEY = "[PLACE_HOLDER]"


# Below is a reusable function for interacting with the Hugging Face API.
def query(file_path, api_url, api_key):
    """
    Ask the Hugging Face API to run the model and return the result.

    Attributes
    ----------
    file_path : str
        The path to the image file that we want to send to the Hugging Face API.
    api_url : str
        The API URL that points to a specific machine learning model.
    api_key : str
        The API key for authentication.
    """
    # Construct the header of the HTTP request that includes the API key.
    headers = {"Authorization": f"Bearer " + api_key}
    # Read the input data.
    with open(file_path, "rb") as f:
        data = f.read()
    # Make a POST request to the API with the key and input data.
    response = requests.request("POST", api_url, headers=headers, data=data)
    # Return the output from the API
    return json.loads(response.content.decode("utf-8"))


"""
# Use the Hugging Face API to ask a model to make predictions.

Now we use the above "query" function to ask the API to predict what is in this image.
You can replace the my_image variable with your own images.
Note that my_image is a path that points to a file.
In this case, "data/000000039769.jpeg" is a relative path.
This means that "000000039769.jpeg" is placed in the "data" folder.
And the "data" folder is placed together with the main.py script in the same folder.
"""
my_image = "data/000000039769.jpeg"
data = query(my_image, API_URL, API_KEY)


"""
# Print the output of the model returned by the API.

The output looks like below:
    [{'score': 0.937, 'label': 'Egyptian cat'}, {'score': 0.038, 'label': 'tabby, tabby cat'}, {'score': 0.014, 'label': 'tiger cat'}, {'score': 0.003, 'label': 'lynx, catamount'}, {'score': 0.001, 'label': 'Siamese cat, Siamese'}]

The output is an array of five dictionaries that represent the top 5 predictions from the model.
Array and dictionary are both data structures.
An array looks like [0, 1, 2, 3], which represents a list of elements (such as numbers).
A dictionary looks like {"key1": "value1", "key2", "value2"}, which represents pairs of keys and values.
In this case, the first element in the array {'score': 0.937, 'label': 'Egyptian cat'} is the first prediction.
It means that the model thinks there are Egyptian cats in the image, with 0.937 probability (which is very high).
"""
print(data)


# Pause before the next part of the tutorial.
input("Press Enter to continue...")
