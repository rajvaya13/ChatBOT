# ChatBOT

An advanced chatbot powered by cutting-edge Large Language Models, built with Python and LLM (Large Language Model) technology.

## Dependencies (assuming Windows)

You can install the required dependencies by running the following command:

```shell
pip install pylzma numpy ipykernel jupyter torch --index-url https://download.pytorch.org/whl/cu118
```

> **Note**: In case you do not possess an NVIDIA GPU, the `device` parameter will automatically default to `cpu`. This is because `device = 'cuda' if torch.cuda.is_available() else 'cpu'`.. If 'device' defaults to 'cpu', it's perfectly acceptable; you will simply encounter slower execution times.

### Files

1. **data_extract.py** - This script is responsible for extracting data from the provided dataset for computation.

2. **generate_training_model.py** - The code is an implementation of a GPT-2 (Generative Pre-trained Transformer 2) language model using PyTorch. This model is designed for natural language processing tasks, including text generation. Here's a brief description of what the code does:

   - Sets up the training environment with PyTorch, device selection (GPU if available), and model hyperparameters.
   - Reads a vocabulary file ("vocab.txt") to define the character set for the language model.
   - Defines functions to extract random text chunks from training and validation data.
   - Implements self-attention mechanisms and transformer blocks for the language model architecture.
   - Creates a GPT-2 language model that consists of token and position embeddings, multiple transformer blocks, and a linear output layer.
   - Trains the model by estimating loss and updating model weights using an AdamW optimizer.
   - Saves the trained model to a file named "model-01.pkl."
   - Provides a function to generate text using the trained model, starting from a given prompt.

   This code serves as a foundational framework for training and using a GPT-2 language model for text generation tasks. It can be extended and customized for specific applications in natural language processing.

3. **chatBot.py** - The code is an implementation of a GPT-2 (Generative Pre-trained Transformer 2) language model for text generation using PyTorch. This code allows you to interactively input a text prompt, and the model generates a continuation of the text. Here's a brief description of what the code does:

   - Sets up the training environment with PyTorch, device selection (GPU if available), and model hyperparameters.
   - Defines a GPT-2 language model architecture consisting of self-attention mechanisms, transformer blocks, and embedding layers.
   - Loads a pre-trained model from a file ("model-01.pkl").
   - Repeatedly prompts the user for input and generates text based on the input using the pre-trained model.
   - The generated text serves as a creative text completion or continuation based on the input provided by the user.

   This code can be used for text generation, creative writing, or interactive storytelling by providing prompts and receiving model-generated responses. It offers an engaging way to interact with a language model and receive text completions.
